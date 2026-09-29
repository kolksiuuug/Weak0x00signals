"""Загрузка и нормализация датасета организаторов этапа 1 (CLAUDE.md §1).

Ожидаемый лист: «Слабые сигналы», 100 строк, колонки:
    Технология | Область | Компании | Почему это слабый сигнал |
    Стадия развития | Тренд упоминаний | Балл (стадия+тренд) | Источники

Что делает модуль:
  * находит файл и лист (нечёткий поиск, чтобы мелкие расхождения не ломали пайплайн);
  * сопоставляет колонки по нормализованному имени;
  * переводит порядковые текстовые шкалы в числа по таблице §1:
        стадия: Концепция/Исследование=1, Прототип/PoC=2, Пилот=3, Раннее внедрение=4
        тренд:  стабильно/иное=1, растёт=2, растёт быстро=3
  * разбирает markdown-ссылки вида [название](url) и голые URL в колонке «Источники»;
  * проверяет инсайт §1: Балл == стадия + тренд, и сообщает долю совпадений.

Модуль НЕ знает про sklearn — он только приводит данные к таблице. Признаки строит
ml/app/features.py, поэтому один и тот же код работает и на датасете, и на SignalDoc.

Python 3.8+: аннотации через typing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("ml.dataset")

# --- имена, которые ищем ----------------------------------------------------

SHEET_CANDIDATES = ("слабые сигналы", "signals", "sheet1", "лист1")

#: нормализованное имя колонки -> каноническое имя в нашем DataFrame
COLUMN_ALIASES: Dict[str, str] = {
    "технология": "technology",
    "название": "technology",
    "названиетехнологии": "technology",
    "область": "area",
    "направление": "area",
    "компании": "companies",
    "игроки": "companies",
    "почемуэтослабыйсигнал": "why_raw",
    "почемуслабыйсигнал": "why_raw",
    "обоснование": "why_raw",
    "стадияразвития": "stage_raw",
    "стадия": "stage_raw",
    "трендупоминаний": "trend_raw",
    "тренд": "trend_raw",
    "баллстадиятренд": "points",
    "балл": "points",
    "оценка": "points",
    "источники": "sources_raw",
    "ссылки": "sources_raw",
}

#: Длина общего префикса, при которой заголовок считается совпавшим с алиасом.
#: 4 символа достаточно: префиксы всех алиасов выше на этой длине различаются
#: («техн», «назв», «обла», «напр», «комп», «игро», «поче», «стад», «трен», «балл»,
#: «оцен», «исто», «ссыл»).
PREFIX_MATCH_LEN = 4

#: Сколько первых строк просматриваем в поисках шапки таблицы.
HEADER_SCAN_ROWS = 10
#: Сколько опознанных колонок в строке достаточно, чтобы считать её заголовками.
HEADER_MIN_MATCHES = 3

CANONICAL_COLUMNS = (
    "technology", "area", "companies", "why_raw",
    "stage_raw", "trend_raw", "points", "sources_raw",
)

# --- порядковые шкалы из CLAUDE.md §1 ---------------------------------------

STAGE_LABELS: Dict[int, str] = {
    1: "Концепция/Исследование",
    2: "Прототип/PoC",
    3: "Пилот",
    4: "Раннее внедрение",
}
TREND_LABELS: Dict[int, str] = {
    1: "упоминания стабильны",
    2: "упоминания растут",
    3: "упоминания растут быстро",
}

STAGE_MIN, STAGE_MAX = 1, 4
TREND_MIN, TREND_MAX = 1, 3
POINTS_MIN, POINTS_MAX = STAGE_MIN + TREND_MIN, STAGE_MAX + TREND_MAX  # 3..7

#: Порядок важен: более специфичные шаблоны проверяются первыми
#: («раннее внедрение» раньше «внедрение», «растет быстро» раньше «растет»).
STAGE_PATTERNS: Tuple[Tuple[str, int], ...] = (
    # Шкала §1 упирается в 4 («Раннее внедрение»), поэтому зрелые стадии из контрольной
    # выборки насыщаются на том же 4. Отличать «раннее внедрение» от «массового» —
    # работа признака maturity_markers, а не порядковой шкалы.
    # Шаблоны намеренно узкие: одиночное слово «стандарт» сюда не берём, иначе фраза
    # «нет отраслевого стандарта» из текста слабого сигнала утянет стадию в 4.
    (r"массов\w*\s+внедрен|массов\w*\s+использован|промышленн\w*\s+эксплуатац", 4),
    (r"серийн\w*\s+произв|зрел\w*\s+технолог|зрел\w*\s+стади", 4),
    # «ранн\w*», а не «ранне\w*»: в датасете встречается «Ранние внедрения»
    (r"ранн\w*\s+внедрен", 4),
    (r"ранн\w*\s+адопц", 4),
    (r"коммерч\w*\s+внедрен", 4),
    (r"early\s+adopt", 4),
    (r"пилот", 3),
    (r"pilot", 3),
    (r"опытн\w*\s+эксплуатац", 3),
    (r"прототип", 2),
    (r"\bpoc\b", 2),
    (r"proof\s+of\s+concept", 2),
    (r"мвп|\bmvp\b", 2),
    (r"концепц", 1),
    (r"исследован", 1),
    (r"research|concept", 1),
    (r"лаборатор", 1),
)

TREND_PATTERNS: Tuple[Tuple[str, int], ...] = (
    (r"растет\s+быстро|быстро\s+растет|резк\w*\s+рост|взрывн\w*\s+рост", 3),
    (r"быстр\w*\s+рост", 3),
    (r"растет|рост|увеличива", 2),
    (r"стабильн|без\s+изменен|ровн|плато", 1),
)

#: markdown-ссылка [название](url) — основной формат колонки «Источники»
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
#: голый URL, если markdown не использован
BARE_URL_RE = re.compile(r"https?://[^\s,;|)\]]+")


# --- вспомогательное --------------------------------------------------------

def _norm_text(value: object) -> str:
    """Нижний регистр, ё → е, схлопнутые пробелы. Основа всех нечётких сравнений."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\s+", " ", text).strip().lower()


def _norm_colname(value: object) -> str:
    """Имя колонки без пробелов и пунктуации: «Балл (стадия+тренд)» -> «баллстадиятренд»."""
    return re.sub(r"[^0-9a-zа-я]+", "", _norm_text(value))


#: Разделитель между меткой и её обоснованием: «Растёт — запуск платформы в июне…»,
#: «Стабильный: поток публикаций ровный», «Ранние внедрения (ограниченный доступ)».
JUSTIFICATION_RE = re.compile(r"\s*[—–:(]|\s+-\s+")
#: Стрелка означает ПЕРЕХОД между стадиями, а не обоснование: «Исследование → Прототип/PoC».
#: Датасет в таких случаях засчитывает целевую стадию, то есть правую часть.
ARROW_RE = re.compile(r"\s*(?:→|->|=>)\s*")
#: Длиннее этого метка не бывает — дальше идёт свободный текст, а не подпись шкалы.
LABEL_MAX_LEN = 60


def _label_candidates(norm: str) -> List[str]:
    """Кусочки ячейки в порядке убывания надёжности.

    Ячейки «Стадия развития» и «Тренд упоминаний» в датасете организаторов устроены как
    «метка — обоснование», а метка может быть переходом «A → B». Обоснование содержит
    слова вроде «быстрый рост», которые ошибочно поднимают шкалу («Стабильный с
    ускорением — … быстрый рост …» давало 3 вместо 1), поэтому его отрезаем.
    """
    head = norm
    match = JUSTIFICATION_RE.search(norm)
    if match and match.start() > 0:
        candidate = norm[:match.start()].strip()
        if candidate and len(candidate) <= LABEL_MAX_LEN:
            head = candidate

    out: List[str] = []
    parts = [part.strip() for part in ARROW_RE.split(head) if part.strip()]
    if len(parts) > 1:
        out.append(parts[-1])       # переход: актуальна правая часть
    if head:
        out.append(head)
    return out


def _ordinal_from_text(text: str, patterns: Tuple[Tuple[str, int], ...], default: int) -> int:
    """Первый сработавший шаблон задаёт значение. Порядок шаблонов = приоритет.

    Сначала смотрим метку до разделителя, затем — весь текст. Второй проход нужен для
    этапа 2, где стадию и тренд приходится выводить из свободного текста источника.
    """
    norm = _norm_text(text)
    if not norm:
        return default
    # число прямо в ячейке («3») считаем уже закодированным
    direct = re.fullmatch(r"\s*([1-9])\s*", norm)
    if direct:
        return int(direct.group(1))

    for candidate in _label_candidates(norm) + [norm]:
        for pattern, value in patterns:
            if re.search(pattern, candidate):
                return value
    return default


def encode_stage(text: object) -> int:
    """Текст стадии -> 1..4 (CLAUDE.md §1). Неизвестное -> 1 (самая ранняя, консервативно)."""
    value = _ordinal_from_text(str(text), STAGE_PATTERNS, default=STAGE_MIN)
    return max(STAGE_MIN, min(STAGE_MAX, value))


def encode_trend(text: object) -> int:
    """Текст тренда -> 1..3 (CLAUDE.md §1). «стабильно / иное» -> 1."""
    value = _ordinal_from_text(str(text), TREND_PATTERNS, default=TREND_MIN)
    return max(TREND_MIN, min(TREND_MAX, value))


def parse_sources(cell: object) -> List[Dict[str, str]]:
    """Ячейка «Источники» -> список {title, url}.

    Поддерживает markdown-ссылки, голые URL и смесь того и другого,
    разделённые переводом строки, запятой, точкой с запятой или «|».
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    text = str(cell)
    out: List[Dict[str, str]] = []
    seen = set()

    for title, url in MD_LINK_RE.findall(text):
        if url not in seen:
            seen.add(url)
            out.append({"title": title.strip() or url, "url": url})

    # голые URL, не попавшие в markdown-разбор
    without_md = MD_LINK_RE.sub(" ", text)
    for url in BARE_URL_RE.findall(without_md):
        url = url.rstrip(".,;")
        if url not in seen:
            seen.add(url)
            out.append({"title": url, "url": url})

    return out


def parse_companies(cell: object) -> List[str]:
    """Ячейка «Компании» -> список названий. Разделители: запятая, «;», «|», перевод строки."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    parts = re.split(r"[,;|\n]+", str(cell))
    out = []
    for part in parts:
        name = part.strip(" . ")
        # отсекаем мусорные хвосты вида «и др.», «etc.»
        if name and _norm_text(name) not in {"и др", "др", "etc", "others", "-", "n/a", "нет"}:
            out.append(name)
    return out


# --- поиск файла и листа ----------------------------------------------------

def find_dataset_file(raw_dir: Path) -> Path:
    """Первый .xlsx/.xls/.csv в каталоге. Явный путь всегда важнее автопоиска."""
    if not raw_dir.exists():
        raise FileNotFoundError(
            "Каталог {} не найден. Положите датасет организаторов в data/raw/ "
            "(см. data/README.md).".format(raw_dir)
        )
    candidates = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".xlsx", ".xls", ".csv"}
        and not p.name.startswith("~$")
    )
    if not candidates:
        raise FileNotFoundError(
            "В {} нет файла датасета (.xlsx/.xls/.csv). Положите файл организаторов "
            "(лист «Слабые сигналы») — см. data/README.md.".format(raw_dir)
        )
    if len(candidates) > 1:
        logger.warning("В %s несколько файлов, беру первый: %s", raw_dir, candidates[0].name)
    return candidates[0]


def _pick_sheet(path: Path) -> object:
    """Имя листа «Слабые сигналы» или первый лист, если такого нет."""
    try:
        sheets = pd.ExcelFile(path).sheet_names
    except Exception as exc:  # noqa: BLE001 — формат неизвестен, деградируем на первый лист
        logger.warning("Не удалось прочитать список листов %s: %s", path.name, exc)
        return 0
    wanted = {_norm_colname(c) for c in SHEET_CANDIDATES}
    for name in sheets:
        if _norm_colname(name) in wanted:
            return name
    logger.warning("Лист «Слабые сигналы» не найден среди %s — беру первый (%s)", sheets, sheets[0])
    return sheets[0]


def _canonical_name(value: object) -> Optional[str]:
    """Каноническое имя колонки для заголовка, либо None.

    Сначала точное совпадение по нормализованному имени, затем совпадение по общему
    префиксу: «Технология (слабый сигнал)» -> technology, «Балл (стадия+тренд)» -> points.
    """
    key = _norm_colname(value)
    if not key:
        return None
    if key in COLUMN_ALIASES:
        return COLUMN_ALIASES[key]
    for alias, canonical in COLUMN_ALIASES.items():
        prefix = min(len(key), len(alias), PREFIX_MATCH_LEN)
        if prefix >= PREFIX_MATCH_LEN and key[:prefix] == alias[:prefix]:
            return canonical
    return None


def _detect_header_row(raw: pd.DataFrame) -> int:
    """Номер строки с заголовками таблицы.

    Реальный файл организаторов начинается со строки-баннера («100 слабых
    технологических сигналов...»), а настоящие заголовки идут следующей строкой.
    Поэтому шапку ищем: берём строку, где больше всего ячеек опознаётся как известные
    колонки. Это устойчивее, чем захардкоженный skiprows=1.
    """
    best_row, best_score = 0, 0
    for index in range(min(HEADER_SCAN_ROWS, len(raw))):
        score = sum(1 for value in raw.iloc[index] if _canonical_name(value))
        if score > best_score:
            best_row, best_score = index, score

    if best_score < HEADER_MIN_MATCHES:
        logger.warning(
            "Строка заголовков не опознана (лучшее совпадение: %d колонок в строке %d). "
            "Считаю заголовками первую строку.", best_score, best_row,
        )
        return 0
    if best_row:
        logger.info("Заголовки таблицы найдены в строке %d (опознано колонок: %d)",
                    best_row + 1, best_score)
    return best_row


def _read_raw(path: Path) -> pd.DataFrame:
    """Файл -> DataFrame с правильной шапкой, без строк-баннеров и пустых колонок."""
    if path.suffix.lower() == ".csv":
        # utf-8-sig, а не utf-8: Excel и PowerShell пишут CSV с BOM, и без этого
        # первый заголовок приезжает как "﻿Технология" и не сопоставляется с алиасом.
        raw = pd.read_csv(path, header=None, encoding="utf-8-sig", dtype=object)
    else:
        raw = pd.read_excel(path, sheet_name=_pick_sheet(path), header=None, dtype=object)

    if raw.empty:
        return raw

    header_row = _detect_header_row(raw)
    frame = raw.iloc[header_row + 1:].reset_index(drop=True)
    frame.columns = [str(value) for value in raw.iloc[header_row]]
    # колонки без заголовка (в файле организаторов первая колонка пустая) только мешают
    return frame.loc[:, [str(c) != "nan" for c in frame.columns]]


def _map_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Переименование колонок по нечёткому совпадению + отчёт о ненайденных."""
    renamed: Dict[object, str] = {}
    for column in frame.columns:
        canonical = _canonical_name(column)
        if canonical:
            renamed[column] = canonical
    frame = frame.rename(columns=renamed)

    missing = [c for c in CANONICAL_COLUMNS if c not in frame.columns]
    if missing:
        logger.warning(
            "В датасете не найдены колонки %s. Исходные заголовки: %s. "
            "Отсутствующие поля будут пустыми.", missing, list(frame.columns)
        )
        for column in missing:
            frame[column] = None
    return frame


# --- публичный вход ---------------------------------------------------------

def load_dataset(path: Optional[Path] = None, raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """Датасет организаторов -> нормализованный DataFrame.

    Колонки на выходе:
        technology, area, companies (list), why_raw, sources (list[dict]),
        stage_raw, trend_raw, stage (1..4), trend (1..3),
        points_expected (стадия+тренд), points (из файла), points_match (bool)
    """
    if path is None:
        path = find_dataset_file(raw_dir or Path("data/raw"))
    path = Path(path)
    logger.info("Читаю датасет: %s", path)

    frame = _map_columns(_read_raw(path))
    frame = frame.dropna(how="all").reset_index(drop=True)

    out = pd.DataFrame()
    out["technology"] = frame["technology"].fillna("").astype(str).str.strip()
    out["area"] = frame["area"].fillna("").astype(str).str.strip()
    out["companies"] = frame["companies"].apply(parse_companies)
    out["why_raw"] = frame["why_raw"].fillna("").astype(str).str.strip()
    out["sources"] = frame["sources_raw"].apply(parse_sources)
    out["stage_raw"] = frame["stage_raw"].fillna("").astype(str).str.strip()
    out["trend_raw"] = frame["trend_raw"].fillna("").astype(str).str.strip()
    out["stage"] = frame["stage_raw"].apply(encode_stage)
    out["trend"] = frame["trend_raw"].apply(encode_trend)
    out["points_expected"] = out["stage"] + out["trend"]
    out["points"] = pd.to_numeric(frame["points"], errors="coerce")
    out["points_match"] = out["points"].eq(out["points_expected"])

    # строки без названия технологии — это подписи и итоги, а не наблюдения
    out = out[out["technology"].str.len() > 0].reset_index(drop=True)
    logger.info("Загружено строк: %d", len(out))
    return out


def load_negatives(path: Path) -> pd.DataFrame:
    """Отрицательный класс: один CSV или каталог из нескольких (CLAUDE.md §0.2, §6 У2).

    Организаторы подтвердили, что негативов в датасете нет и команда собирает их сама,
    а Дима (У3) приносит кандидатов из парсера. Поэтому выборка складывается из
    нескольких файлов: ручная `mature_control.csv` рядом с выгрузками коллеги.

    Колонка `source_file` сохраняется: в отчёте должно быть видно, откуда взялся каждый
    отрицательный пример — метрика без происхождения негативов не интерпретируется.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.glob("*.csv") if not p.name.startswith("~$"))
        if not files:
            raise FileNotFoundError("В каталоге {} нет ни одного .csv".format(path))
    else:
        files = [path]

    frames = []
    for file_path in files:
        frame = load_dataset(file_path)
        frame["source_file"] = file_path.name
        logger.info("Отрицательный класс: %s -> %d строк", file_path.name, len(frame))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    # дедуп по названию: один и тот же зрелый кандидат мог прийти и вручную, и из парсера
    before = len(combined)
    combined = combined.drop_duplicates(subset=["technology"], keep="first").reset_index(drop=True)
    if before != len(combined):
        logger.info("Отрицательный класс: убрано дублей по названию: %d", before - len(combined))
    return combined


def insight_report(frame: pd.DataFrame) -> Dict[str, object]:
    """Проверка инсайта CLAUDE.md §1: Балл == стадия + тренд.

    Возвращает долю совпадений и примеры расхождений — это идёт в отчёт по методологии
    и защищает ключевое архитектурное решение (простая прозрачная модель, а не DL).
    """
    known = frame[frame["points"].notna()]
    if known.empty:
        return {"rows_with_points": 0, "match_rate": None, "mismatches": []}

    match_rate = float(known["points_match"].mean())
    mismatches = [
        {
            "technology": row["technology"],
            "stage_raw": row["stage_raw"],
            "trend_raw": row["trend_raw"],
            "stage": int(row["stage"]),
            "trend": int(row["trend"]),
            "points_expected": int(row["points_expected"]),
            "points_actual": int(row["points"]),
        }
        for _, row in known[~known["points_match"]].iterrows()
    ]
    return {
        "rows_with_points": int(len(known)),
        "match_rate": match_rate,
        "mismatches": mismatches,
    }
