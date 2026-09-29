"""Оценка выдачи по открытому запросу (CLAUDE.md §0.4).

Зачем отдельно от train.py. Организаторы проверяют решение не метрикой классификатора,
а **открытым запросом**: по 6 темам датасета выдача сверяется с самим датасетом, по
остальным — экспертно вручную. Значит числа, которые действительно защищают решение, —
это «сколько размеченных слабых сигналов темы система вернула в ТОП-15», а не F1 на
собственной выборке. Метрики этапа 1 самодекларируемые (§0.4), а эти — проверяемые.

Два режима:

  offline (по умолчанию) — пул кандидатов собирается из датасета организаторов и
      контрольной выборки, «запросом» выступает область. Проверяет ровно нашу часть:
      скоринг, фильтр зрелости и ранжирование. Парсер и стенд не нужны, работает всегда.

  live — POST /api/search на поднятый стенд, опрос до готовности, сверка заголовков
      выдачи с названиями из датасета. Проверяет весь путь целиком, включая поиск по
      живым источникам. Требует работающих сервисов Ромы и Димы.

Что считаем по каждой из 6 тем:
  * recall@15   — доля размеченных слабых сигналов темы, попавших в ТОП-15;
  * precision@15 — доля позиций ТОП-15, оказавшихся размеченными слабыми сигналами;
  * негативов в ТОП — сколько зрелых и хайповых просочилось (offline);
  * отклонено фильтром — сколько кандидатов отсеяно и по каким правилам.

Запуск:
    python -m ml.app.evaluate                       # offline по всем 6 темам
    python -m ml.app.evaluate --area Финтех         # одна тема
    python -m ml.app.evaluate --mode live --api-url http://localhost
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from ml.app import dataset as ds
from ml.app import maturity, ranking
from ml.app.features import observation_from_doc
from ml.app.model import WeakSignalScorer, utc_now_iso
from ml.app.source_meta import classify_url, guess_language
from shared.contracts import TOP_N
from shared.schema import SignalDoc, Source

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("ml.evaluate")

DEFAULT_CONTROL = Path("data/control")
DEFAULT_REPORTS = Path("ml/reports")

#: Порог совпадения названий при сверке живой выдачи с датасетом.
#: Заголовок из парсера почти никогда не совпадает с названием в датасете дословно,
#: поэтому сравниваем по пересечению значимых слов (мера Жаккара).
TITLE_MATCH_THRESHOLD = 0.45

#: Слова, которые не несут смысла при сверке названий.
STOPWORDS = frozenset({
    "и", "в", "на", "для", "с", "по", "из", "к", "о", "об", "при", "как", "что",
    "the", "a", "an", "of", "for", "in", "on", "to", "and", "with",
    "технология", "технологии", "система", "системы", "решение", "решения",
    "платформа", "сервис", "метод", "подход",
})


# --- сопоставление названий -------------------------------------------------

def _tokens(title: str) -> Set[str]:
    """Значимые слова названия: нижний регистр, ё -> е, без стоп-слов и коротышей."""
    text = (title or "").replace("ё", "е").lower()
    words = re.findall(r"[0-9a-zа-я]+", text)
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def title_similarity(left: str, right: str) -> float:
    """Мера Жаккара по значимым словам: 0.0 — ничего общего, 1.0 — полное совпадение."""
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def match_reference(candidate: str, references: Sequence[str]) -> Optional[str]:
    """Название из датасета, которому соответствует кандидат, либо None."""
    best_title, best_score = None, 0.0
    for reference in references:
        score = title_similarity(candidate, reference)
        if score > best_score:
            best_title, best_score = reference, score
    return best_title if best_score >= TITLE_MATCH_THRESHOLD else None


# --- строка датасета -> SignalDoc -------------------------------------------

def row_to_doc(row, doc_id: str) -> SignalDoc:
    """Строка нормализованного датасета -> SignalDoc по контракту §3.

    Метаданные источника восстанавливаются из домена: в датасете есть только
    название и ссылка, а контракт §2.7 требует тип, язык и доверенность.
    Источники с неразборчивым URL отбрасываются — SignalDoc не примет битый HttpUrl.
    """
    sources: List[Source] = []
    for item in (row.get("sources") or []):
        url = item.get("url", "")
        title = item.get("title") or url
        source_type, trust_level = classify_url(url)
        try:
            sources.append(Source(
                title=title,
                url=url,
                date=None,                      # в датасете организаторов даты нет
                source_type=source_type,
                language=guess_language(title, url),
                trust_level=trust_level,
            ))
        except Exception:                       # noqa: BLE001 — битый URL не должен ронять прогон
            logger.debug("Пропущен источник с неразборчивым URL: %r", url)

    return SignalDoc(
        id=doc_id,
        title=str(row.get("technology") or ""),
        area=(str(row.get("area") or "").strip() or None),
        companies=list(row.get("companies") or []),
        raw_text=str(row.get("why_raw") or ""),
        stage=int(row["stage"]),
        trend=int(row["trend"]),
        sources=sources,
    )


def build_pool(positives: pd.DataFrame, negatives: pd.DataFrame) -> Tuple[List[SignalDoc], Set[str]]:
    """Пул кандидатов offline-режима + множество названий настоящих слабых сигналов."""
    docs: List[SignalDoc] = []
    truth: Set[str] = set()

    for index, row in positives.iterrows():
        doc = row_to_doc(row, "pos-{}".format(index))
        docs.append(doc)
        truth.add(doc.title)

    for index, row in negatives.iterrows():
        docs.append(row_to_doc(row, "neg-{}".format(index)))

    return docs, truth


# --- offline-режим ----------------------------------------------------------

def _norm_area(area: Optional[str]) -> str:
    return (area or "").replace("ё", "е").strip().lower()


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Коэффициент Спирмена без scipy (его нет в ml/requirements.txt).

    Пирсон по средним рангам: связки получают средний ранг, иначе «Балл» из пяти
    возможных значений (3..7) даёт систематическое смещение.
    """
    if len(xs) < 3:
        return None

    def ranks(values: Sequence[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(num / (dx * dy), 4) if dx and dy else None


def order_agreement(returned_titles: Sequence[str],
                    points_by_title: Dict[str, int]) -> Tuple[Optional[float], int]:
    """Согласован ли ПОРЯДОК выдачи с «Баллом» заказчика (CLAUDE.md §0.3, §1).

    recall@15 и precision@15 измеряют состав выдачи. Но сдаём мы ранжированный
    список, и жюри смотрит именно порядок: если на первой позиции стоит наименее
    интересный из пятнадцати, состав уже не спасает.

    Внешняя истина про порядок в датасете одна — колонка «Балл (стадия+тренд)».
    Позиция 1 лучшая, поэтому согласованность = ОТРИЦАТЕЛЬНАЯ корреляция позиции
    с баллом. Возвращаем (rho, сколько позиций удалось сопоставить).
    """
    positions: List[float] = []
    points: List[float] = []
    for position, title in enumerate(returned_titles, start=1):
        value = points_by_title.get(title)
        if value is not None:
            positions.append(float(position))
            points.append(float(value))
    return spearman(positions, points), len(positions)


def evaluate_area_offline(area: str,
                          pool: Sequence[SignalDoc],
                          truth: Set[str],
                          scorer: WeakSignalScorer,
                          limit: int,
                          points_by_title: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Одна тема: ранжируем пул этой темы и сверяем ТОП-N с разметкой."""
    wanted = _norm_area(area)
    candidates = [doc for doc in pool if _norm_area(doc.area) == wanted]
    expected = {doc.title for doc in candidates if doc.title in truth}

    results, rejected = ranking.rank(candidates, scorer, limit=limit, query=area)
    returned = [doc.title for doc in results]
    hits = [title for title in returned if title in expected]
    leaked = [title for title in returned if title not in truth]

    # Статистика по правилам отклонения. rejected_reason — текст для человека,
    # поэтому машинный код правила восстанавливаем повторной проверкой. Дубликаты
    # отсеиваются до фильтра зрелости, их отличаем по тексту причины.
    rules: Dict[str, int] = {}
    for doc in rejected:
        if "дубликат" in (doc.rejected_reason or "").lower():
            key = "duplicate"
        else:
            key = maturity.check(observation_from_doc(doc)).rule or "other"
        rules[key] = rules.get(key, 0) + 1

    rho, matched = order_agreement(returned, points_by_title or {})

    return {
        "area": area,
        "candidates": len(candidates),
        "expected_weak": len(expected),
        "returned": len(returned),
        "hits": len(hits),
        "recall": round(len(hits) / len(expected), 4) if expected else None,
        "precision": round(len(hits) / len(returned), 4) if returned else None,
        "negatives_in_top": len(leaked),
        "rejected": len(rejected),
        "rejected_rules": rules,
        "missed": sorted(expected - set(returned))[:10],
        "leaked_titles": leaked[:10],
        # согласованность порядка с «Баллом» заказчика: ближе к -1 — лучше
        "order_rho": rho,
        "order_matched": matched,
    }


# --- live-режим -------------------------------------------------------------

def evaluate_area_live(area: str,
                       expected_titles: Sequence[str],
                       api_url: str,
                       limit: int,
                       timeout: float,
                       poll_interval: float) -> Dict[str, Any]:
    """Одна тема через реальный стенд: POST /api/search -> опрос -> сверка заголовков.

    Сетевые отказы не выпускаем наружу трейсбеком. Причина не в аккуратности: темы
    оцениваются по очереди, и упавший запрос по одной теме не должен уносить весь
    прогон — итог собирается по темам, которые ответили (см. фильтр valid в main).
    Отдельно это важно на защите: недоступный стенд должен давать строку «стенд не
    ответил», а не сто строк httpx в консоли.
    """
    import httpx

    base = api_url.rstrip("/")
    payload: Dict[str, Any] = {}
    try:
        with httpx.Client(timeout=timeout) as client:
            accepted = client.post(
                base + "/api/search",
                json={"query": area, "area": area, "limit": limit},
            )
            accepted.raise_for_status()
            job_id = accepted.json()["job_id"]

            deadline = time.time() + timeout
            while time.time() < deadline:
                response = client.get(base + "/api/search/{}".format(job_id))
                response.raise_for_status()
                payload = response.json()
                if payload.get("status") in {"готово", "ошибка"}:
                    break
                time.sleep(poll_interval)
    except httpx.HTTPStatusError as exc:
        return {"area": area, "error": "Стенд ответил {} на {}.".format(
            exc.response.status_code, exc.request.url)}
    except httpx.HTTPError as exc:
        return {"area": area, "error": "Стенд недоступен ({}): {}.".format(
            type(exc).__name__, exc)}
    except (KeyError, ValueError) as exc:
        # ValueError покрывает и json.JSONDecodeError: ответ не тот, что ждёт контракт.
        return {"area": area, "error": "Стенд вернул неожидаемый ответ: {}.".format(exc)}

    if payload.get("status") != "готово":
        return {
            "area": area,
            "error": "Стенд не вернул результат за {:.0f} с (статус: {}).".format(
                timeout, payload.get("status") or "нет ответа"),
        }

    returned = [item.get("title", "") for item in payload.get("results", [])]
    matched = {}
    for title in returned:
        reference = match_reference(title, expected_titles)
        if reference:
            matched[reference] = title

    return {
        "area": area,
        "expected_weak": len(expected_titles),
        "returned": len(returned),
        "hits": len(matched),
        "recall": round(len(matched) / len(expected_titles), 4) if expected_titles else None,
        "precision": round(len(matched) / len(returned), 4) if returned else None,
        "rejected": len(payload.get("rejected", [])),
        "stats": payload.get("stats", {}),
        "matched_pairs": [{"датасет": k, "выдача": v} for k, v in list(matched.items())[:10]],
        "missed": sorted(set(expected_titles) - set(matched))[:10],
    }


# --- отчёт ------------------------------------------------------------------

def _pct(value: Optional[float]) -> str:
    return "—" if value is None else "{:.1f} %".format(value * 100)


def write_report(report: Dict[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "open_query_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = report["areas"]
    lines: List[str] = [
        "# Оценка выдачи по открытому запросу",
        "",
        "Сгенерировано: `python -m ml.app.evaluate`. Режим: **{}**. Дата: {}.".format(
            report["mode"], report["generated_at"]),
        "",
        "Организаторы проверяют решение открытым запросом: по 6 темам датасета выдача "
        "сверяется с самим датасетом (CLAUDE.md §0.4). Эти метрики, в отличие от F1 "
        "классификатора, заказчик может воспроизвести.",
        "",
    ]

    if report["mode"] == "offline":
        lines += [
            "**Режим offline.** Пул кандидатов — датасет организаторов плюс контрольная "
            "выборка зрелых и хайповых технологий. Проверяется наша часть: скоринг, фильтр "
            "зрелости и ранжирование. Живой поиск по источникам сюда не входит.",
            "",
            "| Тема | Кандидатов | Слабых в теме | В ТОП-{} | Recall | Precision | Порядок (rho) | Негативов в ТОП | Отклонено |".format(report["limit"]),
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            if "error" in row:
                continue
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["area"], row["candidates"], row["expected_weak"], row["returned"],
                _pct(row["recall"]), _pct(row["precision"]),
                "—" if row.get("order_rho") is None else "{:+.3f}".format(row["order_rho"]),
                row["negatives_in_top"], row["rejected"]))
    else:
        lines += [
            "**Режим live.** Запросы уходят на поднятый стенд через `POST /api/search`, "
            "заголовки выдачи сверяются с названиями из датасета по пересечению значимых "
            "слов (порог {:.2f}).".format(TITLE_MATCH_THRESHOLD),
            "",
            "| Тема | Слабых в теме | В выдаче | Совпало | Recall | Precision | Отклонено |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            if "error" in row:
                lines.append("| {} | — | — | — | — | — | ошибка |".format(row["area"]))
                continue
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                row["area"], row["expected_weak"], row["returned"], row["hits"],
                _pct(row["recall"]), _pct(row["precision"]), row["rejected"]))

    totals = report["totals"]
    lines += [
        "",
        "**Итого по всем темам:** recall {} , precision {} (микро-усреднение: "
        "совпадений {} из {} размеченных, выдано позиций {}).".format(
            _pct(totals["recall"]), _pct(totals["precision"]),
            totals["hits"], totals["expected"], totals["returned"]),
        "",
    ]

    if totals.get("order_rho") is not None:
        lines += [
            "### Согласованность порядка выдачи: rho = {:+.3f}".format(totals["order_rho"]),
            "",
            "Recall и precision измеряют **состав** ТОП-{}. Но сдаём мы ранжированный "
            "список, и если на первой позиции стоит наименее интересный из пятнадцати, "
            "правильный состав уже не спасает. Поэтому отдельно измеряется **порядок**: "
            "корреляция Спирмена между позицией в выдаче и «Баллом» датасета "
            "(стадия + тренд, CLAUDE.md §0.3, §1) — единственной внешней истиной про "
            "приоритет, которая у нас есть.".format(report["limit"]),
            "",
            "Позиция 1 — лучшая, поэтому согласованность означает **отрицательное** rho: "
            "−1.0 — порядок в точности повторяет оценку заказчика, 0 — порядок к ней "
            "отношения не имеет, положительное — порядок перевёрнут.",
            "",
            "Это число появилось не просто так. При прежней формуле ранга "
            "(`0.80 × уверенность + 0.20 × фактура`) rho составлял −0.04, то есть порядок "
            "был случайным относительно приоритета заказчика, а по теме Edge — прямо "
            "перевёрнутым (+0.61): наверх попадали кандидаты с самым низким баллом. "
            "Причина в том, что уверенность модели отвечает на вопрос «слабый ли это "
            "сигнал», а не «насколько он важен»: стадия концепции со стабильными "
            "упоминаниями выглядит максимально «слабой», хотя это балл 3 — низший "
            "приоритет. После добавления слагаемого приоритета "
            "(`0.50 / 0.30 / 0.20`) на текущем прогоне получено rho = {:+.3f}. "
            "Состав выдачи остаётся: recall@15 = {} и precision@15 = {}.".format(
                totals["order_rho"], _pct(totals["recall"]), _pct(totals["precision"])),
            "",
        ]

    errors = [row for row in rows if "error" in row]
    if errors:
        lines += ["## Ошибки", ""] + ["* **{}**: {}".format(r["area"], r["error"]) for r in errors] + [""]

    lines += ["## Что не попало в выдачу", ""]
    for row in rows:
        if "error" in row or not row.get("missed"):
            continue
        lines += ["**{}**:".format(row["area"]), ""]
        lines += ["* {}".format(title) for title in row["missed"]]
        lines.append("")

    if report["mode"] == "offline":
        leaked_any = [row for row in rows if row.get("leaked_titles")]
        if leaked_any:
            lines += [
                "## Негативы, просочившиеся в ТОП",
                "",
                "Это прямые ошибки фильтра и ранжирования — каждую стоит разобрать.",
                "",
            ]
            for row in leaked_any:
                lines += ["**{}**:".format(row["area"]), ""]
                lines += ["* {}".format(title) for title in row["leaked_titles"]]
                lines.append("")

    (reports_dir / "open_query_eval.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Отчёт записан: %s", reports_dir / "open_query_eval.md")


# --- точка входа ------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Оценка выдачи по открытому запросу (CLAUDE.md §0.4)")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--area", action="append", default=None,
                        help="Тема (можно повторять). По умолчанию — все темы из датасета")
    parser.add_argument("--limit", type=int, default=TOP_N)
    parser.add_argument("--api-url", default="http://localhost", help="Базовый URL стенда (режим live)")
    parser.add_argument("--timeout", type=float, default=180.0, help="Сколько ждать результат, с")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args(argv)

    positives = ds.load_dataset(args.dataset)
    if positives.empty:
        raise SystemExit("Датасет организаторов пуст — оценивать нечего.")

    areas = args.area or [a for a in positives["area"].dropna().unique() if str(a).strip()]
    logger.info("Тем к оценке: %d — %s", len(areas), ", ".join(map(str, areas)))

    rows: List[Dict[str, Any]] = []
    if args.mode == "offline":
        negatives = ds.load_negatives(args.control) if args.control.exists() else positives.iloc[0:0]
        scorer = WeakSignalScorer.load()
        if not scorer.is_trained:
            logger.warning(
                "Модель не обучена — оценка идёт на прозрачных правилах. "
                "Числа станут итоговыми после python -m ml.app.train."
            )
        pool, truth = build_pool(positives, negatives)
        # «Балл» заказчика по названию технологии — внешняя истина для проверки ПОРЯДКА
        points_by_title = {
            str(row["technology"]): int(row["points"])
            for _, row in positives.iterrows()
            if pd.notna(row["points"])
        }
        for area in areas:
            rows.append(evaluate_area_offline(str(area), pool, truth, scorer, args.limit,
                                              points_by_title))
    else:
        for area in areas:
            expected = [str(t) for t in positives.loc[
                positives["area"].map(_norm_area) == _norm_area(str(area)), "technology"]]
            rows.append(evaluate_area_live(
                str(area), expected, args.api_url, args.limit, args.timeout, args.poll_interval))

    valid = [row for row in rows if "error" not in row]
    total_hits = sum(row["hits"] for row in valid)
    total_expected = sum(row["expected_weak"] for row in valid)
    total_returned = sum(row["returned"] for row in valid)

    report = {
        "generated_at": utc_now_iso(),
        "mode": args.mode,
        "limit": args.limit,
        "areas": rows,
        "totals": {
            "hits": total_hits,
            "expected": total_expected,
            "returned": total_returned,
            "recall": round(total_hits / total_expected, 4) if total_expected else None,
            "precision": round(total_hits / total_returned, 4) if total_returned else None,
            # средняя согласованность порядка по темам, у которых её удалось посчитать
            "order_rho": None,
        },
    }

    rhos = [row["order_rho"] for row in valid if row.get("order_rho") is not None]
    if rhos:
        report["totals"]["order_rho"] = round(sum(rhos) / len(rhos), 4)

    for row in rows:
        if "error" in row:
            logger.error("%-22s %s", row["area"], row["error"])
        else:
            logger.info("%-22s recall=%s precision=%s порядок(rho)=%s (совпало %d из %d)",
                        row["area"], _pct(row["recall"]), _pct(row["precision"]),
                        "—" if row.get("order_rho") is None else "%+.3f" % row["order_rho"],
                        row["hits"], row["expected_weak"])
    logger.info("ИТОГО: recall=%s precision=%s порядок(rho)=%s",
                _pct(report["totals"]["recall"]), _pct(report["totals"]["precision"]),
                "—" if report["totals"]["order_rho"] is None
                else "%+.3f" % report["totals"]["order_rho"])

    write_report(report, args.reports)

    # Если не ответила ни одна тема, прогон считается неуспешным: отчёт с пустыми
    # метриками легко принять за результат, а это отсутствие результата.
    if not valid:
        logger.error(
            "Ни одна тема не оценена. В режиме live это почти всегда недоступный стенд: "
            "проверьте --api-url (сейчас %s) и что /api/search отвечает.", args.api_url)
        return 1
    return 0


def _cli() -> int:
    """Точка входа с человеческим сообщением вместо трейсбека (см. ml/app/train.py)."""
    try:
        return main()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(_cli())
