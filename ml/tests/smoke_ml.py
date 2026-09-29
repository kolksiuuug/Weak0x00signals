"""Дымовой прогон ML-модуля БЕЗ датасета организаторов и БЕЗ обученной модели.

Запуск из корня репозитория:
    python -m ml.tests.smoke_ml

Зачем отдельный скрипт, а не pytest: этот прогон должен работать сразу после
`pip install -r ml/requirements.txt`, на чистой машине, без дополнительных зависимостей
и без БД. Он проверяет ровно то, что можно проверить до появления датасета:
разбор шкал §1, восстановление метаданных источников, признаки, фильтр зрелости,
запасной режим скоринга и ранжирование ТОП-N.

Код возврата 0 — всё в порядке, 1 — есть провалившиеся проверки.
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ml.app import dataset as ds
from ml.app import lexicon, maturity, ranking
from ml.app.features import FEATURE_NAMES, build_features, observation_from_doc, observation_from_row
from ml.app.model import WeakSignalScorer
from ml.app.source_meta import classify_url, guess_language
from shared.schema import SignalDoc, Source, SourceType, TrustLevel

_failures: List[str] = []
_checks = 0


def check(condition: bool, description: str, detail: str = "") -> None:
    global _checks
    _checks += 1
    if condition:
        print("  [ок]    {}".format(description))
    else:
        print("  [ПРОВАЛ] {} {}".format(description, detail))
        _failures.append(description)


def section(title: str) -> None:
    print("\n{}\n{}".format(title, "-" * len(title)))


# --- фикстуры ---------------------------------------------------------------

def weak_doc() -> SignalDoc:
    """Правдоподобный слабый сигнал: ранняя стадия, растущий тренд, научная фактура."""
    return SignalDoc(
        id="weak-1",
        title="Нейроморфные ускорители для промышленной дефектоскопии",
        area="Индустриальный ИИ",
        companies=["Innatera"],
        raw_text=(
            "Небольшая команда представила прототип нейроморфного ускорителя. "
            "Работы ведутся в лабораториях, публикации в статусе препринтов, "
            "коммерческих внедрений нет, нет отраслевого стандарта."
        ),
        stage=2,
        trend=3,
        sources=[
            Source(title="Neuromorphic inference preprint", url="https://arxiv.org/abs/2401.00001",
                   date="2026-01-10", source_type=SourceType.SCIENTIFIC, language="en",
                   trust_level=TrustLevel.HIGH, translated=True),
            Source(title="Патент на архитектуру SNN", url="https://patents.google.com/patent/US11000000B2",
                   date="2026-03-01", source_type=SourceType.PATENT, language="en",
                   trust_level=TrustLevel.HIGH),
        ],
    )


def mature_doc() -> SignalDoc:
    """Зрелая технология: должна быть отклонена фильтром §2.10."""
    return SignalDoc(
        id="mature-1",
        title="Оркестрация контейнеров Kubernetes",
        area="Инфраструктура ИИ",
        companies=["Google", "Red Hat", "VMware", "Amazon", "Microsoft", "Oracle", "IBM", "SUSE"],
        raw_text=(
            "Отраслевой стандарт де-факто, массовое внедрение в промышленной эксплуатации, "
            "сформированный рынок с доминирующими игроками."
        ),
        stage=4,
        trend=1,
        sources=[
            Source(title="Kubernetes Documentation", url="https://kubernetes.io/docs/home/",
                   source_type=SourceType.NEWS, language="en", trust_level=TrustLevel.MEDIUM),
        ],
    )


def social_only_doc() -> SignalDoc:
    """Единственное основание — соцсеть: отклоняется по §2.8."""
    return SignalDoc(
        id="social-1",
        title="Революционный ИИ-стартап из треда",
        area="Финтех",
        companies=[],
        raw_text="Обсуждают все, тренд в соцсетях, изменит всё.",
        stage=1, trend=3,
        sources=[
            Source(title="Тред", url="https://x.com/someone/status/1",
                   source_type=SourceType.SOCIAL, language="ru", trust_level=TrustLevel.LOW),
        ],
    )


def low_confidence_doc() -> SignalDoc:
    """Проходит фильтр зрелости, но уверенность модели низкая: в ТОП попадать не должен."""
    return SignalDoc(
        id="low-1",
        title="Платформа оркестрации складских процессов",
        area="Роботы",
        companies=["Альфа", "Бета", "Гамма", "Дельта", "Эпсилон"],
        raw_text="Решение внедряется несколькими интеграторами, динамика упоминаний ровная.",
        stage=3, trend=1,
        sources=[
            Source(title="Обзор рынка", url="https://www.rbc.ru/technology/1",
                   source_type=SourceType.NEWS, language="ru", trust_level=TrustLevel.MEDIUM),
            Source(title="Аналитика", url="https://www.tadviser.ru/report/1",
                   source_type=SourceType.ANALYTICS, language="ru", trust_level=TrustLevel.MEDIUM),
        ],
    )


def no_sources_doc() -> SignalDoc:
    return SignalDoc(id="empty-1", title="Технология без источников", raw_text="Ничего не подтверждено.",
                     stage=1, trend=2, sources=[])


# --- проверки ---------------------------------------------------------------

def test_scales() -> None:
    section("1. Порядковые шкалы CLAUDE.md §1")
    check(ds.encode_stage("Концепция/Исследование") == 1, "стадия «Концепция/Исследование» -> 1")
    check(ds.encode_stage("Прототип/PoC") == 2, "стадия «Прототип/PoC» -> 2")
    check(ds.encode_stage("Пилот") == 3, "стадия «Пилот» -> 3")
    check(ds.encode_stage("Раннее внедрение") == 4, "стадия «Раннее внедрение» -> 4")
    check(ds.encode_stage("Массовое внедрение") == 4, "стадия «Массовое внедрение» насыщается на 4")
    check(ds.encode_trend("стабильно") == 1, "тренд «стабильно» -> 1")
    check(ds.encode_trend("растёт") == 2, "тренд «растёт» -> 2")
    check(ds.encode_trend("растет быстро") == 3, "тренд «растет быстро» -> 3 (без ё тоже)")
    check(ds.encode_trend("") == 1, "пустой тренд -> 1 («иное»)")
    check(ds.encode_stage("нет отраслевого стандарта") == 1,
          "«нет отраслевого стандарта» НЕ утягивает стадию в зрелость")


def test_sources_parsing() -> None:
    section("2. Разбор колонки «Источники»")
    cell = "[arXiv](https://arxiv.org/abs/2401.1), https://www.nature.com/articles/x"
    parsed = ds.parse_sources(cell)
    check(len(parsed) == 2, "markdown + голый URL разобраны", str(parsed))
    check(parsed[0]["title"] == "arXiv", "название из markdown сохранено")
    check(ds.parse_sources(None) == [], "пустая ячейка -> пустой список")

    companies = ds.parse_companies("Innatera, SynSense; и др.")
    check(companies == ["Innatera", "SynSense"], "«и др.» отброшено", str(companies))


def test_source_meta() -> None:
    section("3. Тип и доверенность источника по домену (§2.7, §2.8)")
    cases = [
        ("https://arxiv.org/abs/1", SourceType.SCIENTIFIC, TrustLevel.HIGH),
        ("https://patents.google.com/patent/US1", SourceType.PATENT, TrustLevel.HIGH),
        ("https://x.com/a/status/1", SourceType.SOCIAL, TrustLevel.LOW),
        ("https://habr.com/ru/post/1", SourceType.BLOG, TrustLevel.LOW),
        ("https://www.prnewswire.com/news/1", SourceType.PRESS, TrustLevel.LOW),
        ("https://www.gartner.com/en/x", SourceType.ANALYTICS, TrustLevel.MEDIUM),
        ("https://www.cbr.ru/psystem/", SourceType.REGISTRY, TrustLevel.HIGH),
        # журнал при научном обществе — не рецензируемая площадка
        ("https://spectrum.ieee.org/robotics", SourceType.NEWS, TrustLevel.MEDIUM),
        ("https://ieeexplore.ieee.org/document/1", SourceType.SCIENTIFIC, TrustLevel.HIGH),
    ]
    for url, expected_type, expected_trust in cases:
        stype, trust = classify_url(url)
        check(stype == expected_type and trust == expected_trust,
              "{} -> {} / {}".format(url.split("/")[2], expected_type.value, expected_trust.value),
              "получено {} / {}".format(stype.value, trust.value))


def test_features() -> None:
    section("4. Признаки")
    values = build_features(observation_from_doc(weak_doc()))
    check(list(values.keys()) == list(FEATURE_NAMES), "порядок признаков совпадает с FEATURE_NAMES")
    check(all(isinstance(v, float) for v in values.values()), "все признаки — числа")
    check(values["has_scientific"] == 1.0 and values["has_patent"] == 1.0,
          "научная публикация и патент распознаны")
    check(values["share_high_trust"] == 1.0, "доля высокой доверенности = 1.0")
    check(values["weakness_markers"] > 0, "маркеры ранней стадии найдены в тексте")

    mature_values = build_features(observation_from_doc(mature_doc()))
    check(mature_values["maturity_markers"] > 0, "маркеры зрелости найдены в тексте зрелой технологии")

    # отрицание переворачивает смысл маркера: «нет отраслевого стандарта» — признак НЕзрелости
    check(values["maturity_markers"] == 0.0,
          "«нет отраслевого стандарта» НЕ засчитано как маркер зрелости",
          "получено {}".format(values["maturity_markers"]))
    check(lexicon.maturity_hits("нет отраслевого стандарта") == [],
          "отрицание перед маркером зрелости гасит его")
    check(lexicon.maturity_hits("закреплено отраслевым стандартом") != [],
          "без отрицания тот же маркер срабатывает")
    check(lexicon.maturity_hits("не имеет отраслевого стандарта") == [],
          "отрицание через два слова тоже видно")
    check(lexicon.weakness_hits("нет коммерческих внедрений") != [],
          "маркеры слабости, сформулированные через отрицание, сохраняются")

    # тот же документ без stage/trend — шкалы восстанавливаются из текста
    doc = weak_doc()
    doc.stage, doc.trend = None, None
    obs = observation_from_doc(doc)
    check(obs.stage_inferred and obs.trend_inferred, "пустые стадия и тренд помечены как выведенные")
    check(1 <= obs.stage <= 4 and 1 <= obs.trend <= 3, "выведенные шкалы попадают в допустимый диапазон")


def test_maturity_filter() -> None:
    section("5. Фильтр зрелости и хайпа (§2.10, §2.8, §2.3)")
    verdict = maturity.check(observation_from_doc(mature_doc()))
    check(verdict.rejected, "зрелая технология отклонена")
    check(bool(verdict.reason), "у отклонения есть текст причины (молчаливый фильтр запрещён)")
    check("зрел" in (verdict.reason or "").lower(), "причина называет зрелость", verdict.reason or "")

    verdict = maturity.check(observation_from_doc(social_only_doc()))
    check(verdict.rejected and verdict.rule == "low_trust_only",
          "единственный источник-соцсеть отклонён по §2.8", str(verdict.rule))

    verdict = maturity.check(observation_from_doc(no_sources_doc()))
    check(verdict.rejected and verdict.rule == "no_sources", "кандидат без источников отклонён по §2.3")

    verdict = maturity.check(observation_from_doc(weak_doc()))
    check(not verdict.rejected, "настоящий слабый сигнал проходит фильтр",
          verdict.reason or "")


def test_scoring_fallback() -> None:
    section("6. Скоринг (запасной режим — артефакт модели может отсутствовать)")
    scorer = WeakSignalScorer.load()
    print("  режим: {}".format("обученная модель" if scorer.is_trained else "прозрачные правила"))

    weak = scorer.score_document(weak_doc())
    mature = scorer.score_document(mature_doc())

    check(0.0 <= weak.score <= 1.0, "балл в диапазоне 0..1")
    check(weak.score > mature.score, "слабый сигнал получает балл выше зрелой технологии",
          "{:.3f} против {:.3f}".format(weak.score, mature.score))
    check(bool(weak.why), "объяснение непустое")
    check(len(weak.predictors) > 0, "предикторы перечислены")
    check(weak.id == "weak-1", "id документа сохранён в ответе")
    check(mature.rejected_reason is not None, "у зрелой технологии заполнена причина отклонения")
    check(not mature.is_weak_signal, "отклонённый кандидат не помечен как слабый сигнал")
    print("\n  Пример объяснения:\n  {}\n".format(weak.why))


def test_ranking() -> None:
    section("7. Ранжирование ТОП-N")
    scorer = WeakSignalScorer.load()
    duplicate = weak_doc()
    duplicate.id = "weak-1-dup"
    docs = [weak_doc(), mature_doc(), social_only_doc(), no_sources_doc(), duplicate]

    results, rejected = ranking.rank(docs, scorer, limit=15)
    check(len(results) >= 1, "в выдаче есть хотя бы один кандидат")
    check(all(doc.score is not None for doc in results), "у всех принятых проставлен балл")
    check(all(doc.rejected_reason for doc in rejected), "у всех отклонённых есть причина (§2.10)")
    check(any("дубликат" in (doc.rejected_reason or "") for doc in rejected),
          "дубликат по названию отсеян")
    # Сортировка идёт по итоговому ранг-баллу, а НЕ по одной уверенности: в ранг входит
    # ещё приоритет по «Баллу» заказчика и качество фактуры. Проверять сортировку по
    # doc.score было бы неверно — эта проверка раньше проходила случайно, на фикстуре,
    # где порядок по уверенности совпадал с порядком по ранга.
    ranks = [
        ranking.rank_score(doc.score or 0.0,
                           ranking.priority_score(observation_from_doc(doc)),
                           ranking.evidence_quality(doc))
        for doc in results
    ]
    check(ranks == sorted(ranks, reverse=True) or len(ranks) == 1,
          "выдача отсортирована по убыванию итогового ранга",
          str([round(r, 4) for r in ranks]))
    check(all("Позиция" in (doc.why or "") for doc in results),
          "в объяснении есть разложение ранга")
    check(all("приоритет" in (doc.why or "").lower() for doc in results),
          "объяснение называет слагаемое приоритета (иначе разложение не сходится с числом)")

    # приоритет по «Баллу» заказчика: границы шкалы §1
    concept = observation_from_doc(no_sources_doc())
    concept.stage, concept.trend = 1, 1
    early = observation_from_doc(weak_doc())
    early.stage, early.trend = 4, 3
    check(abs(ranking.priority_score(concept) - 0.0) < 1e-9,
          "балл 3 (концепция + стабильно) -> приоритет 0.0",
          str(ranking.priority_score(concept)))
    check(abs(ranking.priority_score(early) - 1.0) < 1e-9,
          "балл 7 (раннее внедрение + быстрый рост) -> приоритет 1.0",
          str(ranking.priority_score(early)))

    # при равной уверенности и фактуре выше должен стоять больший балл заказчика
    check(ranking.rank_score(0.9, 1.0, 0.5) > ranking.rank_score(0.9, 0.0, 0.5),
          "при равной уверенности выше идёт кандидат с большим «Баллом» заказчика")
    # но приоритет не должен перевешивать уверенность целиком: вес 0.30 против 0.50
    check(ranking.rank_score(0.95, 0.0, 0.5) > ranking.rank_score(0.30, 1.0, 0.5),
          "приоритет не перебивает уверенность: 0.30 против веса 0.50")

    # квота на область адаптивна: запрос по одной теме отдаёт всю выдачу этой теме
    check(ranking.area_quota(15, 1) == 15, "одна область -> вся выдача (15)")
    check(ranking.area_quota(15, 2) == 8, "две области -> по 8", str(ranking.area_quota(15, 2)))
    check(ranking.area_quota(15, 6) == 4, "шесть областей -> по 4 (нижняя граница)",
          str(ranking.area_quota(15, 6)))
    check(ranking.area_quota(15, 0) == 15, "пустой список областей не ломает квоту")

    # Кандидат ниже порога модели не попадает в ТОП и получает явную причину (§2.10).
    # Порог поднимаем искусственно: иначе проверка зависела бы от того, обучена модель
    # или работают запасные веса, и ломалась бы при каждом переобучении.
    # Документ подобран так, чтобы пройти фильтр зрелости — иначе проверялось бы не то.
    original_threshold = scorer.threshold
    scorer.threshold = 0.99
    try:
        _, rejected_below = ranking.rank([low_confidence_doc()], scorer, limit=15)
    finally:
        scorer.threshold = original_threshold
    check(len(rejected_below) == 1 and "порог" in (rejected_below[0].rejected_reason or ""),
          "кандидат ниже порога отклонён с причиной, называющей порог",
          rejected_below[0].rejected_reason if rejected_below else "не отклонён")


def test_control_dataset() -> None:
    section("8. Контрольная выборка отрицательного класса")
    path = Path("data/control")
    if not path.exists():
        check(False, "каталог data/control существует")
        return
    # тот же путь, что использует обучение: каталог целиком, все *.csv (CLAUDE.md §0.2)
    frame = ds.load_negatives(path)
    check(len(frame) >= 30, "строк не меньше 30", "получено {}".format(len(frame)))
    check("source_file" in frame.columns, "происхождение каждого негатива сохранено (source_file)")
    check(frame["technology"].str.len().gt(0).all(), "у всех строк заполнено название")

    source_counts = frame["sources"].apply(len)
    check(source_counts.gt(0).all(), "у каждой строки есть хотя бы один источник")
    check(source_counts.le(3).all(),
          "источников не больше 3 (иначе модель выучит счётчик источников)",
          "максимум {}".format(int(source_counts.max())))

    rejected = sum(1 for _, row in frame.iterrows()
                   if maturity.check(observation_from_row(row)).rejected)
    check(rejected >= len(frame) * 0.5,
          "фильтр зрелости сам ловит не меньше половины отрицательного класса",
          "поймано {} из {}".format(rejected, len(frame)))
    print("  фильтр зрелости поймал {} из {} отрицательных примеров".format(rejected, len(frame)))

    _check_source_distribution(source_counts)


#: Допустимое расхождение доли строк с тремя источниками между контролем и датасетом.
#: Когда контроль был собран равномерно (33 % против 86 % в датасете), признак
#: n_sources_log получил коэффициент +0.376 — модель выучила счётчик ссылок вместо
#: содержательных признаков. Порог держим узким, чтобы артефакт не вернулся молча
#: при добавлении новых parser_negatives_*.csv от У3.
SOURCE_SHARE_TOLERANCE = 0.15


def _check_source_distribution(control_counts) -> None:
    """Распределение числа источников в контроле должно совпадать с датасетом организаторов.

    Проверка осмысленна только когда датасет на месте: без него сравнивать не с чем.
    """
    try:
        positives = ds.load_dataset(ds.find_dataset_file(Path("data/raw")))
    except (FileNotFoundError, OSError) as exc:
        print("  распределение источников не сверено: датасета нет ({})".format(exc))
        return

    def share_of_three(counts) -> float:
        total = len(counts)
        return float(sum(1 for n in counts if n >= 3)) / total if total else 0.0

    want = share_of_three(positives["sources"].apply(len))
    have = share_of_three(control_counts)
    print("  доля строк с тремя источниками: контроль {:.1f} %, датасет {:.1f} %".format(
        have * 100, want * 100))
    check(abs(have - want) <= SOURCE_SHARE_TOLERANCE,
          "распределение числа источников сведено с датасетом (расхождение <= {:.0f} п.п.)".format(
              SOURCE_SHARE_TOLERANCE * 100),
          "расхождение {:.1f} п.п. — модель выучит счётчик источников".format(
              abs(have - want) * 100))


def test_closed_dataset_if_present() -> None:
    section("9. Датасет организаторов (если уже положен в data/raw)")
    try:
        path = ds.find_dataset_file(Path("data/raw"))
    except FileNotFoundError as exc:
        print("  пропущено: {}".format(exc))
        return

    frame = ds.load_dataset(path)
    check(len(frame) > 0, "датасет прочитан", "строк: {}".format(len(frame)))
    report = ds.insight_report(frame)
    if report["match_rate"] is None:
        print("  колонка «Балл» не найдена — проверку инсайта §1 пропускаю")
        return
    print("  инсайт §1 «Балл = стадия + тренд» воспроизводится на {:.1f} % строк".format(
        report["match_rate"] * 100))
    check(report["match_rate"] >= 0.9,
          "формула §1 воспроизводится не менее чем на 90 % строк",
          "{:.1f} %, расхождений {}".format(report["match_rate"] * 100, len(report["mismatches"])))


def _row_to_doc(row, idx: int) -> SignalDoc:
    """Строка датасета -> SignalDoc так, как его собрал бы парсер (У3).

    Нужна только для проверки train/serve skew ниже: сравнивать ветки можно лишь
    на одних и тех же данных.
    """
    sources = []
    for src in row["sources"]:
        url = src.get("url", "")
        stype, trust = classify_url(url)
        sources.append(Source(
            title=src.get("title") or url,
            url=url,
            source_type=stype,
            language=guess_language(src.get("title") or "", url),
            trust_level=trust,
        ))
    raw_text = " ".join(part for part in (str(row["why_raw"]), str(row["stage_raw"]),
                                         str(row["trend_raw"])) if part)
    return SignalDoc(
        id="row-{}".format(idx),
        title=str(row["technology"]),
        area=str(row["area"]) or None,
        companies=list(row["companies"]),
        raw_text=raw_text,
        stage=int(row["stage"]),
        trend=int(row["trend"]),
        sources=sources,
    )


def test_train_serve_skew() -> None:
    """Признаки из строки датасета и из SignalDoc должны совпадать до последнего знака.

    Это не формальность. Модель обучается на строках датасета, а применяется к
    SignalDoc из открытого пайплайна. Любое расхождение между ветками означает, что
    на живых документах модель работает вне своего распределения — и тогда метрики
    этапа 1 не переносятся на этап 2 вообще.
    """
    section("10. Train/serve skew: обе ветки признаков считают одно и то же")

    frames = [("контроль", ds.load_negatives(Path("data/control")))]
    try:
        frames.append(("датасет организаторов", ds.load_dataset(ds.find_dataset_file(Path("data/raw")))))
    except (FileNotFoundError, OSError):
        print("  датасета в data/raw нет — сверяю только на контрольной выборке")

    for label, frame in frames:
        worst_name, worst_delta, mismatched = "", 0.0, 0
        for idx, row in frame.iterrows():
            from_row = build_features(observation_from_row(row))
            from_doc = build_features(observation_from_doc(_row_to_doc(row, idx)))
            row_bad = False
            for name in FEATURE_NAMES:
                delta = abs(from_row[name] - from_doc[name])
                if delta > worst_delta:
                    worst_name, worst_delta = name, delta
                if delta > 1e-9:
                    row_bad = True
            mismatched += int(row_bad)
        check(mismatched == 0,
              "{}: {} строк, векторы признаков идентичны".format(label, len(frame)),
              "расходятся строк: {}, худший признак {} (дельта {:.6f})".format(
                  mismatched, worst_name, worst_delta))


#: Какой ENUM в SQL какому перечислению контракта соответствует.
SQL_ENUM_BINDINGS = (("source_type", SourceType), ("trust_level", TrustLevel))


def _sql_enum_values(sql: str, type_name: str) -> List[str]:
    """Значения `CREATE TYPE <type_name> AS ENUM (...)` из текста миграции.

    Разбор текстом, а не через БД: проверка должна работать на чистой машине,
    где PostgreSQL не поднят.
    """
    marker = "CREATE TYPE {} AS ENUM".format(type_name)
    start = sql.find(marker)
    if start < 0:
        return []
    open_paren = sql.find("(", start)
    close_paren = sql.find(")", open_paren)
    if open_paren < 0 or close_paren < 0:
        return []
    body = sql[open_paren + 1:close_paren]
    return [chunk.strip().strip("'") for chunk in body.split(",") if chunk.strip()]


def test_db_enums_match_contract() -> None:
    """ENUM в миграции обязан посимвольно совпадать с перечислением из shared/schema.py.

    Миграция утверждает это комментарием, но ничем не проверяет. Значения кириллические
    («научная_статья», «пониженный»), и расхождение в одну букву не ловится ни pydantic,
    ни PostgreSQL по отдельности: вставка просто падает в рантайме на живых данных.
    Схема БД — зона У2, а shared/schema.py правят все, поэтому сверка нужна автоматическая.
    """
    section("11. Схема БД: ENUM совпадают с контрактом §3")

    migration = Path("ml/db/migrations/001_core_schema.sql")
    if not migration.exists():
        check(False, "миграция 001_core_schema.sql на месте")
        return
    sql = migration.read_text(encoding="utf-8")

    for type_name, enum_cls in SQL_ENUM_BINDINGS:
        in_sql = _sql_enum_values(sql, type_name)
        in_py = [member.value for member in enum_cls]
        check(bool(in_sql), "{}: ENUM найден в миграции".format(type_name))
        if not in_sql:
            continue
        missing = [v for v in in_py if v not in in_sql]
        extra = [v for v in in_sql if v not in in_py]
        check(not missing and not extra,
              "{}: значения совпадают с {} ({} шт.)".format(type_name, enum_cls.__name__, len(in_py)),
              "нет в SQL: {} | лишние в SQL: {}".format(missing or "—", extra or "—"))


class _FakeCursor:
    """Курсор-заглушка для проверок раннера миграций без PostgreSQL.

    Нужна потому, что PostgreSQL в дымовом прогоне нет, а раннер миграций
    до сих пор не исполнялся ни разу. Проверить его ветвления без БД можно только так.

    Колонки задаются по таблице: _verify_schema_after_migrate спрашивает и documents,
    и sources, и отвечать им одним набором нельзя — тогда проверка sources.domain
    прошла бы по колонкам documents и ничего не поймала.
    """

    def __init__(self, table_exists, columns=(), source_columns=None) -> None:
        self._table_exists = table_exists
        self._columns = {"documents": list(columns),
                         "sources": list(source_columns if source_columns is not None else ["domain"])}
        self._answer: List[Tuple[object, ...]] = []

    def execute(self, sql: str, params: object = None) -> None:
        if "to_regclass" in sql:
            self._answer = [(self._table_exists,)]
        elif "information_schema.columns" in sql:
            table = params[0] if params else "documents"
            self._answer = [(name,) for name in self._columns.get(table, [])]
        else:
            self._answer = []

    def fetchone(self):
        return self._answer[0] if self._answer else None

    def fetchall(self):
        return list(self._answer)


def test_migration_runner_logic() -> None:
    """Логика раннера миграций — без БД: чек-суммы, порядок, режим совместимости.

    Миграции и сид ни разу не запускались (PostgreSQL локально нет, Docker нет).
    Значит любую ошибку в этом коде найдёт только жюри. Здесь проверяется всё,
    что проверяется без сервера.
    """
    section("12. Раннер миграций: логика без PostgreSQL")

    from ml.db import migrate

    found = migrate.discover()
    check(len(found) >= 2, "миграции найдены", "найдено: {}".format(len(found)))
    names = [name for name, _, _ in found]
    check(names == sorted(names), "миграции идут в порядке имени", "порядок: {}".format(names))
    check(all(name[:3].isdigit() for name in names),
          "имена миграций начинаются с номера", "имена: {}".format(names))

    checksums = {name: checksum for name, _, checksum in found}
    check(len(set(checksums.values())) == len(checksums), "контрольные суммы различаются")
    again = {name: checksum for name, _, checksum in migrate.discover()}
    check(again == checksums, "контрольная сумма устойчива между вызовами")

    os.environ["DATABASE_URL"] = "postgresql+psycopg://u:p@h:5432/db"
    try:
        check(migrate.database_url().startswith("postgresql://"),
              "префикс SQLAlchemy срезается для psycopg", migrate.database_url())
    finally:
        os.environ.pop("DATABASE_URL", None)

    # --- определение чужой схемы -------------------------------------------------
    # Раннер больше НЕ отказывается работать поверх db/init.sql: 001 добавляет
    # недостающие колонки блоком совместимости. Задача детектора — распознать
    # ситуацию и назвать её в логе, а не остановить прогон.

    check(migrate._detect_foreign_schema(_FakeCursor(False, [])) is None,
          "чистая база: чужой схемы не обнаружено")

    ours = list(migrate.REQUIRED_DOCUMENT_COLUMNS) + ["id", "title", "companies", "embedding"]
    check(migrate._detect_foreign_schema(_FakeCursor(True, ours)) is None,
          "своя схема: чужой схемы не обнаружено")

    foreign = ["id", "title", "area", "companies", "raw_text", "stage", "trend", "embedding"]
    missing = migrate._detect_foreign_schema(_FakeCursor(True, foreign))
    check(missing is not None and set(missing) == set(migrate.REQUIRED_DOCUMENT_COLUMNS),
          "схема db/init.sql: детектор называет все недостающие колонки",
          "вернулось: {}".format(missing))

    # --- проверка после применения ----------------------------------------------
    # Она заменила прежний отказ на входе: ошибка ловится по факту, а не по догадке.

    try:
        migrate._verify_schema_after_migrate(_FakeCursor(True, ours))
        check(True, "после миграций: полная схема проходит проверку")
    except SystemExit as exc:
        check(False, "после миграций: полная схема проходит проверку", str(exc)[:90])

    try:
        migrate._verify_schema_after_migrate(_FakeCursor(True, foreign))
        check(False, "после миграций: неполная схема documents валит прогон")
    except SystemExit as exc:
        message = str(exc)
        check("documents" in message and "001_core_schema.sql" in message,
              "после миграций: неполная схема documents валит прогон и называет причину",
              "сообщение: {}".format(message[:90]))

    # sources.domain забыть так же легко, как колонки documents: сид падает на нём.
    try:
        migrate._verify_schema_after_migrate(_FakeCursor(True, ours, source_columns=["url"]))
        check(False, "после миграций: отсутствие sources.domain валит прогон")
    except SystemExit as exc:
        check("sources" in str(exc) and "domain" in str(exc),
              "после миграций: отсутствие sources.domain валит прогон",
              "сообщение: {}".format(str(exc)[:90]))


#: Слова, которые встречаются в определении индекса, но колонками не являются.
_SQL_NOISE = frozenset({
    "lower", "desc", "asc", "where", "is", "not", "null", "and", "or", "with", "using",
    "true", "false", "hnsw", "ivfflat", "gin", "gist", "btree",
    "vector_cosine_ops", "vector_l2_ops", "text_pattern_ops", "m", "ef_construction",
})
#: Первые слова в определении таблицы, которые начинают ограничение, а не колонку.
_SQL_NOT_COLUMN = frozenset({"constraint", "primary", "unique", "foreign", "check", "exclude"})


def _sql_without_comments(sql: str) -> str:
    return "\n".join(re.sub(r"--.*$", "", line) for line in sql.splitlines())


def _split_top_level(body: str) -> List[str]:
    """Разбить тело CREATE TABLE по запятым верхнего уровня.

    Простой split(',') не годится: `CHECK (stage BETWEEN 1 AND 4)` и `vector(1024)`
    содержат свои запятые и скобки.
    """
    parts, current, depth = [], "", 0
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return parts


def _columns_created(sql: str, created: Dict[str, set]) -> None:
    """Пополнить {таблица -> колонки} по CREATE TABLE и ALTER TABLE ADD COLUMN."""
    for table, body in re.findall(
            r"CREATE TABLE IF NOT EXISTS\s+(\w+)\s*\((.*?)\n\);", sql, re.S | re.I):
        columns = created.setdefault(table.lower(), set())
        for part in _split_top_level(body):
            tokens = part.strip().split()
            if tokens and tokens[0].lower() not in _SQL_NOT_COLUMN:
                columns.add(tokens[0].lower())
    for table, column in re.findall(
            r"ALTER TABLE\s+(\w+)\s+ADD COLUMN IF NOT EXISTS\s+(\w+)", sql, re.I):
        created.setdefault(table.lower(), set()).add(column.lower())


def test_migrations_real_grammar() -> None:
    """Разбор миграций НАСТОЯЩИМ парсером PostgreSQL (pglast над libpg_query).

    Зачем отдельно от раздела 13. Та проверка самописная: считает скобки, ищет
    незакрытые DO-блоки. Класс ошибок, который ловит только сервер, она пропускает.
    pglast — это тот же парсер, что внутри PostgreSQL, поэтому синтаксис проверяется
    по-настоящему, без запущенной базы. Миграции до сих пор не исполнялись ни разу,
    так что это единственная имеющаяся гарантия, что они не упадут на первом прогоне.

    pglast — инструмент разработки, в ml/requirements.txt его нет: на стенде он не
    нужен. Если пакет не установлен, раздел помечается пропущенным, а не валит прогон.
    """
    section("15. Миграции: разбор настоящим парсером PostgreSQL")

    try:
        import pglast
        from pglast import ast, enums
    except ImportError:
        print("  [проп]  pglast не установлен (pip install pglast) — раздел пропущен")
        return

    files = sorted(Path("ml/db/migrations").glob("*.sql"))
    parsed = {}
    for path in files:
        sql = path.read_text(encoding="utf-8")
        try:
            parsed[path.name] = pglast.parse_sql(sql)
            check(True, "{}: разбирается парсером PostgreSQL".format(path.name))
        except Exception as exc:  # noqa: BLE001 — любая ошибка разбора это провал
            check(False, "{}: разбирается парсером PostgreSQL".format(path.name), str(exc)[:120])

    if len(parsed) != len(files):
        return

    def replay(sql, schema):
        """Символически проигрываем DDL: CREATE TABLE и ALTER ... ADD COLUMN."""
        for raw in pglast.parse_sql(sql):
            stmt = raw.stmt
            if isinstance(stmt, ast.CreateStmt):
                columns = schema.setdefault(stmt.relation.relname, set())
                for element in (stmt.tableElts or ()):
                    if isinstance(element, ast.ColumnDef):
                        columns.add(element.colname)
            elif isinstance(stmt, ast.AlterTableStmt):
                columns = schema.setdefault(stmt.relation.relname, set())
                for cmd in (stmt.cmds or ()):
                    if (cmd.subtype == enums.AlterTableType.AT_AddColumn
                            and isinstance(cmd.def_, ast.ColumnDef)):
                        columns.add(cmd.def_.colname)

    def index_requirements(sql):
        """(имя индекса, таблица, колонки) для каждого CREATE INDEX, включая WHERE."""
        result = []
        for raw in pglast.parse_sql(sql):
            stmt = raw.stmt
            if not isinstance(stmt, ast.IndexStmt):
                continue
            found = set()

            def walk(node):
                if isinstance(node, ast.ColumnRef):
                    for field in node.fields:
                        if isinstance(field, ast.String):
                            found.add(field.sval)
                if isinstance(node, tuple):
                    for child in node:
                        walk(child)
                elif hasattr(node, "__dict__"):
                    for value in vars(node).values():
                        if isinstance(value, (ast.Node, tuple)):
                            walk(value)

            for param in (stmt.indexParams or ()):
                if param.name:
                    found.add(param.name)
                if param.expr is not None:
                    walk(param.expr)
            if stmt.whereClause is not None:
                walk(stmt.whereClause)
            result.append((stmt.idxname, stmt.relation.relname, found))
        return result

    from ml.db import migrate

    # Два сценария: чистая база и база, уже созданная db/init.sql. Второй — это то,
    # что происходит в docker-compose, и ровно там раньше падал 002.
    init_sql = Path("db/init.sql")
    scenarios = [("чистая база", "")]
    if init_sql.exists():
        scenarios.append(("поверх db/init.sql", init_sql.read_text(encoding="utf-8")))

    for label, base in scenarios:
        schema = {}
        if base:
            replay(base, schema)
        for path in files:
            replay(path.read_text(encoding="utf-8"), schema)

        for column in migrate.REQUIRED_DOCUMENT_COLUMNS:
            check(column in schema.get("documents", set()),
                  "{}: documents.{} существует после миграций".format(label, column))
        for column in migrate.REQUIRED_SOURCE_COLUMNS:
            check(column in schema.get("sources", set()),
                  "{}: sources.{} существует после миграций".format(label, column))

        unresolved = []
        for path in files:
            for name, table, columns in index_requirements(path.read_text(encoding="utf-8")):
                missing = sorted(c for c in columns if c not in schema.get(table, set()))
                if missing:
                    unresolved.append("{} -> {}.{}".format(name, table, ",".join(missing)))
        check(not unresolved,
              "{}: каждый индекс ссылается на существующую колонку".format(label),
              "не разрешаются: {}".format("; ".join(unresolved)))


def test_init_sql_compatibility() -> None:
    """Совместимость миграций со схемой db/init.sql (зона Участника 1).

    Те же таблицы documents/sources/signals создаёт db/init.sql, смонтированный в
    /docker-entrypoint-initdb.d/. На инициализации контейнера он выполняется первым,
    поэтому CREATE TABLE IF NOT EXISTS в 001 становятся no-op. Чтобы 002 не падал на
    CREATE INDEX по несуществующей колонке, 001 содержит блок совместимости на
    ALTER ... ADD COLUMN IF NOT EXISTS. Этот тест держит блок в согласии с тем, что
    реально требуют раннер и сид: добавили колонку в REQUIRED_* — добавьте и ALTER.
    """
    section("14. Совместимость со схемой db/init.sql")

    from ml.db import migrate
    from ml.db import seed as seed_module

    core = Path("ml/db/migrations/001_core_schema.sql").read_text(encoding="utf-8")
    body = _sql_without_comments(core)

    for column in migrate.REQUIRED_DOCUMENT_COLUMNS:
        pattern = r"ALTER\s+TABLE\s+documents\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+" + column + r"(?![A-Za-z0-9_])"
        check(bool(re.search(pattern, body, re.IGNORECASE)),
              "001: documents.{} добавляется поверх чужой схемы".format(column),
              "нет ALTER TABLE documents ADD COLUMN IF NOT EXISTS {}".format(column))

    for column in migrate.REQUIRED_SOURCE_COLUMNS:
        pattern = r"ALTER\s+TABLE\s+sources\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+" + column + r"(?![A-Za-z0-9_])"
        check(bool(re.search(pattern, body, re.IGNORECASE)),
              "001: sources.{} добавляется поверх чужой схемы".format(column),
              "нет ALTER TABLE sources ADD COLUMN IF NOT EXISTS {}".format(column))

    # Ни одной чужой колонки не переопределяем и не удаляем: живые вставки парсера
    # должны продолжать работать на той же базе.
    for forbidden in (r"DROP\s+COLUMN", r"DROP\s+TABLE", r"ALTER\s+COLUMN\s+\w+\s+TYPE"):
        check(not re.search(forbidden, body, re.IGNORECASE),
              "001: нет разрушающих операций ({})".format(forbidden.replace(chr(92) + "s+", " ")),
              "найдена разрушающая операция")

    # Размерность вектора диктует парсер: эмбеддинги считает и пишет он, /ml их не
    # вычисляет. Пока parser/app/embeddings.py не в этой ветке, проверка пропускается,
    # а после слияния с main начинает ловить расхождение.
    match = re.search(r"embedding\s+vector\((\d+)\)", body, re.IGNORECASE)
    check(match is not None, "001: размерность вектора объявлена")
    embeddings_file = Path("parser/app/embeddings.py")
    if match and embeddings_file.exists():
        parser_dim = re.search(r"EMBED_DIM\s*=\s*(\d+)", embeddings_file.read_text(encoding="utf-8"))
        if parser_dim:
            check(int(match.group(1)) == int(parser_dim.group(1)),
                  "001: размерность вектора совпадает с EMBED_DIM парсера",
                  "в миграции {}, у парсера {}".format(match.group(1), parser_dim.group(1)))
    elif match:
        print("  [проп]  сверка с EMBED_DIM парсера: {} нет в этой ветке".format(embeddings_file))

    # Сид должен уметь писать companies и в TEXT[], и в JSONB.
    template = seed_module.UPSERT_DOCUMENT
    check("{companies}" in template,
          "сид: выражение для companies подставляется по типу колонки")
    array_sql = template.format(companies="%(companies)s")
    json_sql = template.format(companies="%(companies)s::jsonb")
    check("{" not in array_sql.replace("{}", "") or "{companies}" not in array_sql,
          "сид: шаблон полностью раскрывается")
    check(seed_module.companies_value(["A"], "%(companies)s") == ["A"],
          "сид: для TEXT[] передаётся список")
    check(seed_module.companies_value(["A"], "%(companies)s::jsonb") == '["A"]',
          "сид: для JSONB передаётся JSON-строка")
    check("::jsonb" in json_sql and "::jsonb" not in array_sql,
          "сид: приведение к jsonb только в JSONB-режиме")


def test_migrations_sql_static() -> None:
    """Статическая проверка SQL-миграций: индексы по существующим колонкам, баланс блоков.

    PostgreSQL здесь нет, а миграции ни разу не исполнялись, поэтому синтаксис против
    настоящего грамматического разбора проверить нечем. Зато проверяется тот класс
    ошибок, который в рукописных миграциях и случается: индекс по колонке, которой
    никто не создал (ровно так ломается 002 поверх чужой схемы), незакрытый DO-блок,
    разъехавшиеся скобки, потерянная точка с запятой.
    """
    section("13. Миграции: статическая проверка SQL")

    files = sorted(Path("ml/db/migrations").glob("*.sql"))
    check(bool(files), "файлы миграций найдены", "найдено: {}".format(len(files)))
    if not files:
        return

    created: Dict[str, set] = {}
    for path in files:
        _columns_created(_sql_without_comments(path.read_text(encoding="utf-8")), created)
    check(set(created) >= {"documents", "sources", "signals"},
          "миграции создают documents / sources / signals",
          "создано: {}".format(sorted(created)))

    unknown_refs: List[str] = []
    for path in files:
        sql = _sql_without_comments(path.read_text(encoding="utf-8"))
        for name, table, columns, tail in re.findall(
                r"CREATE INDEX IF NOT EXISTS\s+(\w+)\s+ON\s+(\w+)\s*(?:USING\s+\w+\s*)?"
                r"\((.*?)\)(.*?);", sql, re.S | re.I):
            known = created.get(table.lower(), set())
            referenced = set(re.findall(r"[a-z_][a-z0-9_]*", (columns + " " + tail).lower()))
            for ref in sorted(referenced):
                if ref not in known and ref not in _SQL_NOISE and not ref.isdigit():
                    unknown_refs.append("{}: {}.{}".format(name, table, ref))
    check(not unknown_refs,
          "все колонки в индексах существуют",
          "нет таких колонок -> {}".format(unknown_refs))

    for path in files:
        sql = _sql_without_comments(path.read_text(encoding="utf-8"))
        check(sql.count("$$") % 2 == 0,
              "{}: DO-блоки закрыты".format(path.name),
              "разделителей $$ нечётное число: {}".format(sql.count("$$")))
        check(sql.count("(") == sql.count(")"),
              "{}: скобки сбалансированы".format(path.name),
              "открывающих {}, закрывающих {}".format(sql.count("("), sql.count(")")))
        check(sql.strip().endswith(";"),
              "{}: последний оператор завершён точкой с запятой".format(path.name))


def main() -> int:
    print("Дымовой прогон ML-модуля (слабые сигналы)")
    for test in (test_scales, test_sources_parsing, test_source_meta, test_features,
                 test_maturity_filter, test_scoring_fallback, test_ranking,
                 test_control_dataset, test_closed_dataset_if_present,
                 test_train_serve_skew, test_db_enums_match_contract,
                 test_migration_runner_logic, test_migrations_sql_static,
                 test_init_sql_compatibility, test_migrations_real_grammar):
        try:
            test()
        except Exception:  # noqa: BLE001 — падение одной секции не должно скрывать остальные
            _failures.append(test.__name__)
            print("  [ИСКЛЮЧЕНИЕ] в {}:".format(test.__name__))
            traceback.print_exc()

    print("\n{}\nПроверок: {}, провалов: {}".format("=" * 60, _checks, len(_failures)))
    if _failures:
        for name in _failures:
            print("  - {}".format(name))
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
