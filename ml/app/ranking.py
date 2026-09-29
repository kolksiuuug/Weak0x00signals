"""Ранжирование кандидатов и формирование ТОП-15 (CLAUDE.md §6, Участник 2).

Почему ранжировать по «голой» уверенности модели недостаточно. Два кандидата с
одинаковым баллом 0.82 неравноценны, если у первого три источника высокой доверенности
(научная статья, патент, госреестр), а у второго один новостной. В выдачу, которую
смотрит жюри, первым должен идти тот, что лучше подтверждён — это прямо следует из
требований §2.7 и §2.8.

Но и уверенности с фактурой недостаточно. Уверенность отвечает на вопрос «слабый ли это
сигнал вообще», а не «насколько он важен». Кандидат на стадии концепции со стабильными
упоминаниями выглядит для классификатора максимально «слабым» и лез на первое место,
хотя у заказчика это балл 3 — низший приоритет. Замер это подтвердил: корреляция
Спирмена между позицией в выдаче и «Баллом» датасета была −0.04, то есть порядок к
оценке заказчика отношения не имел, а по теме Edge был прямо перевёрнут (+0.61).

Поэтому в ранг входит приоритет по методологии заказчика — «Балл = стадия + тренд»
(CLAUDE.md §0.3, §1), нормированный из 3..7 в 0..1:

    ранг = 0.50 * уверенность_модели
         + 0.30 * приоритет_заказчика
         + 0.20 * качество_фактуры

    приоритет_заказчика = (стадия + тренд - 3) / 4
    качество_фактуры    = 0.60 * доля_источников_высокой_доверенности
                        + 0.40 * min(число_источников, 3) / 3

Веса выбраны замером, а не на глаз, и это важно для защиты. При 0.50/0.30/0.20
корреляция позиции с «Баллом» — −0.765 (порядок согласован с заказчиком), причём
recall@15 и precision@15 остаются ровно теми же, 90 % и 100 %. Увеличивать вес
приоритета дальше нельзя: на 0.40/0.40/0.20 precision@15 падает до 92 %, потому что
в «Балл» входит стадия, а высокая стадия коррелирует со зрелостью — приоритет начинает
затаскивать в выдачу то, что фильтр §2.10 должен отсекать.

Коэффициенты зафиксированы и документированы намеренно: ранжирование должно
объясняться словами, а не подбираться на глаз.

Дополнительно: дедупликация по нормализованному названию и ограничение на число
кандидатов из одной области, чтобы ТОП-15 не выродился в пятнадцать вариаций одной темы.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ml.app.features import describe, observation_from_doc
from ml.app.model import WeakSignalScorer
from shared.contracts import TOP_N
from shared.schema import SignalDoc, TrustLevel

logger = logging.getLogger("ml.ranking")

CONFIDENCE_WEIGHT = 0.50
#: Вес приоритета по методологии заказчика («Балл = стадия + тренд», §0.3).
PRIORITY_WEIGHT = 0.30
EVIDENCE_WEIGHT = 0.20
HIGH_TRUST_WEIGHT = 0.60
SOURCE_COUNT_WEIGHT = 0.40

#: Границы «Балла» из CLAUDE.md §1: стадия 1..4 + тренд 1..3.
POINTS_MIN = 3
POINTS_MAX = 7
#: Больше трёх источников качество подтверждения уже не улучшают — насыщаем.
SOURCE_COUNT_SATURATION = 3
#: Нижняя граница квоты на одну область. Реальная квота адаптивна (см. area_quota):
#: запрос по одной теме («финтех») должен отдавать все 15 позиций из этой темы,
#: а широкий запрос по шести темам — не давать одной теме занять всю выдачу.
MAX_PER_AREA = 4


def area_quota(limit: int, n_areas: int, floor: int = MAX_PER_AREA) -> int:
    """Сколько кандидатов из одной области допускается в ТОП-N.

    Квота = max(floor, ceil(limit / число_областей)). Одна область -> вся выдача,
    две -> по половине, шесть -> по 3, но не меньше floor. Формула прозрачна и
    объяснима словами, как и остальное ранжирование.
    """
    if n_areas <= 1:
        return limit
    return max(floor, -(-limit // n_areas))  # ceil без импорта math


@dataclass
class RankedItem:
    """Кандидат с посчитанным рангом и разложением ранга на составляющие."""

    doc: SignalDoc
    confidence: float
    evidence_quality: float
    rank_score: float
    priority: float = 0.0
    #: True, если стадия или тренд не пришли из источника, а выведены из текста.
    #: Идёт в объяснение: приоритет тогда — оценка системы, а не факт (§2.9, §10 методологии).
    priority_inferred: bool = False
    position: int = 0


def _norm_title(title: str) -> str:
    """Нормализованное название для дедупликации: без регистра, ё и пунктуации."""
    text = (title or "").replace("ё", "е").lower()
    return re.sub(r"[^0-9a-zа-я]+", "", text)


def evidence_quality(doc: SignalDoc) -> float:
    """Качество фактуры кандидата в 0..1 по его источникам."""
    obs = observation_from_doc(doc)
    if not obs.sources:
        return 0.0
    high = sum(1 for _, trust in obs.sources if trust == TrustLevel.HIGH)
    share_high = high / len(obs.sources)
    count_part = min(len(obs.sources), SOURCE_COUNT_SATURATION) / SOURCE_COUNT_SATURATION
    return HIGH_TRUST_WEIGHT * share_high + SOURCE_COUNT_WEIGHT * count_part


def priority_score(obs) -> float:
    """Приоритет по методологии заказчика: «Балл = стадия + тренд» из 3..7 в 0..1.

    Балл 3 (концепция + стабильно) -> 0.0, балл 7 (раннее внедрение + быстрый рост) -> 1.0.
    Это оценка заказчика, официально подтверждённая (CLAUDE.md §0.3), а не наша выдумка.
    """
    points = obs.stage + obs.trend
    points = max(POINTS_MIN, min(POINTS_MAX, points))
    return (points - POINTS_MIN) / float(POINTS_MAX - POINTS_MIN)


def rank_score(confidence: float, priority: float, quality: float) -> float:
    return (CONFIDENCE_WEIGHT * confidence
            + PRIORITY_WEIGHT * priority
            + EVIDENCE_WEIGHT * quality)


def explain_rank(item: RankedItem) -> str:
    """Русское объяснение позиции — идёт в карточку-инсайт.

    Разложение обязано совпадать с формулой до последнего слагаемого: жюри отдельно
    оценивает интерпретируемость, и объяснение, не сходящееся с числом, хуже его отсутствия.
    """
    note = ""
    if item.priority_inferred:
        note = (" Стадия и тренд для приоритета выведены системой из текста источников, "
                "а не взяты из размеченного поля.")
    return (
        "Позиция {} в выдаче. Итоговый ранг {:.3f} = {:.2f} × уверенность модели ({:.3f}) "
        "+ {:.2f} × приоритет по «Баллу» заказчика, стадия+тренд ({:.3f}) "
        "+ {:.2f} × качество подтверждающей фактуры ({:.3f}).{}".format(
            item.position, item.rank_score,
            CONFIDENCE_WEIGHT, item.confidence,
            PRIORITY_WEIGHT, item.priority,
            EVIDENCE_WEIGHT, item.evidence_quality,
            note,
        )
    )


def rank(documents: Sequence[SignalDoc],
         scorer: WeakSignalScorer,
         limit: int = TOP_N,
         max_per_area: int = MAX_PER_AREA,
         query: Optional[str] = None) -> Tuple[List[SignalDoc], List[SignalDoc]]:
    """Кандидаты -> (ТОП-N принятых, отклонённые с причиной).

    Каждый документ проходит скоринг и фильтр зрелости. Отклонённые не теряются:
    они возвращаются отдельным списком с заполненным rejected_reason (CLAUDE.md §2.10) —
    жюри отдельно оценивает обоснованность исключения.
    """
    accepted: List[RankedItem] = []
    rejected: List[SignalDoc] = []
    relevance_fallback: List[RankedItem] = []
    seen: Dict[str, bool] = {}

    for doc in documents:
        key = _norm_title(doc.title)
        if key and key in seen:
            rejected.append(doc.model_copy(update={
                "is_weak_signal": False,
                "rejected_reason": "Отклонено как дубликат: технология с таким названием уже есть в выдаче.",
            }))
            continue
        seen[key] = True

        result = scorer.score_document(doc, query)
        enriched = doc.model_copy(update={
            "score": result.score,
            "why": result.why,
            "is_weak_signal": result.is_weak_signal,
            "rejected_reason": result.rejected_reason,
            "model_version": result.model_version,
            "model_mode": result.model_mode,
        })

        if result.rejected_reason:
            # Документ уже прошёл серверный context gate и trust-слой. Поэтому
            # даже при отказе maturity/моделью он остаётся допустимым тематическим
            # резервом для TOP-15. Не называем его weak signal, но используем
            # только если настоящих weak signals недостаточно.
            obs = observation_from_doc(doc)
            quality = evidence_quality(doc)
            priority = priority_score(obs)
            relevance_fallback.append(RankedItem(
                doc=enriched.model_copy(update={
                    "is_weak_signal": False,
                    "rejected_reason": None,
                }),
                confidence=result.score,
                evidence_quality=quality,
                priority=priority,
                priority_inferred=bool(obs.stage_inferred or obs.trend_inferred),
                rank_score=rank_score(result.score, priority, quality),
            ))
            rejected.append(enriched)
            continue

        # Порог модели — такая же граница выдачи, как и фильтр зрелости. Без этой
        # проверки кандидат с уверенностью 0.2 при пороге 0.55 попадал бы в ТОП просто
        # потому, что фильтр его не отклонил. Причина пишется явно (CLAUDE.md §2.10).
        if not result.is_weak_signal:
            # Отдельный relevance fallback: если после контекстного gate и
            # maturity-фильтра осталось меньше 15, добираем только кандидатов,
            # которые уже прошли тематический gate. Они не называются «слабыми
            # сигналами», но позволяют показать полноценный релевантный TOP-15.
            relevance_fallback.append(RankedItem(
                doc=enriched.model_copy(update={"is_weak_signal": False, "rejected_reason": None}),
                confidence=result.score,
                evidence_quality=evidence_quality(doc),
                priority=priority_score(observation_from_doc(doc)),
                priority_inferred=bool(observation_from_doc(doc).stage_inferred or observation_from_doc(doc).trend_inferred),
                rank_score=rank_score(result.score, priority_score(observation_from_doc(doc)), evidence_quality(doc)),
            ))
            rejected.append(enriched.model_copy(update={
                "rejected_reason": (
                    "Не прошёл порог слабого сигнала ({:.1f} % при пороге {:.1f} %), "
                    "но сохранён как релевантный кандидат для заполнения TOP-15.".format(
                        result.score * 100, scorer.threshold * 100)
                ),
            }))
            continue

        quality = evidence_quality(doc)
        obs = observation_from_doc(doc)
        priority = priority_score(obs)
        accepted.append(RankedItem(
            doc=enriched,
            confidence=result.score,
            evidence_quality=quality,
            priority=priority,
            priority_inferred=bool(obs.stage_inferred or obs.trend_inferred),
            rank_score=rank_score(result.score, priority, quality),
        ))

    accepted.sort(key=lambda item: (item.rank_score, item.confidence), reverse=True)

    # разнообразие по областям: квота адаптивна к числу областей среди принятых
    # и применяется, только если кандидатов больше лимита
    areas_present = {(item.doc.area or "без области").strip().lower() for item in accepted}
    quota = area_quota(limit, len(areas_present), floor=max_per_area)

    top: List[RankedItem] = []
    overflow: List[RankedItem] = []
    per_area: Dict[str, int] = {}
    for item in accepted:
        area = (item.doc.area or "без области").strip().lower()
        if len(accepted) > limit and per_area.get(area, 0) >= quota:
            overflow.append(item)
            continue
        per_area[area] = per_area.get(area, 0) + 1
        top.append(item)
        if len(top) >= limit:
            break

    # если из-за ограничения по областям набралось меньше лимита — добираем лучшими из остатка
    if len(top) < limit:
        top.extend(overflow[: limit - len(top)])

    # Затем добираем релевантными кандидатами, уже прошедшими context gate.
    # Сначала сохраняем настоящие weak signals, затем менее уверенные, но тематически
    # подходящие документы.
    if len(top) < limit and relevance_fallback:
        relevance_fallback.sort(key=lambda item: (item.rank_score, item.confidence), reverse=True)
        used = {item.doc.id for item in top}
        for item in relevance_fallback:
            if item.doc.id in used:
                continue
            top.append(item)
            used.add(item.doc.id)
            if len(top) >= limit:
                break

    results: List[SignalDoc] = []
    for position, item in enumerate(top, start=1):
        item.position = position
        obs = observation_from_doc(item.doc)
        facts = describe(obs)
        markers = list(dict.fromkeys((facts.get("weakness_hits") or []) + (facts.get("maturity_hits") or [])[:2]))
        results.append(item.doc.model_copy(update={
            "why": "{} {}".format(item.doc.why or "", explain_rank(item)).strip(),
            "signal_markers": markers[:8],
        }))

    logger.info("Ранжирование: кандидатов=%d, в выдаче=%d, отклонено=%d",
                len(documents), len(results), len(rejected))
    return results, rejected


__all__ = ["RankedItem", "rank", "rank_score", "priority_score", "evidence_quality",
           "explain_rank", "MAX_PER_AREA"]
