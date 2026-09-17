"""ML-сервис (Участник 2). Скелет: правило-заглушка вместо обученной модели.

Наружу не публикуется — доступен только внутри docker-сети по имени `ml`.

CLAUDE.md §1: целевой балл датасета = стадия (1..4) + тренд (1..3), диапазон 3..7.
Заглушка воспроизводит именно эту логику, чтобы контракт `score(doc) -> (score, why)`
не пришлось менять при подключении настоящей scikit-learn модели.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from fastapi import FastAPI

from shared.contracts import (
    HealthResponse,
    ScoreBatchRequest,
    ScoreBatchResponse,
    ScoreRequest,
    ScoreResponse,
)
from shared.schema import SignalDoc

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ml")

app = FastAPI(
    title="Слабые сигналы — ML-сервис",
    description="Скоринг слабости сигнала и фильтр зрелости. Сейчас работает в МОК-режиме.",
    version="0.1.0",
)

STAGE_LABELS = {1: "Концепция/Исследование", 2: "Прототип/PoC", 3: "Пилот", 4: "Раннее внедрение"}
TREND_LABELS = {1: "упоминания стабильны", 2: "упоминания растут", 3: "упоминания растут быстро"}

MIN_SCORE, MAX_SCORE = 3, 7          # диапазон балла «стадия + тренд» из датасета
WEAK_SIGNAL_THRESHOLD = 0.55         # порог отнесения к слабому сигналу (мок)
HYPE_KEYWORDS = ("хайп", "инфошум", "перегрет")


def _weakness(doc: SignalDoc) -> float:
    """МОК-«уверенность в слабости сигнала» (поле score по CLAUDE.md §3).

    ВАЖНО: это НЕ балл датасета. Балл «стадия + тренд» растёт вместе со зрелостью
    технологии, поэтому напрямую как уверенность в слабости он не годится — он
    воспроизводится отдельно и выводится в `why` для интерпретируемости.

    Заглушка отражает суть слабого сигнала: тема рано по стадии, но набирает
    упоминания, игроков мало, публикаций мало. Настоящие веса даёт обученная
    scikit-learn модель Участника 2 — контракт score(doc) -> (score, why) не меняется.
    """
    stage = doc.stage or 1
    trend = doc.trend or 1
    trend_norm = (trend - 1) / 2.0          # 1..3 → 0..1
    earliness = 1.0 - (stage - 1) / 3.0     # 1..4 → 1..0 (чем раньше стадия, тем выше)
    sparse_players = 1.0 if len(doc.companies) <= 2 else 0.0
    sparse_pubs = 1.0 if len(doc.sources) <= 3 else 0.0
    value = 0.55 * trend_norm + 0.25 * earliness + 0.10 * sparse_players + 0.10 * sparse_pubs
    return round(min(1.0, max(0.0, value)), 3)


def _maturity_reason(doc: SignalDoc) -> Optional[str]:
    """МОК фильтра зрелости/хайпа. CLAUDE.md §2.10 — причина отклонения обязательна и явна."""
    text = (doc.raw_text or "").lower()
    if doc.stage == 4 and (doc.trend or 1) <= 1:
        return (
            "Отклонено фильтром зрелости: стадия «Раннее внедрение» при стабильном тренде "
            "упоминаний — рынок сформирован, есть отраслевые лидеры, признаков зарождения нет."
        )
    if any(word in text for word in HYPE_KEYWORDS):
        return (
            "Отклонено как маркетинговый хайп: объём медийных упоминаний кратно превышает "
            "объём верифицированных внедрений и научных публикаций."
        )
    return None


def _predictors(doc: SignalDoc) -> List[str]:
    """Интерпретируемость (CLAUDE.md §2.2): какие признаки повлияли на решение."""
    stage = doc.stage or 1
    trend = doc.trend or 1
    out = [
        "стадия развития: {} (+{})".format(STAGE_LABELS.get(stage, "неизвестно"), stage),
        "тренд упоминаний: {} (+{})".format(TREND_LABELS.get(trend, "неизвестно"), trend),
    ]
    if len(doc.companies) <= 2:
        out.append("концентрация игроков: мало компаний ({}) — признак раннего рынка".format(len(doc.companies)))
    if len(doc.sources) <= 3:
        out.append("разреженность публикаций: источников найдено {}".format(len(doc.sources)))
    return out


def score_document(doc: SignalDoc, query: Optional[str] = None) -> ScoreResponse:
    """Контракт инференса: score(doc) -> (score, why). Здесь — детерминированная заглушка."""
    stage = doc.stage or 1
    trend = doc.trend or 1
    points = max(MIN_SCORE, min(MAX_SCORE, stage + trend))
    score = _weakness(doc)
    rejected = _maturity_reason(doc)
    predictors = _predictors(doc)
    why = "Уверенность {:.0f} %. Балл датасета {}/7 = стадия {} + тренд {}. Признаки: {}. " \
          "[МОК-модель; обученная scikit-learn модель подключается Участником 2]".format(
              score * 100, points, stage, trend, "; ".join(predictors))
    return ScoreResponse(
        id=doc.id,
        score=score,
        is_weak_signal=(rejected is None and score >= WEAK_SIGNAL_THRESHOLD),
        why=why,
        rejected_reason=rejected,
        predictors=predictors,
    )


@app.get("/health", response_model=HealthResponse, summary="Проверка живости сервиса")
def health() -> HealthResponse:
    return HealthResponse(service="ml", mock=True)


@app.post("/score", response_model=ScoreResponse, summary="Скоринг одного документа")
def score(req: ScoreRequest) -> ScoreResponse:
    result = score_document(req.document, req.query)
    logger.info("score: id=%s score=%.3f отклонён=%s", result.id, result.score, bool(result.rejected_reason))
    return result


@app.post("/score/batch", response_model=ScoreBatchResponse, summary="Скоринг пачки документов")
def score_batch(req: ScoreBatchRequest) -> ScoreBatchResponse:
    items = [score_document(doc, req.query) for doc in req.documents]
    logger.info("score/batch: документов=%d", len(items))
    return ScoreBatchResponse(items=items)
