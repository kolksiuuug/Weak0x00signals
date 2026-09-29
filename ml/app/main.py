"""ML-сервис (Участник 2): скоринг слабости сигнала, фильтр зрелости, ранжирование.

Наружу не публикуется — доступен только внутри docker-сети по имени `ml`.

Контракт CLAUDE.md §3 сохранён: POST /score и POST /score/batch принимают и возвращают
ровно те же модели из shared/contracts.py, что и раньше. Заглушка заменена на обученную
scikit-learn модель (ml/app/model.py) с честной интерпретацией вкладов признаков.

Новое, аддитивно и без ломки чужого кода:
  * GET  /model/info — паспорт модели: тип, дата обучения, порог, метрики, признаки.
                       Нужен для логирования и для защиты (CLAUDE.md §6, ИБ-слой У1).
  * POST /rank       — ранжирование пачки кандидатов в ТОП-15 с разделением на принятые
                       и отклонённые. DTO пока локальные: переносить их в shared/contracts.py
                       можно только по согласованию команды (CLAUDE.md §5).

Деградация: если артефакт модели не обучен, сервис поднимается на прозрачных правилах
и честно сообщает об этом в /health (mock=true). Демо не падает из-за отсутствия .joblib.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.app.model import WeakSignalScorer
from ml.app.ranking import rank
from shared.contracts import (
    TOP_N,
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
    description="Скоринг слабости сигнала, фильтр зрелости и ранжирование ТОП-15.",
    version="0.2.0",
)

#: Модель загружается один раз на процесс: joblib.load на каждый запрос — это лишние
#: десятки миллисекунд на кандидата, а кандидатов в одном поиске десятки.
scorer = WeakSignalScorer.load()


# --- локальные DTO ранжирования ---------------------------------------------
# Намеренно НЕ в shared/contracts.py: этот файл импортируют api и parser, менять его
# в одиночку нельзя (CLAUDE.md §5). Перенесём после согласования в общем чате.

class RankRequest(BaseModel):
    documents: List[SignalDoc] = Field(default_factory=list)
    query: Optional[str] = None
    limit: int = Field(TOP_N, ge=1, le=50)


class RankResponse(BaseModel):
    results: List[SignalDoc] = Field(default_factory=list)   # ТОП-N, уже отсортирован
    rejected: List[SignalDoc] = Field(default_factory=list)  # с заполненным rejected_reason
    total_candidates: int = 0


# --- эндпоинты --------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, summary="Проверка живости сервиса")
def health() -> HealthResponse:
    """mock=true означает, что артефакт модели не обучен и работают запасные правила."""
    return HealthResponse(service="ml", version=app.version, mock=not scorer.is_trained)


@app.get("/model/info", summary="Паспорт модели: тип, метрики, порог, признаки")
def model_info() -> Dict[str, Any]:
    return scorer.info()


@app.post("/score", response_model=ScoreResponse, summary="Скоринг одного документа")
def score(req: ScoreRequest) -> ScoreResponse:
    result = scorer.score_document(req.document, req.query)
    logger.info(
        "score: id=%s score=%.3f слабый=%s отклонён=%s",
        result.id, result.score, result.is_weak_signal, bool(result.rejected_reason),
    )
    return result


@app.post("/score/batch", response_model=ScoreBatchResponse, summary="Скоринг пачки документов")
def score_batch(req: ScoreBatchRequest) -> ScoreBatchResponse:
    items = [scorer.score_document(doc, req.query) for doc in req.documents]
    rejected = sum(1 for item in items if item.rejected_reason)
    logger.info("score/batch: документов=%d, отклонено фильтром=%d", len(items), rejected)
    return ScoreBatchResponse(items=items)


@app.post("/rank", response_model=RankResponse, summary="Ранжирование кандидатов в ТОП-15")
def rank_documents(req: RankRequest) -> RankResponse:
    results, rejected = rank(req.documents, scorer, limit=req.limit, query=req.query)
    return RankResponse(
        results=results,
        rejected=rejected,
        total_candidates=len(req.documents),
    )
