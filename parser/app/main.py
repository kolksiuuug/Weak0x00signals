"""Сервис parser + LLM (Участник 3). Реальный сбор источников через коннекторы.

/collect: фан-аут коннекторов (arXiv, OpenAlex, GDELT, Патенты…) с таймаутами,
ретраями и деградацией; нормализация в SignalDoc; дедуп по URL. LLM-подслой
пока на моках — подключается следующим шагом в parser/app/llm.py.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from parser.app import orchestrator
from parser.app.llm import ALLOWED_MODELS, ModelNotAllowed, selected_model
from shared.contracts import CollectRequest, CollectResponse, HealthResponse

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("parser")

app = FastAPI(
    title="Слабые сигналы — сервис сбора источников и LLM",
    description="Коннекторы к открытым источникам + LLM-сервис.",
    version="0.2.0",
)


@app.on_event("startup")
def log_model_choice() -> None:
    """CLAUDE.md §2.4 — выбор модели логируется при старте."""
    try:
        logger.info("LLM белого списка выбрана: %s", selected_model())
    except ModelNotAllowed as exc:
        logger.error("%s", exc)


@app.get("/health", response_model=HealthResponse, summary="Проверка живости сервиса")
def health() -> HealthResponse:
    # Сбор источников — реальный; LLM-слой пока на моках (см. /models).
    return HealthResponse(service="parser", mock=True)


@app.get("/models", summary="Белый список разрешённых LLM")
def models() -> dict:
    """Возвращает разрешённые модели и текущую выбранную — для аудита жюри (§2.4)."""
    try:
        current = selected_model()
        error = None
    except ModelNotAllowed as exc:
        current, error = None, str(exc)
    return {"allowed": list(ALLOWED_MODELS), "selected": current, "error": error}


@app.post("/collect", response_model=CollectResponse, summary="Собрать документы по запросу")
async def collect(req: CollectRequest) -> CollectResponse:
    """Реальный сбор: коннекторы + нормализация + дедуп. Моки здесь не используются."""
    resp = await orchestrator.collect(req.query, area=req.area, limit=req.limit)
    logger.info("collect: запрос=%r, источников=%d, документов=%d",
                req.query, resp.sources_processed, len(resp.documents))
    return resp


@app.get("/sources", summary="Последние реальные источники сбора (дедуп по URL)")
async def sources(limit: int = 50) -> dict:
    items = await orchestrator.all_sources_dedup(limit=limit)
    return {"total": len(items), "items": [s.model_dump(mode="json") for s in items]}