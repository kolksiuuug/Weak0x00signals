"""Сервис parser + LLM (Участник 3). Скелет: только моки, без сетевых вызовов наружу.

Наружу не публикуется — доступен только внутри docker-сети по имени `parser`.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from parser.app import mock_data
from parser.app.llm import ALLOWED_MODELS, ModelNotAllowed, selected_model
from shared.contracts import CollectRequest, CollectResponse, HealthResponse

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("parser")

app = FastAPI(
    title="Слабые сигналы — сервис сбора источников и LLM",
    description="Коннекторы к открытым источникам + LLM-сервис. Сейчас работает в МОК-режиме.",
    version="0.1.0",
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
def collect(req: CollectRequest) -> CollectResponse:
    """МОК: возвращает нормализованные SignalDoc с полностью заполненными источниками.

    Реальная реализация: arXiv API, Crossref/OpenAlex, GDELT, PatentsView, trafilatura;
    ретраи, таймауты, деградация при падении отдельного источника, дедуп по URL.
    """
    docs = mock_data.build_documents(req.query, area=req.area, limit=req.limit)
    processed = sum(len(d.sources) for d in docs)
    logger.info("collect: запрос=%r, документов=%d, источников=%d", req.query, len(docs), processed)
    return CollectResponse(query=req.query, sources_processed=processed, documents=docs)


@app.get("/sources", summary="Все источники мок-корпуса")
def sources() -> dict:
    items = mock_data.all_sources()
    return {"total": len(items), "items": [s.model_dump(mode="json") for s in items]}
