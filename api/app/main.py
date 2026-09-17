"""FastAPI-оркестратор (Участник 1).

Единственный сервис, доступный снаружи — и то через reverse-proxy Caddy по префиксу /api.
Swagger: /api/docs, OpenAPI: /api/openapi.json.
"""

from __future__ import annotations

import logging

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query

from api.app import clients, jobs
from api.app.clients import ServiceUnavailable
from api.app.config import settings
from api.app.pipeline import run_search
from api.app.trust import check_document, dedup_sources
from shared.contracts import (
    HealthResponse,
    SearchAccepted,
    SearchRequest,
    SearchResult,
    SourcesResponse,
)
from shared.schema import SignalDoc

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(
    title="Слабые сигналы — API",
    description=(
        "Оркестратор: свободный запрос → сбор источников → trust-слой → скоринг → ТОП-15 гипотез. "
        "Скелет работает на МОК-данных: реальные коннекторы и модель подключаются участниками 2 и 3."
    ),
    version=settings.version,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


@app.get("/health", response_model=HealthResponse, tags=["служебные"], summary="Проверка живости")
async def health() -> HealthResponse:
    return HealthResponse(service=settings.service_name, version=settings.version, mock=True)


@app.get("/api/health/deep", tags=["служебные"], summary="Проверка всех зависимостей")
async def deep_health() -> dict:
    """Опрос внутренних сервисов и Redis — удобно для отладки на VPS."""
    result = {"api": "ok", "redis": "ok" if await jobs.ping() else "недоступен"}
    for name, probe in (("ml", clients.ml_health), ("parser", clients.parser_health)):
        try:
            await probe()
            result[name] = "ok"
        except ServiceUnavailable as exc:
            result[name] = "недоступен: {}".format(exc)
    return result


@app.post(
    "/api/search",
    response_model=SearchAccepted,
    status_code=202,
    tags=["поиск"],
    summary="Поставить поиск в очередь",
)
async def search(req: SearchRequest, background: BackgroundTasks) -> SearchAccepted:
    """Асинхронный паттерн: сразу отдаём job_id, результат забирается опросом."""
    job = await jobs.create_job(req.query)
    background.add_task(run_search, job.job_id, req)
    logger.info("Принят запрос %r, job_id=%s", req.query, job.job_id)
    return SearchAccepted(job_id=job.job_id, status=job.status, poll_url="/api/search/{}".format(job.job_id))


@app.get(
    "/api/search/{job_id}",
    response_model=SearchResult,
    tags=["поиск"],
    summary="Забрать результат поиска (ТОП-15)",
)
async def search_result(job_id: str) -> SearchResult:
    job = await jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена или срок её хранения истёк.")
    return job


@app.get(
    "/api/signal/{signal_id}",
    response_model=SignalDoc,
    tags=["поиск"],
    summary="Карточка-инсайт по одному сигналу",
)
async def signal_card(signal_id: str) -> SignalDoc:
    """МОК-карточка: тянем документ у parser, прогоняем через trust-слой и ml."""
    try:
        collected = await clients.collect("карточка сигнала {}".format(signal_id), limit=100)
    except ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    doc = next((d for d in collected.documents if d.id == signal_id), None)
    if doc is None:
        raise HTTPException(status_code=404, detail="Сигнал с идентификатором «{}» не найден.".format(signal_id))

    doc, reason = check_document(doc)
    if reason:
        return doc.model_copy(update={"is_weak_signal": False, "rejected_reason": reason})

    try:
        scored = await clients.score_batch([doc], "карточка сигнала")
    except ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if not scored:
        return doc
    item = scored[0]
    return doc.model_copy(update={
        "score": item.score,
        "why": item.why,
        "is_weak_signal": item.is_weak_signal,
        "rejected_reason": item.rejected_reason,
    })


@app.get(
    "/api/sources",
    response_model=SourcesResponse,
    tags=["источники"],
    summary="Проверка источников",
)
async def sources(
    limit: int = Query(100, ge=1, le=1000, description="Сколько источников вернуть"),
) -> SourcesResponse:
    """Список источников после дедупа и пересчёта доверенности trust-слоем."""
    try:
        collected = await clients.collect("проверка источников", limit=100)
    except ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    flat = []
    for doc in collected.documents:
        doc, _ = check_document(doc)
        flat.extend(doc.sources)

    flat = dedup_sources(flat)
    return SourcesResponse(total=len(flat), items=flat[:limit])
