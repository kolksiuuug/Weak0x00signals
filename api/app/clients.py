"""HTTP-клиенты к внутренним сервисам ml и parser.

Обращение строго по имени сервиса в docker-сети — наружу эти сервисы не публикуются.
Деградация: если сервис недоступен, поднимается ServiceUnavailable, задача переходит
в статус «ошибка» с русским текстом, весь остальной пайплайн остаётся работоспособным.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import httpx

from api.app.config import settings
from shared.contracts import (
    CollectRequest,
    CollectResponse,
    ScoreBatchRequest,
    ScoreBatchResponse,
    ScoreResponse,
)
from shared.schema import SignalDoc

logger = logging.getLogger("api.clients")


class ServiceUnavailable(RuntimeError):
    """Внутренний сервис не ответил."""


async def _post(base_url: str, path: str, payload: dict) -> dict:
    url = base_url.rstrip("/") + path
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        logger.error("Сервис %s недоступен: %s", url, exc)
        raise ServiceUnavailable("Сервис {} недоступен: {}".format(url, exc)) from exc


async def _get(base_url: str, path: str) -> dict:
    url = base_url.rstrip("/") + path
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        logger.error("Сервис %s недоступен: %s", url, exc)
        raise ServiceUnavailable("Сервис {} недоступен: {}".format(url, exc)) from exc


async def collect(query: str, area: Optional[str] = None, limit: int = 30) -> CollectResponse:
    """parser: собрать документы-кандидаты по запросу."""
    payload = CollectRequest(query=query, area=area, limit=limit).model_dump(mode="json")
    data = await _post(settings.parser_url, "/collect", payload)
    return CollectResponse.model_validate(data)


async def score_batch(documents: List[SignalDoc], query: str) -> List[ScoreResponse]:
    """ml: посчитать скоринг и объяснение по пачке документов."""
    payload = ScoreBatchRequest(documents=documents, query=query).model_dump(mode="json")
    data = await _post(settings.ml_url, "/score/batch", payload)
    return ScoreBatchResponse.model_validate(data).items


async def parser_health() -> dict:
    return await _get(settings.parser_url, "/health")


async def ml_health() -> dict:
    return await _get(settings.ml_url, "/health")


async def llm_models() -> dict:
    """Белый список и выбранная модель — для логирования и аудита (CLAUDE.md §2.4)."""
    return await _get(settings.parser_url, "/models")
