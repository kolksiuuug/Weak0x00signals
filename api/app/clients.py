"""HTTP-клиенты внутренних сервисов."""
from __future__ import annotations

import logging
from typing import List, Optional

import httpx

from api.app.config import settings
from shared.contracts import CollectRequest, CollectResponse, InsightRequest, InsightResponse, ScoreBatchRequest, ScoreBatchResponse, ScoreResponse
from shared.schema import SignalDoc, Source

logger = logging.getLogger("api.clients")


class ServiceUnavailable(RuntimeError):
    pass


async def _request(method: str, base_url: str, path: str, payload: Optional[dict] = None) -> dict:
    url = base_url.rstrip("/") + path
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
            resp = await client.request(method, url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        logger.error("Сервис %s недоступен: %s", url, exc)
        raise ServiceUnavailable("Сервис {} недоступен: {}".format(url, exc)) from exc


async def collect(query: str, area: Optional[str] = None, limit: int = 30) -> CollectResponse:
    payload = CollectRequest(query=query, area=area, limit=limit).model_dump(mode="json")
    return CollectResponse.model_validate(await _request("POST", settings.parser_url, "/collect", payload))


async def score_batch(documents: List[SignalDoc], query: str) -> List[ScoreResponse]:
    payload = ScoreBatchRequest(documents=documents, query=query).model_dump(mode="json")
    return ScoreBatchResponse.model_validate(await _request("POST", settings.ml_url, "/score/batch", payload)).items


async def enrich(documents: List[SignalDoc], query: str) -> InsightResponse:
    payload = InsightRequest(documents=documents, query=query).model_dump(mode="json")
    return InsightResponse.model_validate(await _request("POST", settings.parser_url, "/enrich", payload))


async def parser_health() -> dict:
    return await _request("GET", settings.parser_url, "/health")


async def ml_health() -> dict:
    return await _request("GET", settings.ml_url, "/health")


async def llm_models() -> dict:
    return await _request("GET", settings.parser_url, "/models")


async def get_document(document_id: str) -> SignalDoc:
    return SignalDoc.model_validate(await _request("GET", settings.parser_url, "/document/" + document_id))


async def parser_sources(limit: int = 200) -> List[Source]:
    data = await _request("GET", settings.parser_url, "/sources")
    return [Source.model_validate(x) for x in data.get("items", [])[:limit]]
