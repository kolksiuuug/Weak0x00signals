"""HTTP-клиенты внутренних сервисов."""
from __future__ import annotations

import asyncio
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
    last_exc = None
    # Короткие сетевые сбои между контейнерами не должны превращать live-поиск
    # в ошибку UI. Повторяем только транспортные/5xx ошибки, не скрывая 4xx.
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
                resp = await client.request(method, url, json=payload)
                if resp.status_code >= 500:
                    resp.raise_for_status()
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response is not None and exc.response.status_code < 500:
                break
            logger.warning("Сервис %s вернул %s, попытка %d/5", url, exc.response.status_code if exc.response else "?", attempt + 1)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
            last_exc = exc
            logger.warning("Сервис %s временно недоступен (%s), попытка %d/5", url, type(exc).__name__, attempt + 1)
        except httpx.HTTPError as exc:
            last_exc = exc
            break
        if attempt < 4:
            await asyncio.sleep(min(4.0, 0.75 * (attempt + 1)))
    exc = last_exc or RuntimeError("неизвестная HTTP ошибка")
    logger.error("Сервис %s недоступен после повторов: %s", url, exc)
    raise ServiceUnavailable("Сервис {} недоступен: {}".format(url, exc)) from exc


async def collect(query: str, area: Optional[str] = None, limit: int = 75) -> CollectResponse:
    payload = CollectRequest(query=query, area=area, limit=limit).model_dump(mode="json")
    return CollectResponse.model_validate(await _request("POST", settings.parser_url, "/collect", payload))


async def score_batch(documents: List[SignalDoc], query: str) -> List[ScoreResponse]:
    payload = ScoreBatchRequest(documents=documents, query=query).model_dump(mode="json")
    return ScoreBatchResponse.model_validate(await _request("POST", settings.ml_url, "/score/batch", payload)).items




async def rank_documents(documents: List[SignalDoc], query: str, limit: int = 15) -> dict:
    payload = {"documents": [doc.model_dump(mode="json") for doc in documents], "query": query, "limit": limit}
    return await _request("POST", settings.ml_url, "/rank", payload)

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
