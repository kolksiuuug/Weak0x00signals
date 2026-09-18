"""Оркестратор сбора: параллельный фан-аут по коннекторам -> нормализация -> дедуп.

Деградация (CLAUDE.md §6 У3): упал один источник — пайплайн продолжает с остальными.
Кэш: по канонизированному запросу (весь ответ) и по URL для отдельных коннекторов.
Стабильный контракт наружу: CollectResponse (shared/contracts.py) не меняется.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from parser.app.cache import cache_get_json, cache_set_json
from parser.app.connectors import ConnectorError, registry
from parser.app.normalize import merge_sources, normalize_all
from shared.contracts import CollectResponse
from shared.schema import SignalDoc

logger = logging.getLogger("parser.orchestrator")


def _query_cache_key(query: str, area: Optional[str], limit: int) -> str:
    return "{}|{}|{}".format((query or "").strip().lower(), area or "", limit)


async def _run_connector(connector, query: str, limit: int) -> List[Dict[str, Any]]:
    """Один коннектор с изоляцией ошибок (деградация) и кэшем по URL-запросу."""
    per_url_key = f"{connector.name}|{query.strip().lower()}|{limit}"
    cached = await cache_get_json("connector", per_url_key)
    if cached is not None:
        logger.info("Кэш коннектора HIT: %s", connector.name)
        return cached

    try:
        raw = await connector.search(query, limit=limit)
        await cache_set_json("connector", per_url_key, raw)
        return raw
    except ConnectorError as exc:
        logger.warning("Деградация: источник «%s» пропущен (%s)", connector.name, exc)
        return []
    except Exception as exc:  # noqa: BLE001 — непредвиденное не роняет весь сбор
        logger.error("Деградация: источник «%s» — неожиданная ошибка: %s", connector.name, exc)
        return []


async def collect(query: str, area: Optional[str] = None, limit: int = 30) -> CollectResponse:
    """Основная точка входа: собрать и нормализовать документы по свободному запросу."""
    limit = max(1, min(int(limit), 100))
    cache_key = _query_cache_key(query, area, limit)

    async def _compute() -> CollectResponse:
        connectors = registry.all()
        per_connector = max(1, limit // max(1, len(connectors))) if connectors else 0

        logger.info(
            "collect: запрос=%r, area=%r, лимит=%d, коннекторов=%d",
            query, area, limit, len(connectors),
        )
        if not connectors:
            logger.warning("collect: нет зарегистрированных коннекторов")
            return CollectResponse(query=query, sources_processed=0, documents=[])

        raw_buckets = await asyncio.gather(
            *(_run_connector(c, query, per_connector) for c in connectors),
            return_exceptions=True,
        )
        raw_results: List[Dict[str, Any]] = []
        for bucket in raw_buckets:
            if isinstance(bucket, BaseException):
                logger.error("Неожиданный сбой фан-аута: %s", bucket)
                continue
            raw_results.extend(bucket or [])

        sources_processed = len(raw_results)
        docs = normalize_all(raw_results, query=query, area=area)

        if not docs:
            logger.warning(
                "collect: источники обработаны (%d), но валидных документов 0",
                sources_processed,
            )

        return CollectResponse(
            query=query,
            sources_processed=sources_processed,
            documents=docs[:limit],
        )

    cached = await cache_get_json("query", cache_key)
    if cached is not None:
        logger.info("collect: кэш по запросу HIT (%s)", cache_key)
        return CollectResponse.model_validate(cached)

    resp = await _compute()
    # Мок-данные в кэш/выдачу не подмешиваем; кэшируем только реальный результат.
    if resp.sources_processed:
        await cache_set_json("query", cache_key, resp.model_dump(mode="json"))
    return resp


async def collect_preview_sources(query: str, limit: int = 20) -> List[SignalDoc]:
    """Для внутреннего GET /sources: короткий сбор, чтобы отдать реальные источники."""
    resp = await collect(query, area=None, limit=limit)
    return resp.documents


async def all_sources_dedup(query: str = "слабые сигналы искусственного интеллекта", limit: int = 50) -> List:
    """Плоский дедуплицированный список источников последнего сбора."""
    docs = await collect_preview_sources(query, limit=limit)
    return merge_sources(docs)