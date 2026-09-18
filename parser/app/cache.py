"""Redis-кэш для результатов сбора (CLAUDE.md §6 У3: «кэш, чтобы демо не тормозило»).

Два уровня:
  1. кэш по запросу  — результаты /collect на TTL, ключ от канонизированного запроса;
  2. кэш по URL     — сырые ответы отдельных коннекторов (если два запроса тянут один URL).

Graceful degradation: если Redis недоступен — просто работаем без кэша (никаких падений).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable, List, Optional

from parser.app.config import settings

logger = logging.getLogger("parser.cache")

try:
    import redis.asyncio as aioredis
except Exception:  # pragma: no cover
    aioredis = None  # type: ignore


def _client():
    """Ленивое подключение напрямую (без пулов) — сервис один инстанс."""
    if not settings.redis_url or aioredis is None:
        return None
    try:
        return aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=1.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Redis недоступен, кэш выключен: %s", exc)
        return None


def _key(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"parser:{prefix}:{digest}"


def _encode(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


async def cache_get(prefix: str, value: str) -> Optional[str]:
    """Вернуть сырую строку из кэша либо None (в т.ч. при недоступном Redis)."""
    client = _client()
    if client is None:
        return None
    try:
        data = await client.get(_key(prefix, value))
        if data:
            logger.info("Кэш HIT: %s/%s", prefix, value[:48])
        return data
    except Exception as exc:  # noqa: BLE001
        logger.debug("Кэш GET потерпел неудачу: %s", exc)
        return None
    finally:
        await client.aclose()


async def cache_set(prefix: str, value: str, payload: str, ttl: Optional[int] = None) -> None:
    client = _client()
    if client is None:
        return
    try:
        await client.set(_key(prefix, value), payload, ex=ttl or settings.cache_ttl_seconds)
        logger.info("Кэш SET: %s/%s", prefix, value[:48])
    except Exception as exc:  # noqa: BLE001
        logger.debug("Кэш SET потерпел неудачу: %s", exc)
    finally:
        await client.aclose()


async def cache_get_json(prefix: str, value: str) -> Optional[Any]:
    raw = await cache_get(prefix, value)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None


async def cache_set_json(prefix: str, value: str, payload: Any, ttl: Optional[int] = None) -> None:
    await cache_set(prefix, value, _encode(payload), ttl=ttl)


async def cache_or_compute(prefix: str, key: str, compute: Callable[[], Any], ttl: Optional[int] = None) -> Any:
    """Декоратор-паттерн: отдаём из кэша или вычисляем и кладём. Кэш, упавший, не роняет compute."""
    cached = await cache_get_json(prefix, key)
    if cached is not None:
        return cached
    result = await compute()
    if result is not None:
        await cache_set_json(prefix, key, result, ttl=ttl)
    return result


__all__ = [
    "cache_get",
    "cache_set",
    "cache_get_json",
    "cache_set_json",
    "cache_or_compute",
]