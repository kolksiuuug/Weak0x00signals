"""Хранилище асинхронных задач поиска в Redis (паттерн job_id + polling).

Интерфейс намеренно узкий (create / get / save), чтобы позже подменить Redis на очередь
(Celery/arq) без правок роутов.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis

from api.app.config import settings
from shared.contracts import JobStatus, SearchResult

_KEY = "signals:job:{}"

_redis: Optional["aioredis.Redis"] = None


def get_redis() -> "aioredis.Redis":
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
    return _redis


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def create_job(query: str) -> SearchResult:
    """Регистрирует задачу в статусе «в очереди» и возвращает её начальное состояние."""
    job = SearchResult(
        job_id=uuid.uuid4().hex,
        status=JobStatus.QUEUED,
        query=query,
        created_at=now_iso(),
    )
    await save_job(job)
    return job


async def save_job(job: SearchResult) -> None:
    payload = json.dumps(job.model_dump(mode="json"), ensure_ascii=False)
    await get_redis().set(_KEY.format(job.job_id), payload, ex=settings.job_ttl_seconds)


async def get_job(job_id: str) -> Optional[SearchResult]:
    payload = await get_redis().get(_KEY.format(job_id))
    if payload is None:
        return None
    return SearchResult.model_validate(json.loads(payload))


async def ping() -> bool:
    try:
        return bool(await get_redis().ping())
    except Exception:
        return False
