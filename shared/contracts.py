"""Транспортные DTO между сервисами (api ↔ ml ↔ parser) и наружу во фронт.

Отделено от shared/schema.py намеренно: schema.py — замороженный контракт данных из
CLAUDE.md §3, а здесь живут формы запросов/ответов HTTP. Их можно править легче,
но всё равно согласованно: их потребляют сразу два сервиса.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from shared.schema import SignalDoc, Source

TOP_N = 15  # CLAUDE.md §1 — итоговая выдача: ТОП-15 гипотез


class JobStatus(str, Enum):
    """Статусы асинхронной задачи поиска (паттерн job_id + polling)."""

    QUEUED = "в_очереди"
    RUNNING = "выполняется"
    DONE = "готово"
    FAILED = "ошибка"


# --- api: наружу ------------------------------------------------------------

class SearchRequest(BaseModel):
    """Свободный запрос пользователя."""

    query: str = Field(..., min_length=2, description="Свободный запрос на русском языке")
    area: Optional[str] = Field(None, description="Необязательный фильтр по области")
    limit: int = Field(TOP_N, ge=1, le=50, description="Сколько гипотез вернуть")


class SearchAccepted(BaseModel):
    """Ответ на POST /api/search — задача принята в работу."""

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    poll_url: str


class SearchStats(BaseModel):
    """Статистика прогона — показывается на экране результатов."""

    sources_processed: int = 0
    candidates_total: int = 0
    candidates_rejected: int = 0
    confident_signals: int = 0          # число сигналов с уверенностью > 0.75
    llm_model: Optional[str] = None     # какая модель выбрана (CLAUDE.md §2.4 — логируется)


class SearchResult(BaseModel):
    """Ответ на GET /api/search/{job_id}."""

    job_id: str
    status: JobStatus
    query: str
    created_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None
    results: List[SignalDoc] = Field(default_factory=list)   # ТОП-15
    rejected: List[SignalDoc] = Field(default_factory=list)  # отсеянные + rejected_reason (§2.10)
    stats: SearchStats = Field(default_factory=SearchStats)


class SourcesResponse(BaseModel):
    """Ответ на GET /api/sources — проверка источников."""

    total: int
    items: List[Source] = Field(default_factory=list)


# --- parser: POST /collect --------------------------------------------------

class CollectRequest(BaseModel):
    query: str
    area: Optional[str] = None
    limit: int = 30


class CollectResponse(BaseModel):
    query: str
    sources_processed: int = 0
    documents: List[SignalDoc] = Field(default_factory=list)


# --- ml: POST /score --------------------------------------------------------

class ScoreRequest(BaseModel):
    """На вход модели — документ по схеме §3 и исходный запрос (для контекста ранжирования)."""

    document: SignalDoc
    query: Optional[str] = None


class ScoreResponse(BaseModel):
    """Ответ модели этапа 1: балл + интерпретация (CLAUDE.md §2.2, §6 У2)."""

    id: str
    score: float
    is_weak_signal: bool
    why: str
    rejected_reason: Optional[str] = None
    predictors: List[str] = Field(default_factory=list)  # признаки, повлиявшие на решение


class ScoreBatchRequest(BaseModel):
    documents: List[SignalDoc] = Field(default_factory=list)
    query: Optional[str] = None


class ScoreBatchResponse(BaseModel):
    items: List[ScoreResponse] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str = "0.1.0"
    mock: bool = True
