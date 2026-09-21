"""Транспортные DTO API ↔ parser ↔ ml."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from shared.schema import SignalDoc, Source

TOP_N = 15


class JobStatus(str, Enum):
    QUEUED = "в_очереди"
    RUNNING = "выполняется"
    DONE = "готово"
    FAILED = "ошибка"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    area: Optional[str] = None
    limit: int = Field(TOP_N, ge=1, le=15)


class SearchAccepted(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    poll_url: str


class SearchStats(BaseModel):
    sources_processed: int = 0
    candidates_total: int = 0
    candidates_rejected: int = 0
    confident_signals: int = 0
    llm_model: Optional[str] = None
    ml_mode: Optional[str] = None
    retrieval_mode: Optional[str] = None
    connector_status: dict = Field(default_factory=dict)


class SearchResult(BaseModel):
    job_id: str
    status: JobStatus
    query: str
    created_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None
    results: List[SignalDoc] = Field(default_factory=list)
    rejected: List[SignalDoc] = Field(default_factory=list)
    stats: SearchStats = Field(default_factory=SearchStats)


class SourcesResponse(BaseModel):
    total: int
    items: List[Source] = Field(default_factory=list)


class CollectRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    area: Optional[str] = None
    limit: int = Field(30, ge=1, le=60)


class CollectResponse(BaseModel):
    query: str
    sources_processed: int = 0
    documents: List[SignalDoc] = Field(default_factory=list)
    connector_status: dict = Field(default_factory=dict)
    retrieval_mode: str = "live"


class ScoreRequest(BaseModel):
    document: SignalDoc
    query: Optional[str] = None


class ScoreResponse(BaseModel):
    id: str
    score: float
    is_weak_signal: bool
    why: str
    rejected_reason: Optional[str] = None
    predictors: List[str] = Field(default_factory=list)
    dataset_score: Optional[int] = None
    model_version: str = "heuristic-v2"
    model_mode: str = "heuristic"


class ScoreBatchRequest(BaseModel):
    documents: List[SignalDoc] = Field(default_factory=list)
    query: Optional[str] = None


class ScoreBatchResponse(BaseModel):
    items: List[ScoreResponse] = Field(default_factory=list)


class InsightRequest(BaseModel):
    documents: List[SignalDoc] = Field(default_factory=list)
    query: str = ""


class InsightResponse(BaseModel):
    documents: List[SignalDoc] = Field(default_factory=list)
    model: Optional[str] = None
    mode: str = "grounded-template"


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str = "1.0.0"
    mock: bool = False
    details: dict = Field(default_factory=dict)
