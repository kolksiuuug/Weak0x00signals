"""Канонический контракт данных между сервисами."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class SourceType(str, Enum):
    SCIENTIFIC = "научная_статья"
    PATENT = "патент"
    NEWS = "новость"
    ANALYTICS = "аналитический_отчёт"
    REGISTRY = "гос_реестр"
    BLOG = "блог"
    SOCIAL = "соцсеть"
    PRESS = "пресс_релиз"


class TrustLevel(str, Enum):
    HIGH = "высокий"
    MEDIUM = "средний"
    LOW = "пониженный"


LOW_TRUST_SOURCE_TYPES = frozenset({SourceType.SOCIAL, SourceType.BLOG, SourceType.PRESS})


class Source(BaseModel):
    title: str
    url: HttpUrl
    date: Optional[str] = None
    source_type: SourceType
    language: str
    trust_level: TrustLevel
    translated: bool = False


class SignalDoc(BaseModel):
    id: str
    title: str
    area: Optional[str] = None
    companies: List[str] = Field(default_factory=list)
    raw_text: str
    stage: Optional[int] = None
    trend: Optional[int] = None
    score: Optional[float] = None
    why: Optional[str] = None
    is_weak_signal: Optional[bool] = None
    rejected_reason: Optional[str] = None
    sources: List[Source] = Field(default_factory=list)
    # Поля итоговой карточки. Они не меняют смысл исходных полей и нужны для ТЗ:
    description: Optional[str] = None
    advantage: Optional[str] = None
    case_example: Optional[str] = None
    evidence_summary: Optional[str] = None
    # Явные поля карточки сигнала: отделяют релевантность запросу от объяснения слабости.
    relevance_reason: Optional[str] = None
    signal_markers: List[str] = Field(default_factory=list)
    source_agreement: Optional[str] = None
    dataset_score: Optional[int] = None
    retrieval_score: Optional[float] = None
    model_version: Optional[str] = None
    model_mode: Optional[str] = None


__all__ = [
    "SourceType", "TrustLevel", "LOW_TRUST_SOURCE_TYPES", "Source", "SignalDoc",
]
