"""ГРАНИЦА СЛИЯНИЯ. Канонический контракт данных из CLAUDE.md §3.

Правила изменения (CLAUDE.md §5):
  * этот файл — единственный источник правды по схеме;
  * менять только по согласованию всей команды в общем чате + правка CLAUDE.md §3;
  * НЕ добавлять сюда служебные DTO конкретных сервисов — для них есть shared/contracts.py.

Совместимость: Python 3.8+ (см. CLAUDE.md §2.1), поэтому аннотации пишем через typing.List /
typing.Optional, а не через list[...] | None.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class SourceType(str, Enum):
    """Тип источника (CLAUDE.md §2.7 — обязательное поле метаданных)."""

    SCIENTIFIC = "научная_статья"
    PATENT = "патент"
    NEWS = "новость"
    ANALYTICS = "аналитический_отчёт"
    REGISTRY = "гос_реестр"
    BLOG = "блог"
    SOCIAL = "соцсеть"
    PRESS = "пресс_релиз"


class TrustLevel(str, Enum):
    """Уровень доверенности источника (CLAUDE.md §2.8)."""

    HIGH = "высокий"        # гос/регуляторы/университеты/научные публикации/патенты
    MEDIUM = "средний"      # отраслевые медиа, аналитика
    LOW = "пониженный"      # соцсети/блоги/пресс-релизы — не единственное основание


#: Типы источников, которые сами по себе не могут быть единственным основанием
#: для включения технологии в выдачу (CLAUDE.md §2.8).
LOW_TRUST_SOURCE_TYPES = frozenset(
    {SourceType.SOCIAL, SourceType.BLOG, SourceType.PRESS}
)


class Source(BaseModel):
    """Один проверенный источник. Полный набор метаданных обязателен (CLAUDE.md §2.7)."""

    title: str
    url: HttpUrl
    date: Optional[str] = None          # ISO 8601 или None, если даты нет
    source_type: SourceType
    language: str                       # "ru", "en", ...
    trust_level: TrustLevel
    translated: bool = False            # True → есть автоперевод/генеративное резюме (пометка §2.9)


class SignalDoc(BaseModel):
    """Технология-кандидат в слабые сигналы — единица обмена между всеми модулями."""

    id: str
    title: str                                  # название технологии-кандидата
    area: Optional[str] = None                  # одна из 6 областей или свободная тема
    companies: List[str] = Field(default_factory=list)
    raw_text: str                               # нормализованный текст источника(ов)
    stage: Optional[int] = None                 # 1..4 (см. таблицу инсайта CLAUDE.md §1)
    trend: Optional[int] = None                 # 1..3
    score: Optional[float] = None               # уверенность модели / балл
    why: Optional[str] = None                   # объяснение: почему это слабый сигнал
    is_weak_signal: Optional[bool] = None
    rejected_reason: Optional[str] = None       # причина, если отсеян фильтром зрелости/хайпа
    sources: List[Source] = Field(default_factory=list)


__all__ = [
    "SourceType",
    "TrustLevel",
    "LOW_TRUST_SOURCE_TYPES",
    "Source",
    "SignalDoc",
]
