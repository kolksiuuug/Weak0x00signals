"""ИБ / trust-слой (Участник 1) — уникальный вклад, оценивается жюри отдельно.

Отвечает за:
  * классификацию доверенности «тип источника + домен → TrustLevel»;
  * дедупликацию источников по URL;
  * правило CLAUDE.md §2.8: технология не может попасть в выдачу, если ВСЕ её источники
    имеют пониженную доверенность;
  * проверку полноты метаданных (§2.7) и пометки автоперевода (§2.9).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from shared.schema import LOW_TRUST_SOURCE_TYPES, SignalDoc, Source, SourceType, TrustLevel

logger = logging.getLogger("api.trust")

#: Домены с заведомо высоким уровнем доверия: гос/регуляторы/университеты/научные базы/патенты.
HIGH_TRUST_DOMAIN_SUFFIXES = (
    ".gov", ".gov.ru", ".edu", ".ac.uk", ".ac.jp",
    "arxiv.org", "doi.org", "nature.com", "science.org", "ieee.org",
    "patents.google.com", "patentsview.org", "openalex.org", "crossref.org",
    "cbr.ru", "rospatent.gov.ru", "minobrnauki.gov.ru", "elibrary.ru",
)

#: Домены-агрегаторы, соцсети и площадки личных блогов — пониженная доверенность.
LOW_TRUST_DOMAIN_SUFFIXES = (
    "t.me", "telegram.me", "x.com", "twitter.com", "vk.com", "facebook.com",
    "linkedin.com", "reddit.com", "medium.com", "habr.com", "dzen.ru",
    "youtube.com", "tiktok.com",
)


def domain_of(url: str) -> str:
    host = (urlparse(str(url)).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def classify(source_type: SourceType, url: str) -> TrustLevel:
    """Правило «тип + домен → уровень доверенности» (CLAUDE.md §2.8).

    Тип источника имеет приоритет: соцсеть/блог/пресс-релиз всегда пониженные,
    на каком бы домене они ни лежали.
    """
    if source_type in LOW_TRUST_SOURCE_TYPES:
        return TrustLevel.LOW
    host = domain_of(url)
    if any(host.endswith(sfx) for sfx in LOW_TRUST_DOMAIN_SUFFIXES):
        return TrustLevel.LOW
    if source_type in (SourceType.SCIENTIFIC, SourceType.PATENT, SourceType.REGISTRY):
        return TrustLevel.HIGH
    if any(host.endswith(sfx) for sfx in HIGH_TRUST_DOMAIN_SUFFIXES):
        return TrustLevel.HIGH
    return TrustLevel.MEDIUM


def dedup_sources(sources: List[Source]) -> List[Source]:
    """Дедуп по нормализованному URL, порядок сохраняется."""
    seen = set()
    out: List[Source] = []
    for src in sources:
        key = str(src.url).rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(src)
    return out


def normalize_sources(sources: List[Source]) -> List[Source]:
    """Дедуп + пересчёт trust_level по нашему правилу (не доверяем метке парсера вслепую)."""
    out: List[Source] = []
    for src in dedup_sources(sources):
        level = classify(src.source_type, str(src.url))
        if level != src.trust_level:
            logger.info(
                "trust: %s — уровень пересчитан %s → %s", domain_of(src.url), src.trust_level.value, level.value
            )
            src = src.model_copy(update={"trust_level": level})
        out.append(src)
    return out


def metadata_issues(src: Source) -> List[str]:
    """Проверка полноты метаданных источника (CLAUDE.md §2.7, §2.9)."""
    issues: List[str] = []
    if not src.title.strip():
        issues.append("нет наименования")
    if not src.date:
        issues.append("нет даты публикации")
    if not src.language.strip():
        issues.append("не указан язык оригинала")
    if src.language != "ru" and not src.translated:
        issues.append("зарубежный источник без пометки об автопереводе/резюме")
    return issues


def check_document(doc: SignalDoc) -> Tuple[SignalDoc, Optional[str]]:
    """Прогоняет документ через trust-слой.

    Возвращает (документ с нормализованными источниками, причина отклонения или None).
    """
    doc = doc.model_copy(update={"sources": normalize_sources(doc.sources)})

    if not doc.sources:
        return doc, "Отклонено trust-слоем: нет ни одного подтверждённого источника (CLAUDE.md §2.3)."

    if all(src.trust_level == TrustLevel.LOW for src in doc.sources):
        kinds = ", ".join(sorted({src.source_type.value for src in doc.sources}))
        return doc, (
            "Отклонено trust-слоем: все источники пониженной доверенности ({}). "
            "Такие источники не могут быть единственным основанием для включения "
            "технологии в выдачу (CLAUDE.md §2.8).".format(kinds)
        )

    broken = [(src, metadata_issues(src)) for src in doc.sources]
    broken = [(src, iss) for src, iss in broken if iss]
    if broken and len(broken) == len(doc.sources):
        details = "; ".join("{}: {}".format(domain_of(s.url), ", ".join(i)) for s, i in broken)
        return doc, "Отклонено trust-слоем: у всех источников неполные метаданные ({}).".format(details)

    return doc, None
