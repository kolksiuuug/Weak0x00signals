"""ИБ/trust-слой: классификация, дедупликация и строгая проверка метаданных."""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from shared.schema import LOW_TRUST_SOURCE_TYPES, SignalDoc, Source, SourceType, TrustLevel

logger = logging.getLogger("api.trust")
HIGH_TRUST_DOMAIN_SUFFIXES = (
    ".gov", ".gov.ru", ".edu", ".ac.uk", ".ac.jp", "arxiv.org", "doi.org",
    "nature.com", "science.org", "ieee.org", "patents.google.com", "patentsview.org",
    "openalex.org", "crossref.org", "cbr.ru", "rospatent.gov.ru", "elibrary.ru",
)
LOW_TRUST_DOMAIN_SUFFIXES = (
    "t.me", "telegram.me", "x.com", "twitter.com", "vk.com", "facebook.com",
    "linkedin.com", "reddit.com", "medium.com", "habr.com", "dzen.ru", "youtube.com", "tiktok.com",
)


def domain_of(url: str) -> str:
    host = (urlparse(str(url)).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def classify(source_type: SourceType, url: str) -> TrustLevel:
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
    seen = set()
    out = []
    for src in sources:
        key = str(src.url).rstrip("/").lower()
        if key not in seen:
            seen.add(key)
            out.append(src)
    return out


def normalize_sources(sources: List[Source]) -> List[Source]:
    out = []
    for src in dedup_sources(sources):
        level = classify(src.source_type, str(src.url))
        out.append(src.model_copy(update={"trust_level": level}))
    return out


def metadata_issues(src: Source) -> List[str]:
    issues = []
    if not src.title.strip():
        issues.append("нет наименования")
    if not src.date:
        issues.append("нет даты публикации")
    if not src.language.strip():
        issues.append("не указан язык оригинала")
    if src.language != "ru" and not src.translated:
        issues.append("нет русского резюме/пометки об автопереводе")
    return issues


def check_document(doc: SignalDoc) -> Tuple[SignalDoc, Optional[str]]:
    sources = normalize_sources(doc.sources)
    if not sources:
        return doc.model_copy(update={"sources": []}), "Отклонено trust-слоем: нет подтверждённых источников."
    broken = [(s, metadata_issues(s)) for s in sources]
    broken = [(s, issues) for s, issues in broken if issues]
    if broken:
        details = "; ".join("{}: {}".format(domain_of(s.url), ", ".join(issues)) for s, issues in broken)
        return doc.model_copy(update={"sources": sources}), "Отклонено trust-слоем: неполные метаданные источников ({})".format(details)
    if all(s.trust_level == TrustLevel.LOW for s in sources):
        return doc.model_copy(update={"sources": sources}), "Отклонено trust-слоем: все источники имеют пониженную доверенность."
    return doc.model_copy(update={"sources": sources}), None
