"""Нормализация результатов коннекторов в общую схему SignalDoc (CLAUDE.md §3).

Задачи:
  * маппинг raw-dict'ов коннекторов -> SignalDoc с полностью заполненными Source;
  * канонизация URL и дедуп: два коннектора не должны дублировать один документ;
  * стабильный id документа (не зависит от порядка/дублей).
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from shared.schema import SignalDoc, Source, SourceType, TrustLevel

logger = logging.getLogger("parser.normalize")

_WS_RE = re.compile(r"\s+")
_SLUG_RE = re.compile(r"[^a-zа-яё0-9]+")


def clean_text(text: str) -> str:
    """Один нормализованный текст: сжать пробелы, убрать хвостовые пустоты."""
    return _WS_RE.sub(" ", text or "").strip()


def canonical_url(url: str) -> str:
    """Каноническая форма URL для дедупа и ключей кэша.

    Делает http->https (кроме localhost), убирает фрагмент и концевой слэш,
    сортирует query-параметры (некоторые источники отдают их в разном порядке).
    """
    url = (url or "").strip()
    if not url:
        return ""
    url = url.split("#", 1)[0]
    if url.startswith("http://") and "localhost" not in url and "127.0.0.1" not in url:
        url = "https://" + url[len("http://"):]
    if "://" in url:
        scheme, rest = url.split("://", 1)
        path, _, query = rest.partition("?")
        path = path.rstrip("/")
        if query:
            params = sorted(query.split("&"))
            query = "&".join(params)
            url = f"{scheme}://{path}?{query}"
        else:
            url = f"{scheme}://{path}"
    return url


def _slug(title: str) -> str:
    s = _SLUG_RE.sub("-", title.lower()).strip("-")
    return s[:48] or "signal"


def _id_for(url: str, title: str) -> str:
    digest = hashlib.md5(canonical_url(url).encode("utf-8")).hexdigest()[:8]
    return f"{_slug(title)}-{digest}"


def build_source(
    title: str,
    url: str,
    date: Optional[str],
    source_type: SourceType,
    trust_level: TrustLevel,
    language: str,
    translated: bool = False,
) -> Optional[Source]:
    """Собирает Source из сырых данных. None, если не хватает обязательных полей (§2.7)."""
    title = clean_text(title)
    url = canonical_url(url)
    if not title or not url:
        logger.debug("Источник отброшен: пустые title/url (%r, %r)", title, url)
        return None
    from pydantic import HttpUrl

    try:
        return Source(
            title=title,
            url=HttpUrl(url),
            date=date,
            source_type=source_type,
            language=language,
            trust_level=trust_level,
            translated=translated,
        )
    except Exception as exc:  # noqa: BLE001 — невалидный URL не роняет пайплайн
        logger.warning("Источник отброшен (невалидный URL «%s»): %s", url, exc)
        return None


def raw_to_signal_doc(raw: Dict[str, Any], query: Optional[str] = None, area: Optional[str] = None) -> Optional[SignalDoc]:
    """Превращает один raw-dict коннектора в SignalDoc или None (если данных мало)."""
    title = clean_text(raw.get("title", ""))
    url = raw.get("url", "")
    if not title or not url:
        logger.debug("Документ отброшен: пустые title/url (%r, %r)", title, url)
        return None
    raw_text = clean_text(raw.get("raw_text", ""))
    if not raw_text:
        logger.debug("Документ отброшен (нет текста), id=%s", raw.get("doc_id", "?"))
        return None

    translated = bool(raw.get("translated", False))
    source = build_source(
        title=raw.get("source_title") or title,
        url=url,
        date=raw.get("date"),
        source_type=raw.get("source_type", SourceType.NEWS),
        trust_level=raw.get("trust_level", TrustLevel.MEDIUM),
        language=raw.get("language") or "en",
        translated=translated,
    )
    if source is None:
        return None

    # Кандидат, привязанный к запросу: в raw_text фиксируем запрос (прозрачность для ML/LLM).
    text = raw_text if not query else "{} [запрос исследования: «{}»]".format(raw_text, query)

    return SignalDoc(
        id=_id_for(url, title),
        title=title,
        area=area or raw.get("area"),
        companies=list(raw.get("companies") or []),
        raw_text=text,
        stage=raw.get("stage"),
        trend=raw.get("trend"),
        score=raw.get("score"),
        why=raw.get("why"),
        is_weak_signal=raw.get("is_weak_signal"),
        rejected_reason=raw.get("rejected_reason"),
        sources=[source],
    )


def normalize_all(
    raw_results: Iterable[Dict[str, Any]],
    query: Optional[str] = None,
    area: Optional[str] = None,
) -> List[SignalDoc]:
    """Сырые результаты всех коннекторов -> список SignalDoc (после дедупа)."""
    docs = [d for r in raw_results if (d := raw_to_signal_doc(r, query=query, area=area))]
    return dedupe_documents(docs)


def dedupe_documents(docs: List[SignalDoc]) -> List[SignalDoc]:
    """Дедуп по каноническому URL первого источника.

    При дубле объединяем источники: если у двух документов один URL — это один
    документ; сохраняем тот, у которого больше источников/ботаче метаданные.
    """
    by_url: Dict[str, SignalDoc] = {}
    order: List[str] = []
    for doc in docs:
        if not doc.sources:
            continue
        key = canonical_url(str(doc.sources[0].url))
        if not key:
            continue
        if key in by_url:
            prev = by_url[key]
            if len(doc.sources) > len(prev.sources) or len(doc.raw_text) > len(prev.raw_text):
                by_url[key] = doc
        else:
            by_url[key] = doc
            order.append(key)

    seen: List[SignalDoc] = [by_url[k] for k in order]
    logger.info("Дедуп: на входе %d, после дедупа %d", len(docs), len(seen))
    return seen


def merge_sources(docs: Iterable[SignalDoc]) -> List[Source]:
    """Уникальный плоский список источников по всем документам (для /sources)."""
    out: List[Source] = []
    keys: set = set()
    for d in docs:
        for s in d.sources:
            key = canonical_url(str(s.url))
            if key and key not in keys:
                keys.add(key)
                out.append(s)
    return out