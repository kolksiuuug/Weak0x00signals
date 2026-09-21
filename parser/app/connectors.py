"""Реальные коннекторы открытых источников. Каждый коннектор возвращает только фактически найденные данные."""
from __future__ import annotations

import asyncio
import email.utils
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import httpx

from shared.schema import SignalDoc, Source, SourceType, TrustLevel

logger = logging.getLogger("parser.connectors")

UA = "Weak0x00Signals/1.0 (+hackathon-research)"
AREAS = ["Индустриальный ИИ", "Инфраструктура ИИ", "Роботы", "Финтех", "Защита ИИ", "Edge"]


@dataclass
class RawHit:
    connector: str
    title: str
    text: str
    url: str
    date: Optional[str]
    source_type: SourceType
    language: str = "en"
    area: Optional[str] = None
    companies: Optional[List[str]] = None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(text or "")).strip()


def _iso_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = value.strip()
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d", "%Y-%m-%d", "%a, %d %b %Y %H:%M:%S %z"):
        try:
            return datetime.strptime(raw, fmt).astimezone(timezone.utc).date().isoformat()
        except ValueError:
            pass
    return raw[:10] if re.match(r"\d{4}-\d{2}-\d{2}", raw) else None


def _http(timeout: float):
    return httpx.AsyncClient(timeout=timeout, headers={"User-Agent": UA}, follow_redirects=True)


async def arxiv(query: str, limit: int) -> Tuple[List[RawHit], str]:
    url = "https://export.arxiv.org/api/query"
    try:
        async with _http(15) as client:
            r = await client.get(url, params={"search_query": 'all:"{}"'.format(query), "start": 0, "max_results": limit, "sortBy": "submittedDate", "sortOrder": "descending"})
            r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for entry in root.findall("a:entry", ns):
            title = _clean(entry.findtext("a:title", default="", namespaces=ns))
            abstract = _clean(entry.findtext("a:summary", default="", namespaces=ns))
            page = entry.findtext("a:id", default="", namespaces=ns)
            published = _iso_date(entry.findtext("a:published", default="", namespaces=ns))
            if title and page:
                out.append(RawHit("arxiv", title, abstract, page, published, SourceType.SCIENTIFIC, "en"))
        return out, "ok"
    except Exception as exc:  # noqa: BLE001
        logger.warning("arXiv: %s", exc)
        return [], "ошибка: {}".format(type(exc).__name__)


async def openalex(query: str, limit: int) -> Tuple[List[RawHit], str]:
    url = "https://api.openalex.org/works"
    try:
        async with _http(15) as client:
            r = await client.get(url, params={"search": query, "per-page": limit, "sort": "publication_date:desc"})
            r.raise_for_status()
            data = r.json()
        out = []
        for item in data.get("results", []):
            title = _clean(item.get("display_name"))
            inv = item.get("abstract_inverted_index") or {}
            abstract = " ".join(sorted(((pos, token) for token, positions in inv.items() for pos in positions), key=lambda x: x[0]) and [])
            # Обход выше намеренно заменяем ниже: позиции разрежены и могут иметь дубликаты.
            if inv:
                pairs = []
                for token, positions in inv.items():
                    for pos in positions:
                        pairs.append((pos, token))
                abstract = " ".join(token for _, token in sorted(pairs))
            loc = item.get("primary_location") or {}
            page = loc.get("landing_page_url") or loc.get("pdf_url") or item.get("doi")
            if not page:
                continue
            inst = []
            for author in item.get("authorships", [])[:5]:
                for institution in author.get("institutions", [])[:2]:
                    name = institution.get("display_name")
                    if name:
                        inst.append(name)
            out.append(RawHit("openalex", title, _clean(abstract), page, item.get("publication_date"), SourceType.SCIENTIFIC, "en", companies=inst))
        return out, "ok"
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenAlex: %s", exc)
        return [], "ошибка: {}".format(type(exc).__name__)


async def crossref(query: str, limit: int) -> Tuple[List[RawHit], str]:
    url = "https://api.crossref.org/works"
    try:
        async with _http(15) as client:
            r = await client.get(url, params={"query.bibliographic": query, "rows": limit, "select": "DOI,title,abstract,published,URL,type"})
            r.raise_for_status()
            data = r.json()
        out = []
        for item in data.get("message", {}).get("items", []):
            title = _clean(" ".join(item.get("title") or []))
            abstract = _clean(re.sub(r"<[^>]+>", " ", item.get("abstract", "")))
            page = item.get("URL") or ("https://doi.org/" + item["DOI"] if item.get("DOI") else "")
            date_parts = (item.get("published") or {}).get("date-parts") or []
            date = None
            if date_parts and date_parts[0]:
                date = "-".join(str(x).zfill(2) for x in (date_parts[0] + [1, 1])[:3])
            if title and page:
                out.append(RawHit("crossref", title, abstract, page, date, SourceType.SCIENTIFIC, "en"))
        return out, "ok"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Crossref: %s", exc)
        return [], "ошибка: {}".format(type(exc).__name__)


async def gdelt(query: str, limit: int) -> Tuple[List[RawHit], str]:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    try:
        async with _http(15) as client:
            r = await client.get(url, params={"query": query, "mode": "artlist", "maxrecords": min(limit, 50), "format": "json", "sort": "HybridRel"})
            r.raise_for_status()
            data = r.json()
        out = []
        for item in data.get("articles", []):
            title = _clean(item.get("title"))
            page = item.get("url")
            if not title or not page:
                continue
            text = "{} {}".format(title, item.get("domain", ""))
            out.append(RawHit("gdelt", title, _clean(text), page, _iso_date(item.get("seendate")), SourceType.NEWS, str(item.get("language", "en")).lower()))
        return out, "ok"
    except Exception as exc:  # noqa: BLE001
        logger.warning("GDELT: %s", exc)
        return [], "ошибка: {}".format(type(exc).__name__)


async def patentsview(query: str, limit: int) -> Tuple[List[RawHit], str]:
    # PatentsView использует официальный API; при отсутствии ключа коннектор штатно отключается.
    api_key = os.getenv("PATENTSVIEW_API_KEY")
    if not api_key:
        return [], "пропущен: PATENTSVIEW_API_KEY не задан"
    url = "https://search.patentsview.org/api/v1/patent/"
    try:
        async with _http(15) as client:
            r = await client.get(url, params={"q": '{{"_text_any": {{"patent_title": "{}"}}}}'.format(query), "f": '["patent_id","patent_title","patent_date"]', "o": '{{"size":{}}}'.format(limit), "api_key": api_key})
            r.raise_for_status()
            data = r.json()
        out = []
        for item in data.get("patents", []):
            title = _clean(item.get("patent_title"))
            pid = item.get("patent_id")
            if title and pid:
                out.append(RawHit("patentsview", title, title, "https://patents.google.com/patent/{}".format(pid), _iso_date(item.get("patent_date")), SourceType.PATENT, "en"))
        return out, "ok"
    except Exception as exc:  # noqa: BLE001
        logger.warning("PatentsView: %s", exc)
        return [], "ошибка: {}".format(type(exc).__name__)


async def collect_all(query: str, limit_per_source: int = 8):
    tasks = [
        arxiv(query, limit_per_source),
        openalex(query, limit_per_source),
        crossref(query, limit_per_source),
        gdelt(query, limit_per_source),
        patentsview(query, limit_per_source),
    ]
    results = await asyncio.gather(*tasks)
    hits: List[RawHit] = []
    status: Dict[str, str] = {}
    for connector, (items, state) in zip(("arxiv", "openalex", "crossref", "gdelt", "patentsview"), results):
        hits.extend(items)
        status[connector] = state
    return hits, status


async def fetch_article_text(url: str) -> str:
    """Извлекает основной текст только для найденной ссылки. Ошибка не ломает весь конвейер."""
    try:
        import trafilatura
        async with _http(10) as client:
            r = await client.get(url)
            r.raise_for_status()
        text = trafilatura.extract(r.text, include_comments=False, include_tables=False) or ""
        return _clean(text)[:7000]
    except Exception:
        return ""
