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


async def _request_with_429_retry(client, url, *, params=None, attempts=3):
    retry_statuses = {429, 500, 502, 503, 504}

    for attempt in range(1, attempts + 1):
        try:
            r = await client.get(url, params=params)
        except Exception as exc:
            if attempt >= attempts:
                raise

            wait = min(8, 2 ** (attempt - 1))
            logger.warning(
                "request error: url=%s attempt=%d/%d error=%s retry_in=%ss",
                url,
                attempt,
                attempts,
                type(exc).__name__,
                wait,
            )
            await asyncio.sleep(wait)
            continue

        if r.status_code not in retry_statuses:
            return r

        if attempt >= attempts:
            logger.warning(
                "request failed after retries: url=%s status=%d",
                url,
                r.status_code,
            )
            return r

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            try:
                wait = int(float(retry_after)) if retry_after else 5
            except (TypeError, ValueError):
                wait = 5

            wait = min(max(wait, 2), 10)
        else:
            wait = min(8, 2 ** (attempt - 1))

        logger.warning(
            "HTTP %d: url=%s attempt=%d/%d retry_in=%ss",
            r.status_code,
            url,
            attempt,
            attempts,
            wait,
        )

        await asyncio.sleep(wait)

    return r


def _http(timeout: float):
    return httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=15), headers={"User-Agent": UA}, follow_redirects=True)


async def arxiv(query: str, limit: int) -> Tuple[List[RawHit], str]:
    url = "https://export.arxiv.org/api/query"

    # Поиск не должен вызывать LLM. Ранее arXiv сначала просил GigaChat
    # расширить запрос, а затем parser ещё раз вызывал LLM для rerank/validation.
    # При лимите провайдера это превращало один search в серию 429.
    # Расширение теперь детерминированное: оно не создаёт сетевых обращений к LLM.
    queries = ['all:"{}"'.format(" ".join(query.split()))]
    low = query.lower().replace("ё", "е")

    if any(x in low for x in ("промышлен", "производств", "industrial", "manufactur", "factory")) and any(x in low for x in ("ии", "ai", "искусствен", "artificial", "intelligence")):
        queries.append('all:"industrial" AND all:"artificial" AND all:"intelligence"')
        queries.append('all:"manufacturing" AND all:"artificial" AND all:"intelligence"')
    elif any(x in low for x in ("робот", "robot", "автоматизац", "automation")):
        queries.append('all:"robotics" AND all:"automation"')
    elif any(x in low for x in ("квант", "quantum")):
        queries.append('all:"quantum"')

    queries = list(dict.fromkeys(queries))[:3]

    logger.info(
        "arXiv search queries: original=%r final=%s (LLM expansion disabled)",
        query,
        queries,
    )

    try:
        async with _http(30) as client:
            for search_query in queries:
                r = await _request_with_429_retry(
                    client,
                    url,
                    params={
                        "search_query": search_query,
                        "start": 0,
                        "max_results": limit,
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                    },
                    attempts=2,
                )
                r.raise_for_status()

                root = ET.fromstring(r.text)
                ns = {"a": "http://www.w3.org/2005/Atom"}
                out = []

                for entry in root.findall("a:entry", ns):
                    title = _clean(
                        entry.findtext("a:title", default="", namespaces=ns)
                    )
                    abstract = _clean(
                        entry.findtext("a:summary", default="", namespaces=ns)
                    )
                    page = entry.findtext(
                        "a:id", default="", namespaces=ns
                    )
                    published = _iso_date(
                        entry.findtext(
                            "a:published", default="", namespaces=ns
                        )
                    )

                    if title and page:
                        out.append(
                            RawHit(
                                "arxiv",
                                title,
                                abstract,
                                page,
                                published,
                                SourceType.SCIENTIFIC,
                                "en",
                            )
                        )

                if out:
                    logger.info(
                        "arXiv: query=%r items=%d",
                        search_query,
                        len(out),
                    )
                    return out, "ok"

        return [], "ok: результатов нет"

    except Exception as exc:
        logger.warning("arXiv: %s: %s", type(exc).__name__, exc)
        return [], "ошибка: {}".format(type(exc).__name__)


async def openalex(query: str, limit: int) -> Tuple[List[RawHit], str]:
    url = "https://api.openalex.org/works"

    try:
        async with _http(30) as client:
            r = await _request_with_429_retry(
                client,
                url,
                params={
                    "search": query,
                    "per-page": limit,
                    "sort": "publication_date:desc",
                },
                attempts=3,
            )
            r.raise_for_status()
            data = r.json()

        out = []

        for item in data.get("results", []):
            title = _clean(item.get("display_name") or "")

            inv = item.get("abstract_inverted_index") or {}
            pairs = []

            for token, positions in inv.items():
                for pos in positions:
                    pairs.append((pos, token))

            abstract = " ".join(
                token for _, token in sorted(pairs)
            )

            loc = item.get("primary_location") or {}
            page = (
                loc.get("landing_page_url")
                or loc.get("pdf_url")
                or item.get("doi")
            )

            if not title or not page:
                continue

            inst = []

            for author in item.get("authorships", [])[:5]:
                for institution in author.get("institutions", [])[:2]:
                    name = institution.get("display_name")
                    if name:
                        inst.append(name)

            out.append(
                RawHit(
                    "openalex",
                    title,
                    _clean(abstract),
                    page,
                    item.get("publication_date"),
                    SourceType.SCIENTIFIC,
                    "en",
                    companies=inst,
                )
            )

        logger.info("OpenAlex: items=%d", len(out))
        return out, "ok"

    except Exception as exc:
        logger.warning(
            "OpenAlex: %s: %s",
            type(exc).__name__,
            exc,
        )
        return [], "ошибка: {}".format(type(exc).__name__)


async def crossref(query: str, limit: int) -> Tuple[List[RawHit], str]:
    url = "https://api.crossref.org/works"

    try:
        async with _http(30) as client:
            r = await _request_with_429_retry(
                client,
                url,
                params={
                    "query.bibliographic": query,
                    "rows": limit,
                    "select": "DOI,title,abstract,published,URL,type",
                },
                attempts=3,
            )
            r.raise_for_status()
            data = r.json()

        out = []

        for item in data.get("message", {}).get("items", []):
            title = _clean(
                " ".join(item.get("title") or [])
            )

            abstract = _clean(
                re.sub(
                    r"<[^>]+>",
                    " ",
                    item.get("abstract", ""),
                )
            )

            page = item.get("URL")

            if not page and item.get("DOI"):
                page = "https://doi.org/" + item["DOI"]

            date_parts = (
                item.get("published") or {}
            ).get("date-parts") or []

            date = None

            if date_parts and date_parts[0]:
                parts = date_parts[0]
                date = "-".join(
                    str(x).zfill(2)
                    for x in (parts + [1, 1])[:3]
                )

            if title and page:
                out.append(
                    RawHit(
                        "crossref",
                        title,
                        abstract,
                        page,
                        date,
                        SourceType.SCIENTIFIC,
                        "en",
                    )
                )

        logger.info("Crossref: items=%d", len(out))
        return out, "ok"

    except Exception as exc:
        logger.warning(
            "Crossref: %s: %s",
            type(exc).__name__,
            exc,
        )
        return [], "ошибка: {}".format(type(exc).__name__)



async def collect_all(query: str, limit_per_source: int = 50):
    """Собирает расширенный пул из открытых источников.

    Каждый источник получает до ``limit_per_source`` результатов. Если провайдер
    вернул пустой список, это считаем временной деградацией и повторяем запрос
    ещё до двух раз. Обычные HTTP-ошибки уже ретраятся внутри коннекторов.
    """
    runners = {
        "arxiv": arxiv,
        "openalex": openalex,
        "crossref": crossref,
    }
    hits: List[RawHit] = []
    status: Dict[str, str] = {}

    for name, runner in runners.items():
        items: List[RawHit] = []
        state = "не запускался"
        for attempt in range(1, 4):
            try:
                items, state = await runner(query, limit_per_source)
            except Exception as exc:
                logger.warning("collect_all: source=%s exception=%s attempt=%d/3", name, exc, attempt)
                items, state = [], "ошибка: {}".format(type(exc).__name__)

            if items:
                if attempt > 1:
                    logger.info("collect_all: source=%s recovered on retry %d items=%d", name, attempt, len(items))
                break

            if attempt < 3:
                wait = 2 * attempt
                logger.warning(
                    "collect_all: source=%s returned 0 results; retry=%d/3 in %ss",
                    name, attempt + 1, wait,
                )
                await asyncio.sleep(wait)

        logger.info(
            "collect_all: source=%s items=%d state=%s limit=%d attempts<=3",
            name, len(items), state, limit_per_source,
        )

        if items:
            hits.extend(items)
        status[name] = state

        await asyncio.sleep(1)

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
