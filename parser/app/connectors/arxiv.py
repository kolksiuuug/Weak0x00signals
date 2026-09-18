"""Коннектор arXiv (Atom API, export.arxiv.org).

Официальный API arXiv без ключа: GET export.arxiv.org/api/query?search_query=...
Ответ — Atom XML. Парсим stdlib xml.etree (без лишних зависимостей).

Нормализация: каждый результат — raw-dict для `parse/schema.py`-normalize.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

from parser.app.config import settings
from parser.app.connectors import BaseConnector, registry
from shared.schema import SourceType, TrustLevel

logger = logging.getLogger("parser.connectors.arxiv")

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"

# arXiv id из canonical-ссылки (для стабильного doc_id и каноничного URL).
_ARXIV_ID_RE = re.compile(r"/abs/([^/]+)")

_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip()


class ArxivConnector(BaseConnector):
    name = "arxiv"
    source_type = SourceType.SCIENTIFIC
    trust_level = TrustLevel.HIGH
    language_default = "en"
    timeout = settings.arxiv_timeout

    def __init__(self) -> None:
        super().__init__()
        self.api_url = settings.arxiv_api_url

    async def _search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        query = _clean(query)
        if not query:
            return []
        params = {
            "search_query": f"all:\"{query}\"",
            "start": "0",
            "max_results": str(max(1, min(limit, settings.max_per_connector))),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        async with self._client() as client:
            resp = await client.get(self.api_url, params=params)
            resp.raise_for_status()
            content = resp.content

        root = ET.fromstring(content)
        out: List[Dict[str, Any]] = []
        for entry in root.findall(f"{{{ATOM_NS}}}entry"):
            item = self._parse_entry(entry)
            if item:
                out.append(item)
        logger.info("arXiv: запрос=%r, результатов=%d", query, len(out))
        return out

    def _parse_entry(self, entry: ET.Element) -> Dict[str, Any]:
        title = _clean(entry.findtext(f"{{{ATOM_NS}}}title") or "")
        url = ""
        for link in entry.findall(f"{{{ATOM_NS}}}link"):
            rel = link.get("rel")
            href = link.get("href")
            if rel == "alternate" and href:
                url = href
                break
        if not url:
            id_node = entry.find(f"{{{ATOM_NS}}}id")
            url = _clean(id_node.text) if id_node is not None and id_node.text else ""
        if not title or not url:
            return {}

        m = _ARXIV_ID_RE.search(url)
        arxiv_id = m.group(1) if m else url.rsplit("/", 1)[-1]
        # Каноничный arXiv URL без версии (v1/v2/v3 == один документ для дедупа).
        base_id = arxiv_id.split("v")[0]
        canonical_url = f"https://arxiv.org/abs/{base_id}"

        summary = _clean(entry.findtext(f"{{{ATOM_NS}}}summary") or "")
        authors = [
            _clean(a.findtext(f"{{{ATOM_NS}}}name") or "")
            for a in entry.findall(f"{{{ATOM_NS}}}author")
            if a.findtext(f"{{{ATOM_NS}}}name")
        ]

        categories: List[str] = []
        for c in entry.findall(f"{{{ARXIV_NS}}}category"):
            cat = c.get("term")
            if cat:
                categories.append(cat)
        published_raw = _clean(entry.findtext(f"{{{ATOM_NS}}}published") or "")
        date_iso = published_raw[:10] if published_raw else None  # ISO 8601: YYYY-MM-DD

        return {
            "doc_id": f"arxiv:{base_id}",
            "title": title,
            "url": canonical_url,
            "date": date_iso,
            "source_type": self.source_type,
            "trust_level": self.trust_level,
            "language": self.language_default,
            "raw_text": summary,
            "companies": [],  # arXiv не даёт компаний; ML заполнит по тексту, если надо
            "extra": {
                "authors": authors,
                "arxiv_id": arxiv_id,
                "categories": categories,
            },
        }


#: Реестр подхватывается оркестратором автоматически.
arxiv_connector = ArxivConnector()
registry.register(arxiv_connector)