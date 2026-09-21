"""Parser/RAG-сервис: live search по открытым источникам + grounding-only enrichment."""
from __future__ import annotations

import hashlib
import logging
import os
import re
from collections import Counter
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException

from parser.app.connectors import AREAS, RawHit, collect_all, fetch_article_text
from parser.app.database import get_document, list_sources, persist_documents, semantic_scores
from parser.app.dataset import load_docs
from parser.app.llm import ALLOWED_MODELS, ModelNotAllowed, enrich_document, selected_model
from shared.contracts import CollectRequest, CollectResponse, HealthResponse, InsightRequest, InsightResponse
from shared.schema import SignalDoc, Source, SourceType, TrustLevel

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("parser")

app = FastAPI(title="Слабые сигналы — RAG/источники", version="1.0.0")

STAGE_PATTERNS = (
    (4, ("deployed", "deployment", "production", "commercial", "commercially", "массово", "внедр", "промышленн")),
    (3, ("pilot", "pilots", "trial", "field test", "пилот", "испытан")),
    (2, ("prototype", "prototyp", "proof of concept", "poc", "демонстрац", "прототип")),
    (1, ("concept", "research", "laboratory", "lab", "исследован", "концепц", "теоретическ")),
)
TREND_WORDS = ("growing", "rising", "increase", "emerging", "accelerat", "рост", "раст", "увелич", "emerg")
HYPE_WORDS = ("hype", "overhyped", "marketing buzz", "перегрет", "инфошум", "хайп")


def _area(text: str, requested: Optional[str]) -> Optional[str]:
    if requested:
        return requested
    low = text.lower()
    mapping = {
        "Роботы": ("robot", "drone", "манипулятор", "робот", "дрон"),
        "Финтех": ("payment", "finance", "bank", "credit", "платеж", "финтех", "кредит"),
        "Защита ИИ": ("security", "privacy", "poison", "watermark", "защит", "шифрован", "attest"),
        "Edge": ("edge", "microcontroller", "embedded", "on-device", "микроконтрол", "устройств"),
        "Инфраструктура ИИ": ("chiplet", "memory", "accelerator", "datacenter", "cxl", "ускорител", "памят"),
        "Индустриальный ИИ": ("industrial", "manufactur", "predictive maintenance", "промышлен", "станк", "производств"),
    }
    return next((name for name, words in mapping.items() if any(w in low for w in words)), None)


def _stage(text: str) -> int:
    low = text.lower()
    for score, words in STAGE_PATTERNS:
        if any(w in low for w in words):
            return score
    return 1


def _source_trust(st: SourceType) -> TrustLevel:
    if st in {SourceType.BLOG, SourceType.SOCIAL, SourceType.PRESS}:
        return TrustLevel.LOW
    if st in {SourceType.SCIENTIFIC, SourceType.PATENT, SourceType.REGISTRY}:
        return TrustLevel.HIGH
    return TrustLevel.MEDIUM


def _fingerprint(text: str) -> str:
    words = sorted(set(re.findall(r"[a-zа-яё0-9]{4,}", text.lower())))
    return " ".join(words[:18])


def _similar(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _group_hits(hits: List[RawHit], query: str, area: str | None, limit: int) -> List[SignalDoc]:
    groups: List[List[RawHit]] = []
    for hit in hits:
        hit_text = hit.title + " " + hit.text
        if area and area != _area(hit_text, None):
            continue
        placed = False
        fp = _fingerprint(hit.title)
        for group in groups:
            if _similar(fp, _fingerprint(group[0].title)) >= 0.35:
                group.append(hit)
                placed = True
                break
        if not placed:
            groups.append([hit])

    docs: List[SignalDoc] = []
    today = date.today()
    for idx, group in enumerate(groups[:limit]):
        primary = max(group, key=lambda x: x.date or "")
        all_text = " ".join(_clean_hit_text(x) for x in group if x.text)
        dates = [date.fromisoformat(x.date) for x in group if x.date and re.match(r"^\d{4}-\d{2}-\d{2}$", x.date)]
        recent = sum(1 for d in dates if d >= today - timedelta(days=180))
        older = sum(1 for d in dates if d < today - timedelta(days=180))
        ratio = recent / max(1, recent + older)
        trend = 3 if len(group) >= 5 or (recent >= 3 and ratio >= 0.6) else 2 if len(group) >= 2 or recent >= 2 else 1
        stage = _stage(primary.title + " " + all_text)
        srcs: Dict[str, Source] = {}
        companies = []
        for hit in group:
            host = re.sub(r"^www\.", "", (hit.url.split("/")[2] if "://" in hit.url else "").lower())
            source_title = hit.title
            translated = hit.language.lower() != "ru"
            srcs[hit.url.rstrip("/").lower()] = Source(
                title=source_title,
                url=hit.url,
                date=hit.date,
                source_type=hit.source_type,
                language=hit.language or "en",
                trust_level=_source_trust(hit.source_type),
                translated=translated,
            )
            companies.extend(hit.companies or [])
        companies = list(dict.fromkeys([c for c in companies if c]))[:8]
        raw = all_text[:9000]
        docs.append(SignalDoc(
            id="live-" + hashlib.sha256((primary.url + primary.title).encode()).hexdigest()[:16],
            title=primary.title,
            area=_area(primary.title + " " + raw, area),
            companies=companies,
            raw_text=raw,
            stage=stage,
            trend=trend,
            dataset_score=stage + trend,
            sources=list(srcs.values())[:8],
            retrieval_score=round(min(1.0, 0.25 + 0.12 * len(group) + 0.18 * ratio), 3),
        ))
    docs.sort(key=lambda d: (d.retrieval_score or 0, d.trend or 0, -(d.stage or 1)), reverse=True)
    return docs[:limit]


def _clean_hit_text(hit: RawHit) -> str:
    return (hit.title + ". " + hit.text).strip()


@app.on_event("startup")
def startup():
    try:
        logger.info("LLM модель: %s", selected_model())
    except ModelNotAllowed as exc:
        logger.error("%s", exc)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(service="parser", version="1.0.0", mock=False, details={"mode": "live-rag", "areas": AREAS})


@app.get("/models")
def models() -> dict:
    try:
        current = selected_model()
        error = None
    except ModelNotAllowed as exc:
        current, error = None, str(exc)
    return {"allowed": list(ALLOWED_MODELS), "selected": current, "error": error}


@app.post("/collect", response_model=CollectResponse)
async def collect(req: CollectRequest) -> CollectResponse:
    hits, status = await collect_all(req.query, limit_per_source=max(3, min(10, req.limit // 3)))
    docs = _group_hits(hits, req.query, req.area, req.limit)

    # Если live API временно недоступны, используем только локальный официальный датасет, если он присутствует.
    retrieval_mode = "live"
    if not docs:
        dataset_docs = load_docs(min(req.limit, 100))
        if dataset_docs:
            retrieval_mode = "official_dataset_fallback"
            docs = dataset_docs
    if os.getenv("DEV_MOCK_FALLBACK", "false").lower() == "true" and not docs:
        from parser.app.mock_data import build_documents
        retrieval_mode = "development_mock"
        docs = build_documents(req.query, req.area, req.limit)

    persist_documents(docs)
    semantic = semantic_scores(req.query, [d.id for d in docs])
    if semantic:
        docs = [d.model_copy(update={"retrieval_score": round(max(0.0, min(1.0, semantic.get(d.id, d.retrieval_score or 0.0))), 3)}) for d in docs]
        docs.sort(key=lambda d: d.retrieval_score or 0.0, reverse=True)
    logger.info("collect: query=%r candidates=%d sources=%d mode=%s semantic=%s", req.query, len(docs), sum(len(d.sources) for d in docs), retrieval_mode, bool(semantic))
    return CollectResponse(query=req.query, sources_processed=sum(len(d.sources) for d in docs), documents=docs, connector_status=status, retrieval_mode=retrieval_mode)


@app.get("/document/{document_id}", response_model=SignalDoc)
def document(document_id: str) -> SignalDoc:
    data = get_document(document_id)
    if data:
        return SignalDoc.model_validate(data)
    docs = load_docs(100)
    found = next((d for d in docs if d.id == document_id), None)
    if found:
        return found
    raise HTTPException(status_code=404, detail="Документ не найден")


@app.post("/enrich", response_model=InsightResponse)
async def enrich(req: InsightRequest) -> InsightResponse:
    enriched = []
    model = None
    modes = []
    for doc in req.documents:
        e, model, mode = await enrich_document(doc, req.query)
        enriched.append(e)
        modes.append(mode)
    final_mode = "llm" if modes and all(m == "llm" for m in modes) else "grounded-template"
    persist_documents(enriched)
    return InsightResponse(documents=enriched, model=model, mode=final_mode)


@app.get("/sources")
def sources() -> dict:
    items = list_sources(200)
    return {"total": len(items), "items": items}
