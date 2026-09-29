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
from parser.app.llm import ALLOWED_MODELS, ModelNotAllowed, contextual_rerank, enrich_document, selected_model
from shared.contracts import CollectRequest, CollectResponse, HealthResponse, InsightRequest, InsightResponse
from shared.schema import SignalDoc, Source, SourceType, TrustLevel
from parser.app.context_validator import validate_context

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

_STOPWORDS = {
    "для", "и", "или", "в", "на", "по", "из", "с", "к", "технологии",
    "технология", "технологий", "ии", "ai", "the", "and", "for", "of", "with",
}

def _query_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-zа-яё0-9]{4,}", (text or "").lower())
    return {t[:7] for t in tokens if t not in _STOPWORDS}


_CONTEXT_ALIASES = (
    # Не используем одиночные ``machine``/``learning``: они слишком широкие
    # и пропускают статьи, где ML упомянут вскользь.
    ("ai", "ии", "искусствен", "artificial intelligence", "нейросет", "machine learning", "deep learning", "ml"),
    ("industrial", "промышлен", "manufactur", "factory", "production", "производств", "завод", "industrial ai"),
    ("robot", "робот", "robotics", "automation", "автоматизац"),
    ("security", "кибер", "безопасн", "защит", "cyber", "privacy"),
    ("quantum", "квант"),
    ("data", "данн", "аналит", "dataset", "analytics"),
    ("energy", "энерг", "power", "электро"),
    ("medical", "медицин", "health", "clinical", "клинич"),
)


_CONTEXT_STOPWORDS = {"для", "и", "или", "в", "на", "по", "из", "с", "до", "как", "the", "of", "for", "and", "in", "to", "with", "a", "an"}

_CARD_STOPWORDS = {
    "study", "studies", "paper", "method", "methods", "approach", "system",
    "using", "based", "toward", "towards", "new", "novel", "research",
    "исследование", "исследования", "метод", "методы", "подход", "система",
    "новый", "новая", "новые", "технология", "технологии", "разработка",
}

def _topic_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-zа-яё0-9]{4,}", (text or "").lower().replace("ё", "е"))
    return {t[:12] for t in tokens if t not in _CARD_STOPWORDS and t not in _STOPWORDS}

def _topic_similarity(a: RawHit, b: RawHit) -> float:
    """Сходство двух находок по смысловым словам заголовка + начала текста.

    Нужен именно cross-source clustering: arXiv/OpenAlex/Crossref часто описывают
    одну тему разными заголовками. Мы не склеиваем всё по одному слову запроса,
    а требуем заметное пересечение содержательных терминов.
    """
    aa = _topic_tokens((a.title or "") + " " + (a.text or "")[:1800])
    bb = _topic_tokens((b.title or "") + " " + (b.text or "")[:1800])
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / max(1, len(aa | bb))

def _canonical_title(text: str) -> str:
    return " ".join(sorted(_topic_tokens(text)))

def _relevance_reason(query: str, group: List[RawHit], area: Optional[str]) -> str:
    body = " ".join((h.title or "") + " " + (h.text or "")[:1200] for h in group)
    q = _context_tokens(query)
    b = _context_tokens(body)
    overlap = [token for token in q if any(token.startswith(x) or x.startswith(token) for x in b)]
    query_low = query.lower().replace("ё", "е")
    anchors = []
    for aliases in _CONTEXT_ALIASES:
        if any(alias in query_low for alias in aliases) and any(alias in body.lower().replace("ё", "е") for alias in aliases):
            anchors.append(aliases[0])
    parts = []
    if anchors:
        parts.append("тематические якоря: " + ", ".join(dict.fromkeys(anchors[:4])))
    elif overlap:
        parts.append("совпадающие термины: " + ", ".join(overlap[:5]))
    if area:
        parts.append("область: " + area)
    return "Релевантен исходному запросу: " + ("; ".join(parts) if parts else "содержательная связь подтверждена найденными материалами") + "."


def _context_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-zа-яё0-9]{2,}", (text or "").lower().replace("ё", "е"))
    return {token for token in tokens if token not in _CONTEXT_STOPWORDS}


def _context_gate(query: str, documents):
    """Мягкий тематический gate: режет явный оффтоп, но допускает близкую
    формулировку темы в заголовке/аннотации. Для составного запроса нужен
    технологический якорь и достаточное количество прямых/семантических
    совпадений, а не буквальное совпадение каждого слова.
    """
    q_tokens = _context_tokens(query)
    if not q_tokens:
        return list(documents), []

    def has_alias(body: str, alias: str) -> bool:
        if " " in alias:
            return alias in body
        return any(tok.startswith(alias) or alias.startswith(tok) for tok in _context_tokens(body))

    query_groups = []
    for group in _CONTEXT_ALIASES:
        if any(has_alias(query.lower().replace("ё", "е"), alias) for alias in group):
            query_groups.append(group)

    accepted, rejected = [], []
    for doc in documents:
        body_text = " ".join([
            doc.title or "",
            doc.raw_text or "",
            " ".join(doc.companies or []),
        ]).lower().replace("ё", "е")
        body_tokens = _context_tokens(body_text)

        group_hits = 0
        for group in query_groups:
            if any(has_alias(body_text, alias) for alias in group):
                group_hits += 1

        direct_hits = sum(
            1 for token in q_tokens
            if any(token.startswith(h) or h.startswith(token) for h in body_tokens)
        )

        if not query_groups:
            valid = direct_hits >= (1 if len(q_tokens) <= 3 else 2)
        elif len(query_groups) == 1:
            # Один явный тематический якорь уже достаточен: поисковые источники
            # часто формулируют ту же тему через синонимы, а не словами запроса.
            valid = group_hits >= 1
        else:
            # Для двух явных тем сохраняем оба смысловых якоря — это основная
            # защита от оффтопа. Для 3+ тем допускаем один вторичный якорь.
            required_groups = len(query_groups) if len(query_groups) <= 2 else len(query_groups) - 1
            valid = group_hits >= required_groups

        if valid:
            accepted.append(doc)
        else:
            rejected.append(doc.model_copy(update={
                "is_weak_signal": False,
                "rejected_reason": "Отклонено контекстным фильтром: документ недостаточно связан с тематикой исходного запроса.",
            }))
            logger.info(
                "context gate reject: title=%r groups=%d/%d direct=%d",
                doc.title[:120], group_hits, len(query_groups), direct_hits,
            )

    return accepted, rejected

def _query_relevance(query: str, doc: SignalDoc) -> float:
    q = _query_tokens(query)
    if not q:
        return 0.0
    body = _query_tokens(" ".join([doc.title, doc.area or "", doc.raw_text or "", doc.why or ""]))
    overlap = len(q & body) / max(1, len(q))
    area_bonus = 0.35 if doc.area and _area(query, None) == doc.area else 0.0
    return min(1.0, overlap + area_bonus)

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
    """Оценивает стадию по совокупности контекста, а не по первому совпадению.

    В live-поиске один абзац может одновременно содержать слова «research» и
    «deployed». Берём наиболее подтверждённую стадию: число совпадений + небольшой
    вес более конкретным формулировкам. Это уменьшает схлопывание live-документов
    в одну и ту же порядковую фичу.
    """
    low = text.lower()
    evidence = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    for score, words in STAGE_PATTERNS:
        hits = sum(1 for w in words if w in low)
        if hits:
            evidence[score] += float(hits)
    if not any(evidence.values()):
        return 1
    return max(evidence, key=lambda stage: (evidence[stage], stage))


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


def _split_sentences(text: str) -> List[str]:
    """Простой extractive splitter для grounded-карточки без LLM."""
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip(" -") for p in parts if len(p.strip()) >= 45]


def _extract_grounded_description(doc_title: str, raw_text: str, source_count: int) -> str:
    """Без LLM не показываем англоязычный abstract. Заголовок намеренно сохраняется как есть.

    Полный перевод содержимого выполняется при открытии карточки через выбранную LLM.
    Этот fallback нужен только чтобы в UI не появлялся исходный английский текст.
    """
    return (
        "Подробное описание будет переведено на русский по содержанию найденных источников "
        "без изменения исходного заголовка. Карточка собрана по {} связанным источникам."
    ).format(source_count)


def _extract_advantage(raw_text: str) -> str:
    text = " ".join(_split_sentences(raw_text))
    patterns = (
        (r"\b(?:improv|increase|enhanc|boost|higher accuracy|better)\w*\b|повыш\w+|улучш\w+|увелич\w+", "повышение/улучшение показателя"),
        (r"\b(?:reduce|decrease|lower|save|saving)\w*\b|сниж\w+|сокращ\w+|эконом\w+", "снижение затрат или трудозатрат"),
        (r"\b(?:automate|automation)\w*\b|автоматизац\w+", "автоматизация операций"),
        (r"\b(?:detect|detection|classification)\w*\b|обнаружен\w+|классификац\w+|контрол\w+", "автоматизированное обнаружение/контроль"),
    )
    found = []
    for pattern, label in patterns:
        if re.search(pattern, text, flags=re.I) and label not in found:
            found.append(label)
    if found:
        return "В доступном тексте явно прослеживается: {}. Это описание извлечено из найденных материалов без добавления внешних прогнозов.".format(", ".join(found[:3]))
    return "В доступном тексте отдельное преимущество не выделено однозначно; система не добавляет внешние утверждения."


def _extract_case(raw_text: str) -> str:
    sentences = _split_sentences(raw_text)
    markers = (
        "pilot", "piloted", "deployed", "deployment", "production", "implemented",
        "field test", "real-world", "пилот", "внедр", "испытан", "в производстве", "в эксплуатации",
    )
    for sentence in sentences:
        low = sentence.lower()
        if any(marker in low for marker in markers):
            return "Кейс/испытание, указанное в источнике: {}".format(sentence[:900])
    return "В найденных источниках конкретный кейс внедрения не указан; доступный материал описывает исследование или метод."


def _russian_card_fallback(doc_title: str, stage: int, trend: int, source_count: int, raw_text: str = "") -> Dict[str, str]:
    stage_name = {1: "исследования", 2: "прототипа / PoC", 3: "пилота", 4: "раннего внедрения"}.get(stage, "исследования")
    trend_name = {1: "стабильной", 2: "растущей", 3: "быстро растущей"}.get(trend, "неопределённой")
    weakness = (
        "Сигнал считается слабым, потому что доступные материалы указывают на стадию {} и {} динамику, "
        "а подтверждающая база пока ограничена {} связанными источниками. Это означает раннюю/неустоявшуюся стадию, а не доказанную зрелость технологии."
    ).format(stage_name, trend_name, source_count)
    return {
        "why": weakness,
        "description": _extract_grounded_description(doc_title, raw_text, source_count),
        "advantage": _extract_advantage(raw_text),
        "case_example": _extract_case(raw_text),
        "evidence_summary": "Карточка сформирована по {} связанным материалам открытого научного поиска; содержание и выводы ограничены доступными заголовками и аннотациями источников.".format(source_count),
    }


def _group_hits(hits: List[RawHit], query: str, area: str | None, limit: int) -> List[SignalDoc]:
    groups: List[List[RawHit]] = []
    for hit in hits:
        hit_text = hit.title + " " + hit.text
        if area and area != _area(hit_text, None):
            continue
        placed = False
        fp = _fingerprint(hit.title)
        # Сначала объединяем дубли одного материала, затем — близкие темы из
        # разных коннекторов. Это позволяет карточке иметь несколько источников.
        for group in groups:
            first = group[0]
            same_title = _similar(fp, _fingerprint(first.title)) >= 0.35
            topic_a = _topic_tokens(hit.title + " " + hit.text[:900])
            topic_b = _topic_tokens(first.title + " " + first.text[:900])
            shared_topic_terms = topic_a & topic_b
            same_topic = _topic_similarity(hit, first) >= 0.10 and len(shared_topic_terms) >= 2
            same_query_anchor = bool(_context_tokens(hit_text) & _context_tokens(first.title + " " + first.text[:900]))
            if same_title or (same_topic and same_query_anchor):
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
            **_russian_card_fallback(primary.title, stage, trend, len(srcs), raw),
            relevance_reason=_relevance_reason(query, group, _area(primary.title + " " + raw, area)),
            source_agreement=(
                "Тема подтверждена {} найденными материалами из {} источников поиска.".format(
                    len(group), len(srcs)
                )
                if len(group) > 1
                else "Для темы найден 1 материал; независимое подтверждение из нескольких источников не получено."
            ),
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
    target = max(10, min(15, req.limit))
    candidate_pool = max(req.limit * 5, 120)
    # 3 live-коннектора × до 50 результатов дают широкий резерв для context gate + ML.
    hits, status = await collect_all(req.query, limit_per_source=50)
    docs = _group_hits(hits, req.query, req.area, candidate_pool)
    retrieval_mode = "live"

    # Сначала отсекаем предметно чужие документы. Это происходит ДО ML scoring,
    # поэтому «пробиотик для кур» не сможет попасть в TOP только из-за случайного
    # совпадения слов/области.
    docs, context_rejected = _context_gate(req.query, docs)

    # Если строгий gate оставил слишком мало кандидатов, добираем только из его
    # собственных отклонённых документов. Это не отменяет фильтр: допускаются
    # лишь документы с явным тематическим якорем и дополнительным прямым
    # совпадением. Так сохраняется защита от очевидного оффтопа, но выдача не
    # схлопывается до 5–8 карточек на широких научных запросах.
    if len(docs) < target and context_rejected:
        q_low = req.query.lower().replace("ё", "е")
        soft = []
        for rejected_doc in context_rejected:
            body = " ".join([rejected_doc.title or "", rejected_doc.raw_text or "", " ".join(rejected_doc.companies or [])]).lower().replace("ё", "е")
            q_words = _context_tokens(q_low)
            direct = sum(1 for token in q_words if any(token.startswith(h) or h.startswith(token) for h in _context_tokens(body)))
            has_ai = any(alias in body for alias in ("artificial intelligence", "искусственный интеллект", "нейросет", "machine learning", "deep learning", "ai"))
            has_industrial = any(alias in body for alias in ("industrial", "manufactur", "factory", "промышлен", "производств", "завод"))
            query_ai = any(alias in q_low for alias in (" ии", "ai", "искусствен", "artificial intelligence", "нейросет", "machine learning"))
            query_industrial = any(alias in q_low for alias in ("промышлен", "industrial", "manufactur", "factory", "производств", "завод"))
            if direct >= 2 or (query_ai and has_ai) or (query_industrial and has_industrial and direct >= 2):
                soft.append(rejected_doc)
        docs.extend(soft[:max(0, target - len(docs))])
        if soft:
            context_rejected = context_rejected[len(soft):]
            logger.info("context gate soft refill: added=%d target=%d", min(len(soft), max(0, target-len(docs)+len(soft))), target)

    if os.getenv("DEV_MOCK_FALLBACK", "false").lower() == "true" and not docs:
        from parser.app.mock_data import build_documents
        retrieval_mode = "development_mock"
        docs = build_documents(req.query, req.area, req.limit)

    persist_documents(docs)

    # 1. Быстрый semantic pre-rank. Контекстный gate выше уже является обязательной
    # тематической проверкой; LLM rerank/validator здесь намеренно НЕ вызываются.
    # для более дорогой контекстной оценки LLM.
    semantic = semantic_scores(req.query, [d.id for d in docs])

    if semantic:
        docs = [
            d.model_copy(
                update={
                    "retrieval_score": round(
                        max(
                            0.0,
                            min(1.0, semantic.get(d.id, d.retrieval_score or 0.0)),
                        ),
                        3,
                    )
                }
            )
            for d in docs
        ]
        docs.sort(key=lambda d: d.retrieval_score or 0.0, reverse=True)

        logger.info(
            "semantic pre-rank: candidates=%d top=%s",
            len(docs),
            [
                {
                    "score": d.retrieval_score,
                    "title": d.title[:100],
                }
                for d in docs[:25]
            ],
        )

        logger.info(
            "contextual LLM rerank disabled: deterministic context gate already applied; candidates=%d",
            len(docs),
        )

    logger.info(
        "collect: query=%r candidates=%d sources=%d target=%d mode=%s semantic=%s",
        req.query,
        len(docs),
        sum(len(d.sources) for d in docs),
        target,
        retrieval_mode,
        bool(semantic),
    )
    return CollectResponse(query=req.query, sources_processed=sum(len(d.sources) for d in docs), documents=docs, connector_status=status, retrieval_mode=retrieval_mode)


@app.get("/document/{document_id}", response_model=SignalDoc)
def document(document_id: str) -> SignalDoc:
    data = get_document(document_id)
    if data:
        return SignalDoc.model_validate(data)
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
