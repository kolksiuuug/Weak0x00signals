"""Строгий LLM-сервис: только белый список, выбор модели явный, генерация grounding-only."""
from __future__ import annotations

import base64
import json
import logging
import os
import uuid
import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("parser.llm")

ALLOWED_MODELS = (
    "GigaChat 2 Lite", "GigaChat 2 Pro", "GigaChat 2 Max",
    "YandexGPT Lite 5", "YandexGPT Pro 5", "YandexGPT Pro 5.1",
    "Qwen3.6 35B-A3B", "Qwen3 235B", "gpt-4.1", "gpt-5.6-luna",
)
GIGACHAT_MODEL_IDS = {
    "GigaChat 2 Lite": "GigaChat-2",
    "GigaChat 2 Pro": "GigaChat-2-Pro",
    "GigaChat 2 Max": "GigaChat-2-Max",
}
TRANSLATION_NOTE = "автоперевод/генеративное резюме"

# Один процесс parser = один LLM-клиент. Не создаём OAuth-сессию и chat request
# заново для каждого документа. После 429 включается cooldown, чтобы не долбить
# провайдера повторными запросами в рамках того же прогона.
_GIGACHAT_TOKEN: Optional[str] = None
_GIGACHAT_TOKEN_EXPIRES_AT = 0.0
_GIGACHAT_LOCK = asyncio.Lock()
_LLM_COOLDOWN_UNTIL = 0.0
_LLM_COOLDOWN_LOCK = asyncio.Lock()


async def _llm_in_cooldown() -> bool:
    return time.monotonic() < _LLM_COOLDOWN_UNTIL


async def _set_llm_cooldown(seconds: float = 45.0) -> None:
    global _LLM_COOLDOWN_UNTIL
    async with _LLM_COOLDOWN_LOCK:
        _LLM_COOLDOWN_UNTIL = max(_LLM_COOLDOWN_UNTIL, time.monotonic() + seconds)



class ModelNotAllowed(ValueError):
    pass


def selected_model() -> str:
    name = os.getenv("LLM_MODEL", "GigaChat 2 Max").strip()
    if name not in ALLOWED_MODELS:
        raise ModelNotAllowed("Модель «{}» отсутствует в белом списке ТЗ.".format(name))
    logger.info("LLM: явно выбрана модель=%s, источник выбора=LLM_MODEL", name)
    return name


def _prompt(query: str, doc) -> str:
    source_lines = []
    for s in doc.sources:
        source_lines.append("- {} | {} | {} | {} | {} | translated={}".format(
            s.title, s.url, s.date or "дата не указана", s.source_type.value, s.language, s.translated
        ))
    return """
Ты работаешь как модуль доказательного RAG.
Запрещено добавлять факты, компании, цифры, кейсы или преимущества, которых нет в контексте ниже.
Каждое утверждение должно быть выводимо из raw_text и списка источников.
Ответ строго JSON с полями why, description, advantage, case_example, evidence_summary.
КРИТИЧЕСКИ ВАЖНО: значение поля title НЕ нужно возвращать и НЕ нужно переводить — исходный заголовок документа должен остаться на языке источника.
ВСЕ остальные значения JSON ОБЯЗАТЕЛЬНО пиши на русском языке. Если исходник на английском, сначала переведи его смысл на русский, сохраняя названия моделей, стандартов, методов, аббревиатуры и собственные имена без перевода (например YOLO11, CNN, arXiv). Не оставляй английские предложения и не вставляй английские цитаты.
Поле description — это аккуратный перевод/сжатый пересказ доступного содержания источника на русском, без добавления фактов.
Поле case_example: только реальный пример внедрения/исследования, если он явно есть в источниках; иначе напиши "В найденных источниках конкретный кейс не указан.".
Поле advantage: только преимущество, которое явно следует из источников; без прогнозов и обещаний.

Запрос пользователя: {query}
Технология: {title}
Контекст:
{raw}
Источники:
{sources}
""".format(query=query, title=doc.title, raw=doc.raw_text[:9000], sources="\n".join(source_lines))


async def _openai_compatible(prompt: str, model: str) -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY не задан")
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    async with httpx.AsyncClient(timeout=float(os.getenv("LLM_TIMEOUT", "45"))) as client:
        r = await client.post(
            base + "/chat/completions",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            json={"model": model, "temperature": 0.1, "messages": [
                {"role": "system", "content": "Отвечай только валидным JSON без markdown."},
                {"role": "user", "content": prompt},
            ]},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def _yandex(prompt: str, model: str) -> str:
    key = os.getenv("YANDEX_API_KEY", "")
    folder = os.getenv("YANDEX_FOLDER_ID", "")
    if not key or not folder:
        raise RuntimeError("YANDEX_API_KEY/YANDEX_FOLDER_ID не заданы")
    model_uri = "gpt://{}/yandexgpt/latest".format(folder)
    async with httpx.AsyncClient(timeout=float(os.getenv("LLM_TIMEOUT", "45"))) as client:
        r = await client.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers={"Authorization": "Api-Key " + key},
            json={"modelUri": model_uri, "completionOptions": {"stream": False, "temperature": 0.1, "maxTokens": 900},
                  "messages": [{"role": "system", "text": "Отвечай только валидным JSON."}, {"role": "user", "text": prompt}]},
        )
        r.raise_for_status()
        return r.json()["result"]["alternatives"][0]["message"]["text"]

async def _gigachat(prompt: str, model: str) -> str:
    global _GIGACHAT_TOKEN, _GIGACHAT_TOKEN_EXPIRES_AT
    key = os.getenv("GIGACHAT_API_KEY", "")
    if not key:
        raise RuntimeError("GIGACHAT_API_KEY не задан")

    if await _llm_in_cooldown():
        raise RuntimeError("GigaChat cooldown после rate limit")

    api_model = GIGACHAT_MODEL_IDS.get(model, model)
    timeout = float(os.getenv("LLM_TIMEOUT", "45"))

    async with _GIGACHAT_LOCK:
        async with httpx.AsyncClient(
            timeout=timeout,
            verify=os.getenv("GIGACHAT_VERIFY_SSL", "true").lower() == "true",
        ) as client:
            now = time.monotonic()
            if not _GIGACHAT_TOKEN or now >= _GIGACHAT_TOKEN_EXPIRES_AT - 30:
                oauth = await client.post(
                    "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                    headers={
                        "Authorization": "Basic " + key,
                        "RqUID": os.getenv("GIGACHAT_RQUID") or str(uuid.uuid4()),
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                    },
                    data={"scope": "GIGACHAT_API_PERS"},
                )
                if oauth.status_code == 429:
                    await _set_llm_cooldown(60)
                    raise RuntimeError("GigaChat OAuth rate limit (429)")
                oauth.raise_for_status()
                payload = oauth.json()
                _GIGACHAT_TOKEN = payload["access_token"]
                _GIGACHAT_TOKEN_EXPIRES_AT = now + float(payload.get("expires_in", 1800))

            r = await client.post(
                "https://api.giga.chat/v1/chat/completions",
                headers={
                    "Authorization": "Bearer " + _GIGACHAT_TOKEN,
                    "Content-Type": "application/json",
                },
                json={
                    "model": api_model,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": "Отвечай только валидным JSON."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            if r.status_code == 429:
                await _set_llm_cooldown(60)
                raise RuntimeError("GigaChat chat rate limit (429)")
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            logger.info("GigaChat: успешный grounded request, модель=%s", api_model)
            return content


async def complete_grounded(prompt: str) -> Tuple[str, str, str]:
    model = selected_model()
    try:
        if model.startswith("YandexGPT"):
            raw = await _yandex(prompt, model)
        elif model.startswith("GigaChat"):
            raw = await _gigachat(prompt, model)
        else:
            raw = await _openai_compatible(prompt, model)
        return raw, model, "llm"
    except Exception as exc:  # noqa: BLE001
        if os.getenv("LLM_REQUIRED", "false").lower() == "true":
            logger.error("LLM обязательна, но недоступна: %s", exc)
            raise RuntimeError("Обогащение карточки требует доступной LLM: {}".format(exc)) from exc
        # В необязательном режиме сохраняем поиск рабочим, но не показываем
        # пользователю исходный англоязычный текст: карточка останется с русским fallback.
        logger.warning("LLM недоступна (%s), включён русскоязычный grounded-template fallback", exc)
        return "", model, "grounded-template"


async def expand_search_queries(query: str, max_queries: int = 4) -> List[str]:
    """Генерирует дополнительные поисковые формулировки строго на основе запроса пользователя.

    Исходный запрос не изменяется и должен использоваться вызывающим кодом отдельно.
    При любой ошибке возвращается пустой список.
    """
    query = " ".join((query or "").split()).strip()
    if not query:
        return []

    prompt = """Ты модуль расширения поискового запроса для научного поиска arXiv.

Твоя задача — взять ЗАПРОС ПОЛЬЗОВАТЕЛЯ и предложить до 4 альтернативных
поисковых формулировок на АНГЛИЙСКОМ языке.

СТРОГИЕ ПРАВИЛА:
1. Сохраняй исходную предметную область запроса.
2. Не добавляй новую предметную область, которой нет в запросе.
3. Не придумывай компании, технологии, отрасли или конкретные применения.
4. Используй научные термины, синонимы и естественные английские формулировки.
5. Формулировки должны быть пригодны для поиска научных статей.
6. Если запрос уже на английском — только аккуратно расширь его синонимами.
7. Верни ТОЛЬКО JSON-массив строк.
8. Не включай пояснения, markdown или нумерацию.

Запрос пользователя:
{query}
""".format(query=query)

    if os.getenv("LLM_SEARCH_EXPANSION", "false").lower() != "true":
        logger.info("Search expansion disabled: deterministic connector expansion is used")
        return []

    try:
        raw, _, mode = await complete_grounded(prompt)

        if mode != "llm" or not raw:
            return []

        data = json.loads(raw)

        if not isinstance(data, list):
            return []

        result = []
        seen = set()

        for item in data:
            if not isinstance(item, str):
                continue

            item = " ".join(item.split()).strip().strip('"')
            if not item:
                continue

            key = item.lower()
            if key == query.lower() or key in seen:
                continue

            seen.add(key)
            result.append(item)

            if len(result) >= max_queries:
                break

        logger.info(
            "Search expansion: query=%r expansions=%d",
            query,
            len(result),
        )
        return result

    except Exception as exc:
        logger.warning(
            "Search expansion failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        return []


def _source_sentences(doc) -> List[str]:
    """Выделяет короткие фактические предложения из abstract/raw_text для fallback."""
    import re
    text = " ".join([doc.raw_text or "", " ".join(s.title or "" for s in doc.sources)])
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip(" -") for p in parts if len(p.strip()) >= 45][:12]


def _english_fact_to_ru(text: str) -> str:
    """Безопасный extractive fallback: переводит типовые научные конструкции.
    Не пытается «додумывать» отсутствующие факты.
    """
    import re
    replacements = [
        (r"(?i)the aim of (?:the )?study is to", "цель исследования —"),
        (r"(?i)the aim is to", "цель работы —"),
        (r"(?i)this paper presents", "в работе представлены"),
        (r"(?i)this study presents", "в исследовании представлены"),
        (r"(?i)the study presents", "исследование описывает"),
        (r"(?i)the results show that", "результаты показывают, что"),
        (r"(?i)results show that", "результаты показывают, что"),
        (r"(?i)the proposed (?:method|approach|system) (?:can|aims to)", "предлагаемый подход направлен на"),
        (r"(?i)the proposed system", "предлагаемая система"),
        (r"(?i)the proposed method", "предлагаемый метод"),
        (r"(?i)was developed", "был разработан"),
        (r"(?i)were developed", "были разработаны"),
        (r"(?i)was tested", "был протестирован"),
        (r"(?i)were tested", "были протестированы"),
        (r"(?i)pilot testing", "пилотное тестирование"),
        (r"(?i)pilot study", "пилотное исследование"),
        (r"(?i)real-world", "в реальных условиях"),
        (r"(?i)industrial setting", "промышленных условиях"),
        (r"(?i)manufacturing", "промышленном производстве"),
        (r"(?i)artificial intelligence", "искусственного интеллекта"),
        (r"(?i)machine learning", "машинного обучения"),
        (r"(?i)computer vision", "компьютерного зрения"),
        (r"(?i)reduce labor costs", "сократить трудозатраты"),
        (r"(?i)reduce costs", "снизить затраты"),
        (r"(?i)improve (?:detection|classification|prediction) accuracy", "повысить точность обнаружения, классификации или прогнозирования"),
        (r"(?i)improve accuracy", "повысить точность"),
        (r"(?i)optimize production processes", "оптимизировать производственные процессы"),
        (r"(?i)optimize processes", "оптимизировать процессы"),
        (r"(?i)quality control", "контроль качества"),
        (r"(?i)data augmentation", "аугментация данных"),
        (r"(?i)convolutional neural network", "сверточная нейронная сеть"),
        (r"(?i)neural network", "нейронная сеть"),
    ]
    result=text.strip()
    for pattern,repl in replacements:
        result=re.sub(pattern,repl,result)
    # Если предложение всё ещё почти полностью английское, не показываем его
    # как «русский перевод»: используем только фактологический шаблон.
    latin=len(re.findall(r"[A-Za-z]",result))
    cyr=len(re.findall(r"[А-Яа-яЁё]",result))
    if latin > max(12, cyr * 0.8):
        return ""
    return result


def _fallback(doc) -> Dict[str, str]:
    """Содержательный русскоязычный fallback без LLM.

    Он строится только из наблюдаемых признаков raw_text/title и не выдаёт пустые
    заглушки. Английский title при этом не переводится и остаётся отдельным полем.
    """
    import re
    stage = {1: "исследования", 2: "прототипа / PoC", 3: "пилота", 4: "раннего внедрения"}.get(doc.stage or 1, "ранней стадии")
    trend = {1: "стабильной", 2: "растущей", 3: "быстро растущей"}.get(doc.trend or 1, "неопределённой")
    raw = (doc.raw_text or "").lower()
    title = (doc.title or "").lower()

    concepts = []
    concept_map = (
        (("artificial intelligence", "искусственн"), "искусственный интеллект"),
        (("machine learning", "машинн"), "машинное обучение"),
        (("computer vision", "компьютерн"), "компьютерное зрение"),
        (("neural network", "нейронн"), "нейронные сети"),
        (("quality control", "контрол.*качеств"), "контроль качества"),
        (("automation", "автоматизац"), "автоматизация"),
        (("defect detection", "обнаружен.*дефект"), "обнаружение дефектов"),
        (("classification", "классификац"), "классификация объектов"),
        (("prediction", "прогнозирован"), "прогнозирование"),
        (("optimization", "оптимизац"), "оптимизация процессов"),
        (("industrial", "промышленн", "manufactur", "factory", "production"), "промышленное применение"),
        (("robot", "робот"), "роботизированные системы"),
        (("sensor", "сенсор", "датчик"), "сенсорные технологии"),
        (("dataset", "набор.*данн", "data augmentation"), "работа с данными и обучающими выборками"),
    )
    for needles, label in concept_map:
        if any((re.search(n, raw) if any(ch in n for ch in ".*?[]") else n in raw) for n in needles):
            if label not in concepts:
                concepts.append(label)

    if concepts:
        concept_text = ", ".join(concepts[:5])
        description = (
            "Материал посвящён исследованию технологии и её применению. "
            "В источнике рассматриваются: {}. "
            "По содержанию работа относится к стадии {} и содержит фактические сведения о методах, испытаниях или результатах."
        ).format(concept_text, stage)
    else:
        description = (
            "Материал посвящён исследованию технологического подхода. "
            "По доступному тексту работа относится к стадии {} и содержит фактические сведения о методах или результатах исследования."
        ).format(stage)

    advantage_map = (
        (("reduce labor costs", "сокращение трудозатрат"), "сокращение трудозатрат"),
        (("reduce costs", "снижение затрат"), "снижение затрат"),
        (("improve accuracy", "improve detection accuracy", "повышен.*точност"), "повышение точности"),
        (("optimize production", "оптимизац"), "оптимизация производственных процессов"),
        (("quality control", "контрол.*качеств"), "улучшение контроля качества"),
        (("automation", "автоматизац"), "автоматизация операций"),
    )
    advantages = []
    for needles, label in advantage_map:
        if any((re.search(n, raw) if any(ch in n for ch in ".*?[]") else n in raw) for n in needles):
            if label not in advantages:
                advantages.append(label)
    if advantages:
        advantage = "В источнике прослеживаются следующие подтверждённые преимущества: {}. Они основаны на содержании материала, без добавления внешних прогнозов.".format(", ".join(advantages[:4]))
    else:
        advantage = "Источник описывает применяемый подход и его результаты, однако отдельный измеримый эффект в доступном тексте явно не выделен."

    case_markers = ("pilot", "пилот", "deployed", "deployment", "implemented", "внедр", "production", "промышленн", "tested", "тестирован", "field test", "real-world")
    if any(m in raw or m in title for m in case_markers):
        case = "В источнике присутствует описание испытания, пилота или применения в практической среде. Конкретный формат применения определяется по приведённым в источнике условиям исследования."
    else:
        case = "Источник содержит описание исследовательской работы; отдельный практический кейс внедрения в доступном тексте явно не выделен."

    source_types = ", ".join(sorted({s.source_type.value for s in doc.sources})) or "научные публикации"
    evidence = (
        "Основание карточки — {} источник(а): {}. В качестве доказательств используются содержание публикации, её описание методов/результатов и признаки стадии развития технологии."
    ).format(len(doc.sources), source_types)

    why = (doc.why or "").strip()
    if not why:
        why = (
            "Документ прошёл тематический фильтр по запросу и отнесён к сигналам на стадии {} с {} динамикой. "
            "Итоговая оценка учитывает уверенность модели, приоритет стадии/тренда и качество подтверждающих источников."
        ).format(stage, trend)

    return {
        "why": why,
        "description": description,
        "advantage": advantage,
        "case_example": case,
        "evidence_summary": evidence,
    }



async def contextual_rerank(query: str, documents):
    """Контекстное ранжирование. При отказе LLM сохраняет semantic TOP-K."""
    if not documents:
        return []


    items = [
        {
            "id": document.id,
            "title": (document.title or "")[:300],
        }
        for document in documents
    ]

    prompt = """Ты выполняешь задачу сортировки результатов научного поиска.

Исходный поисковый запрос:
{query}

Отсортируй документы по тому, насколько их тема соответствует
исходному поисковому запросу.

Оценивай именно смысловую тематическую близость всей темы документа
к запросу, а не отдельные совпадающие слова.

Не объясняй решение.
Не меняй формулировку запроса.
Не придумывай информацию.
Верни только JSON-массив объектов.

Формат:
[
  {{"id": "ID", "score": 100}},
  {{"id": "ID", "score": 80}}
]

100 = тема непосредственно соответствует запросу.
70 = очень близкая тема.
40 = косвенная связь.
0 = тема практически не соответствует.

Результаты:
{documents}
""".format(
        query=query,
        documents=json.dumps(items, ensure_ascii=False),
    )

    try:
        raw, _, mode = await complete_grounded(prompt)

        if mode != "llm" or not raw:
            logger.warning("Contextual rerank unavailable; keeping semantic candidates")
            return documents

        cleaned = raw.strip()

        if cleaned.startswith("```"):
            if cleaned.startswith("```json"):
                cleaned = cleaned[len("```json"):]
            else:
                cleaned = cleaned[len("```"):]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        left = cleaned.find("[")
        right = cleaned.rfind("]")

        if left < 0 or right <= left:
            logger.warning(
                "Contextual rerank unavailable; keeping semantic candidates raw=%r",
                raw[:500],
            )
            return documents

        data = json.loads(cleaned[left:right + 1])

        if not isinstance(data, list):
            return documents

        valid_ids = {document.id for document in documents}
        scores = {}

        for item in data:
            if not isinstance(item, dict):
                continue

            doc_id = item.get("id")

            if doc_id not in valid_ids:
                continue

            try:
                score = float(item.get("score"))
            except (TypeError, ValueError):
                continue

            scores[doc_id] = max(0.0, min(100.0, score))

        if not scores:
            logger.warning(
                "Contextual rerank unavailable; no valid scores; "
                "keeping semantic candidates"
            )
            return documents

        ranked = sorted(
            documents,
            key=lambda document: scores.get(document.id, -1),
            reverse=True,
        )

        result = [
            document.model_copy(
                update={
                    "retrieval_score": round(
                        scores[document.id] / 100.0,
                        3,
                    )
                }
            )
            for document in ranked
            if document.id in scores
        ]

        logger.info(
            "Contextual rerank: candidates=%d scored=%d top=%s",
            len(documents),
            len(scores),
            [
                {
                    "score": round(scores[document.id], 1),
                    "title": document.title[:100],
                }
                for document in result
            ],
        )

        return result or documents

    except Exception as exc:
        logger.warning(
            "Contextual rerank failed: %s: %s; keeping semantic candidates",
            type(exc).__name__,
            exc,
        )
        return documents

async def enrich_document(doc, query: str):
    prompt = _prompt(query, doc)
    raw, model, mode = await complete_grounded(prompt)
    data = None
    if raw:
        try:
            candidate = json.loads(raw)
            fields = ("why", "description", "advantage", "case_example", "evidence_summary")
            if all(isinstance(candidate.get(k), str) and candidate[k].strip() for k in fields):
                # Не допускаем случайный английский текст в русской карточке.
                # Если модель нарушила языковое требование, сохраняем русский
                # grounded fallback вместо показа исходного англоязычного abstract.
                latin_dominant = sum(1 for k in fields if len(re.findall(r"[A-Za-z]", candidate[k])) > max(8, len(re.findall(r"[А-Яа-яЁё]", candidate[k])) * 1.2))
                if latin_dominant == 0:
                    data = candidate
                else:
                    logger.warning("LLM enrichment returned English-dominant fields for %s; using Russian fallback", doc.id)
        except Exception:
            logger.warning("LLM вернула невалидный JSON для %s", doc.id)
    data = data or _fallback(doc)
    data["description"] = data["description"][:3000]
    data["advantage"] = data["advantage"][:1600]
    data["case_example"] = data["case_example"][:1600]
    data["evidence_summary"] = data["evidence_summary"][:1200]
    return doc.model_copy(update=data), model, mode


def translate_ru(text: str, language: str) -> str:
    # Перевод выполняется в enrich через контекст; эта функция сохраняет старый импортный контракт.
    return text if language == "ru" else "[{}] {}".format(TRANSLATION_NOTE, text)
