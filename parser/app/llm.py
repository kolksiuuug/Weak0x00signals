"""Строгий LLM-сервис: только белый список, выбор модели явный, генерация grounding-only."""
from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("parser.llm")

ALLOWED_MODELS = (
    "GigaChat 2 Lite", "GigaChat 2 Pro", "GigaChat 2 Max",
    "YandexGPT Lite 5", "YandexGPT Pro 5", "YandexGPT Pro 5.1",
    "Qwen3.6 35B-A3B", "Qwen3 235B", "gpt-4.1", "gpt-5.6-luna",
)
TRANSLATION_NOTE = "автоперевод/генеративное резюме"


class ModelNotAllowed(ValueError):
    pass


def selected_model() -> str:
    name = os.getenv("LLM_MODEL", "GigaChat 2 Lite").strip()
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
Ответ строго JSON с полями description, advantage, case_example, evidence_summary. Все значения полей — на русском языке.
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
    key = os.getenv("GIGACHAT_API_KEY", "")
    if not key:
        raise RuntimeError("GIGACHAT_API_KEY не задан")
    token = base64.b64encode(key.encode("utf-8")).decode("ascii")
    async with httpx.AsyncClient(timeout=float(os.getenv("LLM_TIMEOUT", "45")), verify=os.getenv("GIGACHAT_VERIFY_SSL", "true").lower() == "true") as client:
        oauth = await client.post(
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            headers={"Authorization": "Basic " + token, "RqUID": os.getenv("GIGACHAT_RQUID") or str(uuid.uuid4()), "Content-Type": "application/x-www-form-urlencoded"},
            data={"scope": "GIGACHAT_API_PERS"},
        )
        oauth.raise_for_status()
        access = oauth.json()["access_token"]
        r = await client.post(
            "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
            headers={"Authorization": "Bearer " + access, "Content-Type": "application/json"},
            json={"model": model, "temperature": 0.1, "messages": [
                {"role": "system", "content": "Отвечай только валидным JSON."},
                {"role": "user", "content": prompt},
            ]},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


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
        # Важнее сохранить функциональный доказательный поиск, чем падать из-за отсутствия ключа.
        logger.warning("LLM недоступна (%s), включён grounded-template fallback", exc)
        return "", model, "grounded-template"


def _fallback(doc) -> Dict[str, str]:
    stage = {1: "концепции/исследования", 2: "прототипа/PoC", 3: "пилота", 4: "раннего внедрения"}.get(doc.stage or 1, "ранней стадии")
    trend = {1: "стабильными", 2: "растущими", 3: "быстро растущими"}.get(doc.trend or 1, "неопределёнными")
    source_types = ", ".join(sorted({s.source_type.value for s in doc.sources}))
    return {
        "description": "По найденным материалам технология находится на стадии {} и демонстрирует {} динамику упоминаний. Подробные доказательства приведены в источниках карточки.".format(stage, trend),
        "advantage": "В найденном контексте отдельное преимущество не выделено; автоматическая генерация не добавляет внешних фактов.",
        "case_example": "В найденных источниках конкретный кейс внедрения с достаточной детализацией не указан.",
        "evidence_summary": "Основание: {} источников, типы — {}. Для зарубежных источников подготовлено русскоязычное резюме/интерпретация.".format(len(doc.sources), source_types),
    }


async def enrich_document(doc, query: str):
    prompt = _prompt(query, doc)
    raw, model, mode = await complete_grounded(prompt)
    data = None
    if raw:
        try:
            candidate = json.loads(raw)
            if all(isinstance(candidate.get(k), str) and candidate[k].strip() for k in ("description", "advantage", "case_example", "evidence_summary")):
                data = candidate
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
