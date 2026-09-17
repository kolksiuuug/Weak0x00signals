"""Интерфейс LLM-сервиса (Участник 3). В скелете — заглушки без сетевых вызовов.

CLAUDE.md §2.4: модель берётся ТОЛЬКО из белого списка ТЗ, автовыбор и роутеры запрещены,
факт выбора модели обязан логироваться. Здесь реализован именно этот каркас:
имя модели читается из переменной окружения, валидируется по белому списку и пишется в лог.
"""

from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger("parser.llm")

#: Белый список моделей из ТЗ (CLAUDE.md §2.4). Ничего вне этого списка использовать нельзя.
ALLOWED_MODELS = (
    "GigaChat 2 Lite",
    "GigaChat 2 Pro",
    "GigaChat 2 Max",
    "YandexGPT Lite 5",
    "YandexGPT Pro 5",
    "YandexGPT Pro 5.1",
    "Qwen3.6 35B-A3B",
    "Qwen3 235B",
    "gpt-4.1",
    "gpt-5.6-luna",
)

#: Пометка, обязательная для автоперевода и генеративных резюме (CLAUDE.md §2.9).
TRANSLATION_NOTE = "автоперевод/генеративное резюме"


class ModelNotAllowed(ValueError):
    """Попытка использовать модель вне белого списка ТЗ."""


def selected_model() -> str:
    """Явно выбранная модель. Никакого автовыбора — только значение из окружения."""
    name = os.getenv("LLM_MODEL", "GigaChat 2 Lite").strip()
    if name not in ALLOWED_MODELS:
        raise ModelNotAllowed(
            "Модель «{}» отсутствует в белом списке ТЗ. Разрешены: {}".format(
                name, ", ".join(ALLOWED_MODELS)
            )
        )
    logger.info("Выбрана LLM: %s (источник выбора: переменная окружения LLM_MODEL)", name)
    return name


def summarize_ru(text: str, sources: List[str]) -> str:
    """МОК: резюме на русском строго по переданным источникам (CLAUDE.md §2.3).

    Реальная реализация обязана передавать в промпт только текст найденных источников
    и отказываться отвечать, если источников нет.
    """
    if not sources:
        return "Недостаточно проверенных источников для формирования резюме."
    model = selected_model()
    logger.info("summarize_ru: модель=%s, источников=%d", model, len(sources))
    return "[МОК-резюме, модель {}] {}".format(model, text[:280])


def translate_ru(text: str, language: str) -> str:
    """МОК: перевод на русский с обязательной пометкой (CLAUDE.md §2.9)."""
    if language == "ru":
        return text
    model = selected_model()
    logger.info("translate_ru: модель=%s, язык оригинала=%s", model, language)
    return "[{}] {}".format(TRANSLATION_NOTE, text)


def generate_hypothesis(title: str, evidence: List[str]) -> str:
    """МОК: гипотеза слабого сигнала строго по найденным свидетельствам."""
    if not evidence:
        return "Гипотеза не формируется: нет подтверждённых источников."
    model = selected_model()
    logger.info("generate_hypothesis: модель=%s, свидетельств=%d", model, len(evidence))
    return (
        "[МОК-гипотеза, модель {}] Технология «{}» может дать преимущество раннего входа: "
        "публикаций мало, игроков немного, зрелого рынка ещё нет.".format(model, title)
    )
