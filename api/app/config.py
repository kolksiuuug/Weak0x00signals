"""Конфигурация api-сервиса. Все значения — из окружения, секретов в коде нет (CLAUDE.md §5)."""

from __future__ import annotations

import os


class Settings:
    """Плоские настройки. Значения по умолчанию рассчитаны на docker-сеть compose."""

    service_name: str = "api"
    version: str = "0.1.0"

    # Внутренние сервисы: обращение по имени контейнера, порты наружу не публикуются.
    parser_url: str = os.getenv("PARSER_URL", "http://parser:8000")
    ml_url: str = os.getenv("ML_URL", "http://ml:8000")

    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    database_url: str = os.getenv("DATABASE_URL", "")

    http_timeout: float = float(os.getenv("HTTP_TIMEOUT", "30"))
    job_ttl_seconds: int = int(os.getenv("JOB_TTL_SECONDS", "3600"))

    # Порог «уверенного» сигнала для статистики на экране (ТЗ: доля сигналов с уверенностью > 75 %).
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
