"""Конфигурация parser-сервиса.

Все значения читаются из окружения с дефолтами под docker-сеть compose и локальный запуск.
Секреты (LLM/ключи) — только через переменные окружения (CLAUDE.md §5); в коде их нет.
Таймауты и ретраи задаются здесь, чтобы коннекторы не дёргали их из env каждый раз.
"""

from __future__ import annotations

import os


class Settings:
    """Плоские настройки коннекторов и Redis-кэша."""

    service_name: str = "parser"
    version: str = "0.2.0"

    # Redis для кэша результатов (по запросу и по URL). Пустая строка → кэш выключен.
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

    # Общие дефолты HTTP для всех коннекторов.
    http_timeout: float = float(os.getenv("HTTP_TIMEOUT", "60"))
    http_retries: int = int(os.getenv("HTTP_RETRIES", "3"))
    http_retry_backoff: float = float(os.getenv("HTTP_RETRY_BACKOFF", "1.0"))

    # Лимиты выборки: сколько максимум результатов на 1 коннектор.
    default_limit: int = int(os.getenv("DEFAULT_LIMIT", "30"))
    max_per_connector: int = int(os.getenv("MAX_PER_CONNECTOR", "40"))

    # arXiv
    arxiv_api_url: str = os.getenv("ARXIV_API_URL", "https://export.arxiv.org/api/query")
    arxiv_timeout: float = float(os.getenv("ARXIV_TIMEOUT", "30"))

    # OpenAlex (без ключа основной пул работает; ключ повышает лимиты)
    openalex_api_url: str = os.getenv("OPENALEX_API_URL", "https://api.openalex.org/works")
    openalex_email: str = os.getenv("OPENALEX_EMAIL", "")

    # Crossref
    crossref_api_url: str = os.getenv("CROSSREF_API_URL", "https://api.crossref.org/works")
    crossref_mailto: str = os.getenv("CROSSREF_MAILTO", "")

    # GDELT DOC API v2
    gdelt_doc_url: str = os.getenv(
        "GDELT_DOC_URL", "https://api.gdeltproject.org/api/v2/doc/doc"
    )

    # PatentsView API v3
    patentsview_api_url: str = os.getenv(
        "PATENTSVIEW_API_URL", "https://api.patentsview.org/patents/query"
    )

    # User-Agent, который отправляем источникам (arXiv/OpenAlex просят идентифицироваться)
    user_agent: str = os.getenv(
        "PARSER_USER_AGENT",
        "Weak0x00signals-parser/0.2 (hackathon; contact: team@example.com)",
    )

    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()