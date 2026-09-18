"""Базовый интерфейс коннекторов и registry.

Каждый коннектор:
  * объявляет `name`, тип источника и уровень доверенности из §3 контракта;
  * реализует `_search(query, limit)` — сетевой вызов без логики ретраев;
  * наследник получает ретраи/таймауты `search()` через tenacity бесплатно.

Деградация (CLAUDE.md §6 У3): если один источник упал — пайплайн продолжает.
Здесь это обеспечивается тем, что `search()` кидает ConnectorError, а оркестратор
отлавливает его на каждый коннектор отдельно и продолжает с остальными.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from parser.app.config import settings
from shared.schema import SourceType, TrustLevel

logger = logging.getLogger("parser.connectors")


class ConnectorError(RuntimeError):
    """Исчерпывающий сбой коннектора после всех ретраев (нельзя продолжить этот источник)."""


def _is_retryable(exc: BaseException) -> bool:
    """Только временные ошибки: сеть, таймаут, 5xx, 429. 4xx — нет смысла повторять."""
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (408, 429) or exc.response.status_code >= 500
    return False


#: Предикат для tenacity: повторять только retryable-ошибки.
retryable = retry_if_exception(_is_retryable)


class BaseConnector:
    """Каркас коннектора: таймауты, ретраи, идентификация, деградация."""

    name: str = "base"
    source_type: SourceType = SourceType.NEWS
    trust_level: TrustLevel = TrustLevel.MEDIUM
    language_default: str = "en"

    #: жёсткие дефолты из settings; коннектор может переопределить.
    timeout: float = settings.http_timeout
    retries: int = settings.http_retries
    retry_backoff: float = settings.http_retry_backoff

    headers: Dict[str, str] = {}

    def __init__(self) -> None:
        if not self.headers:
            self.headers = {"User-Agent": settings.user_agent}

    async def search(self, query: str, limit: int = settings.default_limit) -> List[Dict[str, Any]]:
        """Публичный метод: с ретраями и таймаутами. Возвращает raw-результаты (dict).

        На выходе — список словарей, которые `normalize` превращает в SignalDoc.
        Любой сбой (после ретраев) превращается в ConnectorError для оркестратора.
        """
        stop = stop_after_attempt(max(1, self.retries))
        wait = wait_exponential(multiplier=self.retry_backoff, max=30.0)
        try:
            for attempt in Retrying(stop=stop, wait=wait, retry=retryable, reraise=True):
                with attempt:
                    return await self._search(query, limit=limit)
        except Exception as exc:  # noqa: BLE001 — все ошибки коннектора ловим здесь
            logger.warning("Коннектор «%s» упал после %d попыток: %s", self.name, self.retries, exc)
            raise ConnectorError(str(exc)) from exc

    async def _search(self, query: str, limit: int) -> List[Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout, headers=self.headers)


class registry:
    """Реестр коннекторов. Добавление нового = зарегистрировать здесь, больше ничего менять."""

    _connectors: List[BaseConnector] = []

    @classmethod
    def register(cls, connector: BaseConnector) -> BaseConnector:
        cls._connectors.append(connector)
        logger.info("Коннектор зарегистрирован: %s", connector.name)
        return connector

    @classmethod
    def all(cls) -> List[BaseConnector]:
        return list(cls._connectors)

    @classmethod
    def names(cls) -> List[str]:
        return [c.name for c in cls._connectors]

    @classmethod
    def by_name(cls, name: str) -> Optional[BaseConnector]:
        for c in cls._connectors:
            if c.name == name:
                return c
        return None


def discover_connectors() -> None:
    """Импорт модулей коннекторов выполняет их регистрацию в registry.

    Модули импортируются лениво снизу пакета, чтобы не было кольцевых импортов:
    __init__ уже определил BaseConnector и registry до этого вызова.
    Если один коннектор не импортируется (битая зависимость) — остальные работают.
    """
    from parser.app.connectors import arxiv  # noqa: F401  (регистрирует сам себя)

    _ = arxiv


__all__ = ["BaseConnector", "ConnectorError", "registry", "discover_connectors", "retryable"]