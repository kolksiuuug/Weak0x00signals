"""Определение типа и доверенности источника по домену (CLAUDE.md §2.7, §2.8).

Зачем это в /ml, если trust-слой — зона Участника 1. В датасете организаторов колонка
«Источники» содержит только название и ссылку: ни типа, ни языка, ни доверенности.
Чтобы признаки на обучении и на инференсе считались ОДИНАКОВО, нужен детерминированный
способ восстановить метаданные из URL.

Правило приоритета: если SignalDoc уже принёс source_type/trust_level от Участника 1 —
используем их как есть, ничего не переопределяем. Домен разбираем только когда данных нет
(то есть практически только на датасете организаторов).

Соответствие §2.8: соцсети, блоги, агрегаторы, пресс-релизы -> TrustLevel.LOW.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple
from urllib.parse import urlparse

from shared.schema import SourceType, TrustLevel

#: (регулярка по домену, тип источника, доверенность).
#: Порядок важен: первое совпадение выигрывает, поэтому специфичное идёт раньше общего.
DOMAIN_RULES: Tuple[Tuple[str, SourceType, TrustLevel], ...] = (
    # --- патенты -----------------------------------------------------------
    (r"patents\.google\.|patentsview\.|espacenet\.|patentscope|wipo\.int|fips\.ru|rupto\.ru",
     SourceType.PATENT, TrustLevel.HIGH),

    # --- научно-популярные издания при научных обществах --------------------
    # Идут ПЕРЕД научным блоком: spectrum.ieee.org и communications.acm.org — это
    # журналы, а не рецензируемые площадки. Иначе правило ieee\.org ниже выдаёт им
    # высокую доверенность, и хайповый материал получает вес научной публикации.
    # Путь в URL здесь не помогает: правила сопоставляются с хостом.
    (r"spectrum\.ieee\.org|cacm\.acm\.org|communications\.acm\.org|"
     r"technologyreview\.com|newscientist\.com|scientificamerican\.com",
     SourceType.NEWS, TrustLevel.MEDIUM),

    # --- научные публикации ------------------------------------------------
    (r"arxiv\.org|biorxiv\.|medrxiv\.|ssrn\.com|doi\.org|crossref\.org|openalex\.org",
     SourceType.SCIENTIFIC, TrustLevel.HIGH),
    (r"nature\.com|science\.org|sciencedirect\.com|springer\.|link\.springer|wiley\.com|"
     r"ieee\.org|ieeexplore|acm\.org|mdpi\.com|frontiersin\.org|plos\.org|pubmed|ncbi\.nlm\.nih\.gov",
     SourceType.SCIENTIFIC, TrustLevel.HIGH),
    (r"cyberleninka\.ru|elibrary\.ru|mathnet\.ru|naukaru\.ru",
     SourceType.SCIENTIFIC, TrustLevel.HIGH),

    # --- гос. реестры, регуляторы, университеты ----------------------------
    (r"(^|\.)gov(\.|$)|government\.ru|\.gov\.ru|minobrnauki|digital\.gov\.ru|cbr\.ru|"
     r"nalog\.ru|rosstat|eur-lex\.europa\.eu|europa\.eu|nist\.gov|fda\.gov",
     SourceType.REGISTRY, TrustLevel.HIGH),
    (r"(^|\.)edu(\.|$)|\.ac\.uk|\.edu\.|mit\.edu|stanford\.edu|hse\.ru|msu\.ru|spbu\.ru|"
     r"skoltech\.ru|mipt\.ru|itmo\.ru",
     SourceType.SCIENTIFIC, TrustLevel.HIGH),

    # --- пресс-релизы (§2.8: пониженная доверенность) ----------------------
    (r"prnewswire\.|businesswire\.|globenewswire\.|newswire\.|eurekalert\.org|"
     r"press\.|pressroom\.|interfax\.ru/press",
     SourceType.PRESS, TrustLevel.LOW),

    # --- соцсети и агрегаторы (§2.8) ---------------------------------------
    (r"twitter\.com|(^|\.)x\.com|linkedin\.com|reddit\.com|vk\.com|ok\.ru|t\.me|telegram\.|"
     r"youtube\.com|youtu\.be|tiktok\.com|facebook\.com|instagram\.com|news\.ycombinator\.com",
     SourceType.SOCIAL, TrustLevel.LOW),

    # --- блоги и площадки пользовательского контента (§2.8) ----------------
    (r"medium\.com|substack\.com|dzen\.ru|blogspot\.|wordpress\.com|habr\.com|vc\.ru|"
     r"dev\.to|hashnode\.|telegra\.ph|livejournal\.com",
     SourceType.BLOG, TrustLevel.LOW),

    # --- аналитические отчёты ----------------------------------------------
    (r"gartner\.com|idc\.com|forrester\.com|mckinsey\.com|bcg\.com|deloitte\.|pwc\.|kpmg\.|"
     r"statista\.com|cbinsights\.com|crunchbase\.com|pitchbook\.com|iea\.org|weforum\.org|"
     r"tadviser\.ru|json\.tv",
     SourceType.ANALYTICS, TrustLevel.MEDIUM),

    # --- отраслевые и деловые медиа ----------------------------------------
    (r"reuters\.com|bloomberg\.com|ft\.com|wsj\.com|economist\.com|nytimes\.com|"
     r"techcrunch\.com|wired\.com|theverge\.com|arstechnica\.com|"
     r"kommersant\.ru|vedomosti\.ru|rbc\.ru|tass\.ru|ria\.ru|interfax\.ru|forbes\.|"
     r"cnews\.ru|comnews\.ru|iz\.ru",
     SourceType.NEWS, TrustLevel.MEDIUM),
)

_COMPILED = tuple((re.compile(pattern), stype, trust) for pattern, stype, trust in DOMAIN_RULES)

#: Чем закрываем неизвестный домен. Не HIGH: неизвестному источнику доверия не выдаём.
DEFAULT_TYPE = SourceType.NEWS
DEFAULT_TRUST = TrustLevel.MEDIUM

#: Домены-исключения, где даже средняя доверенность не оправдана (агрегаторы ссылок).
AGGREGATOR_RE = re.compile(r"news\.google\.|yandex\.ru/news|flipboard\.|feedly\.|"
                           r"allsides\.|smi2\.|rambler\.ru/news")


#: Кириллица в названии — надёжный признак русскоязычного источника.
_CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)


def guess_language(title: str, url: str) -> str:
    """Язык оригинала (CLAUDE.md §2.7). Грубо, но детерминированно и объяснимо.

    Используется там, где язык не пришёл извне: в датасете организаторов у источника
    есть только название и ссылка. Если Участник 3 язык проставил — он приоритетнее.
    """
    if _CYRILLIC_RE.search(title or ""):
        return "ru"
    host = domain_of(url)
    if host.endswith(".ru") or host.endswith(".рф") or host.endswith(".su"):
        return "ru"
    return "en"


def domain_of(url: str) -> str:
    """Хост из URL в нижнем регистре, без www. Пустая строка, если URL неразборчив."""
    try:
        host = urlparse(str(url)).netloc.lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def classify_url(url: str) -> Tuple[SourceType, TrustLevel]:
    """URL -> (тип источника, уровень доверенности) по таблице доменов."""
    host = domain_of(url)
    if not host:
        return DEFAULT_TYPE, TrustLevel.LOW  # без разборчивого URL доверия нет
    if AGGREGATOR_RE.search(host):
        return SourceType.NEWS, TrustLevel.LOW  # §2.8: агрегаторы — пониженная доверенность
    for pattern, stype, trust in _COMPILED:
        if pattern.search(host):
            return stype, trust
    return DEFAULT_TYPE, DEFAULT_TRUST


def resolve(url: str,
            source_type: Optional[SourceType] = None,
            trust_level: Optional[TrustLevel] = None) -> Tuple[SourceType, TrustLevel]:
    """Метаданные источника с приоритетом уже проставленных значений.

    Если Участник 1 (trust-слой) или Участник 3 (парсер) уже определили тип и доверенность —
    берём их. Домен разбираем только для недостающих полей.
    """
    if source_type is not None and trust_level is not None:
        return source_type, trust_level
    guessed_type, guessed_trust = classify_url(url)
    return (source_type or guessed_type), (trust_level or guessed_trust)


__all__ = ["domain_of", "classify_url", "resolve", "DOMAIN_RULES"]
