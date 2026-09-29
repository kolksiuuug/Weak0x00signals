"""Инженерные признаки «слабости» сигнала (CLAUDE.md §6, Участник 2).

Ключевое требование: признаки считаются ОДИНАКОВО из строки датасета организаторов
(этап 1) и из SignalDoc, пришедшего из открытого RAG-пайплайна (этап 2). Иначе модель,
обученная на датасете, на живых документах работает вне своего распределения.

Поэтому обе ветки сначала сводятся к промежуточному объекту Observation, и только
он превращается в вектор признаков.

Признаки намеренно немногочисленные, плотные и объяснимые: 100 наблюдений не
выдерживают сотен разреженных фич, а жюри отдельно оценивает интерпретируемость.
Никаких TF-IDF-мешков и эмбеддингов в основной модели — вклад такого признака
нельзя показать словами.

Python 3.8+: аннотации через typing, dataclasses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ml.app import lexicon, source_meta
from ml.app.dataset import (
    STAGE_LABELS, STAGE_MAX, STAGE_MIN, TREND_LABELS, TREND_MAX, TREND_MIN,
    encode_stage, encode_trend,
)
from shared.schema import SignalDoc, SourceType, TrustLevel

#: Порог «мало игроков»: раннему рынку свойственна концентрация вокруг 1-2 команд.
FEW_PLAYERS_MAX = 2
#: Порог «разреженность публикаций»: у слабого сигнала источников единицы.
SPARSE_SOURCES_MAX = 3

#: Имя признака -> подпись для человека. Подпись идёт в поле `why` ответа /score,
#: поэтому формулировки сразу пишутся так, как их увидит пользователь.
FEATURE_SPECS: Tuple[Tuple[str, str], ...] = (
    ("stage", "стадия развития (1 — концепция … 4 — раннее внедрение)"),
    ("trend", "тренд упоминаний (1 — стабильно … 3 — растёт быстро)"),
    ("n_companies_log", "число компаний-участников"),
    ("few_players", "концентрация игроков: не более {} компаний".format(FEW_PLAYERS_MAX)),
    ("n_sources_log", "число найденных источников"),
    ("sparse_pubs", "разреженность публикаций: не более {} источников".format(SPARSE_SOURCES_MAX)),
    ("n_high_trust_log", "число источников высокой доверенности"),
    ("n_low_trust_log", "число источников пониженной доверенности"),
    ("share_high_trust", "доля источников высокой доверенности"),
    ("share_low_trust", "доля источников пониженной доверенности (соцсети, блоги, пресс-релизы)"),
    ("has_scientific", "есть научная публикация"),
    ("has_patent", "есть патент"),
    ("share_news", "доля новостных источников"),
    ("weakness_markers", "текстовые маркеры ранней стадии"),
    ("maturity_markers", "текстовые маркеры зрелости технологии"),
    ("hype_markers", "текстовые маркеры маркетингового хайпа"),
    ("early_language_density", "плотность терминов ранней стадии в тексте"),
    ("maturity_language_density", "плотность терминов зрелой технологии в тексте"),
    ("hype_language_density", "плотность терминов хайпа в тексте"),
    ("technical_context_density", "плотность технического контекста"),
    ("experimental_context_density", "плотность экспериментального и исследовательского контекста"),
)

FEATURE_NAMES: Tuple[str, ...] = tuple(name for name, _ in FEATURE_SPECS)
FEATURE_LABELS: Dict[str, str] = dict(FEATURE_SPECS)


# Широкие, но интерпретируемые контекстные термины для live-текстов.
# Они дополняют строгий lexicon: одинаковая стадия «исследование» не означает,
# что два документа содержат одинаковую плотность признаков раннего рынка.
_EARLY_LANGUAGE = (
    "research", "researching", "study", "studies", "proposed", "prototype",
    "proof of concept", "poc", "pilot", "emerging", "novel", "nascent",
    "explores", "experimental", "лаборатор", "исследован", "концепц",
    "прототип", "эксперимент", "новый подход", "перспективн", "пилот",
)
_MATURITY_LANGUAGE = (
    "production", "deployed", "deployment", "commercial", "mass adoption",
    "industrial use", "standard", "market leader", "массов", "внедрен",
    "промышленн", "стандарт", "серийн", "рынок сформирован",
)
_HYPE_LANGUAGE = (
    "hype", "buzz", "revolutionary", "game changer", "game-changer",
    "breakthrough", "will change everything", "хайп", "революц", "ажиотаж",
    "инфошум", "прорыв",
)
_EXPERIMENTAL_CONTEXT = (
    "prototype", "pilot", "experiment", "experimental", "benchmark", "simulation",
    "proof of concept", "poc", "preliminary", "research", "study", "paper",
    "прототип", "пилот", "эксперимент", "симуляц", "предварительн", "исследован",
    "научн", "лаборатор", "тестирован", "испытан",
)
_TECHNICAL_CONTEXT = (
    "algorithm", "model", "dataset", "benchmark", "experiment", "experimental",
    "architecture", "inference", "training", "simulation", "sensor", "control",
    "predictive", "maintenance", "manufacturing", "robotics", "industrial",
    "алгоритм", "модель", "датасет", "эксперимент", "архитектур", "инференс",
    "обучен", "симуляц", "сенсор", "управлен", "прогнозн", "техническ",
    "производств", "промышленн",
)

def _language_density(text: str, terms: Sequence[str]) -> float:
    norm = lexicon.normalize_ru(text)
    if not norm:
        return 0.0
    hits = sum(1 for term in terms if lexicon.normalize_ru(term) in norm)
    # Нормируем на sqrt длины: длинная статья не получает искусственный бонус
    # только за количество слов, а короткие тезисы не подавляются слишком сильно.
    return min(1.0, hits / max(1.0, math.sqrt(len(norm) / 80.0)))

def _marker_score(matches: Sequence[str]) -> float:
    """Плавно переводит число совпавших маркеров в диапазон 0..1.

    В отличие от min(1, n/3), новые совпадения после третьего не теряются
    полностью: 1 -> 0.39, 2 -> 0.63, 3 -> 0.78, 5 -> 0.92, 10 -> 0.99.
    """
    if not matches:
        return 0.0
    return 1.0 - math.exp(-len(matches) / 2.0)


@dataclass
class Observation:
    """Наблюдение, приведённое к единому виду: и строка датасета, и живой SignalDoc."""

    title: str
    text: str                                       # обоснование + нормализованный текст источников
    stage: int                                      # 1..4
    trend: int                                      # 1..3
    companies: List[str] = field(default_factory=list)
    sources: List[Tuple[SourceType, TrustLevel]] = field(default_factory=list)
    area: Optional[str] = None
    doc_id: Optional[str] = None
    #: True, если стадия/тренд восстановлены из текста, а не взяты из размеченного поля.
    stage_inferred: bool = False
    trend_inferred: bool = False


# --- адаптеры ---------------------------------------------------------------

def observation_from_row(row) -> Observation:
    """Строка нормализованного датасета (ml/app/dataset.load_dataset) -> Observation."""
    sources = [source_meta.classify_url(src.get("url", "")) for src in (row.get("sources") or [])]
    text_parts = [str(row.get("technology") or ""), str(row.get("why_raw") or ""),
                  str(row.get("stage_raw") or ""), str(row.get("trend_raw") or "")]
    return Observation(
        title=str(row.get("technology") or ""),
        text=" ".join(part for part in text_parts if part),
        stage=int(row.get("stage") or STAGE_MIN),
        trend=int(row.get("trend") or TREND_MIN),
        companies=list(row.get("companies") or []),
        sources=sources,
        area=str(row.get("area") or "") or None,
    )


def _infer_stage(text: str) -> int:
    value = encode_stage(text)
    return max(STAGE_MIN, min(STAGE_MAX, int(value)))


def _infer_trend(text: str) -> int:
    value = encode_trend(text)
    return max(TREND_MIN, min(TREND_MAX, int(value)))


def observation_from_doc(doc: SignalDoc) -> Observation:
    """SignalDoc из открытого пайплайна -> Observation.

    Стадия и тренд в SignalDoc необязательны (Участник 3 их проставляет не всегда).
    Если поля пустые — восстанавливаем их из текста теми же правилами, что разбирают
    колонки датасета, и помечаем как выведенные: это попадает в объяснение.
    """
    sources: List[Tuple[SourceType, TrustLevel]] = [
        source_meta.resolve(str(src.url), src.source_type, src.trust_level)
        for src in doc.sources
    ]
    text = " ".join(part for part in (doc.title, doc.why or "", doc.raw_text or "") if part)

    stage_inferred = doc.stage is None
    trend_inferred = doc.trend is None
    stage = (
        int(doc.stage)
        if doc.stage is not None
        else _infer_stage(text)
    )
    trend = (
        int(doc.trend)
        if doc.trend is not None
        else _infer_trend(text)
    )

    return Observation(
        title=doc.title,
        text=text,
        stage=max(STAGE_MIN, min(STAGE_MAX, stage)),
        trend=max(TREND_MIN, min(TREND_MAX, trend)),
        companies=list(doc.companies or []),
        sources=sources,
        area=doc.area,
        doc_id=doc.id,
        stage_inferred=stage_inferred,
        trend_inferred=trend_inferred,
    )


# --- вектор признаков -------------------------------------------------------

def _share(values: Sequence[bool]) -> float:
    return float(sum(1 for v in values if v)) / len(values) if values else 0.0


def build_features(obs: Observation) -> Dict[str, float]:
    """Observation -> словарь признаков. Ключи и порядок совпадают с FEATURE_NAMES."""
    n_companies = len(obs.companies)
    n_sources = len(obs.sources)
    types = [stype for stype, _ in obs.sources]
    trusts = [trust for _, trust in obs.sources]

    weakness = lexicon.weakness_hits(obs.text)
    maturity = lexicon.maturity_hits(obs.text)
    hype = lexicon.hype_hits(obs.text)

    values: Dict[str, float] = {
        "stage": float(obs.stage),
        "trend": float(obs.trend),
        "n_companies_log": math.log1p(n_companies),
        "few_players": 1.0 if n_companies <= FEW_PLAYERS_MAX else 0.0,
        "n_sources_log": math.log1p(n_sources),
        "sparse_pubs": 1.0 if n_sources <= SPARSE_SOURCES_MAX else 0.0,
        "n_high_trust_log": math.log1p(sum(1 for t in trusts if t == TrustLevel.HIGH)),
        "n_low_trust_log": math.log1p(sum(1 for t in trusts if t == TrustLevel.LOW)),
        "share_high_trust": _share([t == TrustLevel.HIGH for t in trusts]),
        "share_low_trust": _share([t == TrustLevel.LOW for t in trusts]),
        "has_scientific": 1.0 if SourceType.SCIENTIFIC in types else 0.0,
        "has_patent": 1.0 if SourceType.PATENT in types else 0.0,
        "share_news": _share([t == SourceType.NEWS for t in types]),
        "weakness_markers": _marker_score(weakness),
        "maturity_markers": _marker_score(maturity),
        "hype_markers": _marker_score(hype),
        "early_language_density": _language_density(obs.text, _EARLY_LANGUAGE),
        "maturity_language_density": _language_density(obs.text, _MATURITY_LANGUAGE),
        "hype_language_density": _language_density(obs.text, _HYPE_LANGUAGE),
        "technical_context_density": _language_density(obs.text, _TECHNICAL_CONTEXT),
        "experimental_context_density": _language_density(obs.text, _EXPERIMENTAL_CONTEXT),
    }
    return {name: values[name] for name in FEATURE_NAMES}


def feature_vector(obs: Observation) -> List[float]:
    """Признаки в фиксированном порядке FEATURE_NAMES — то, что уходит в sklearn."""
    features = build_features(obs)
    return [features[name] for name in FEATURE_NAMES]


def describe(obs: Observation) -> Dict[str, object]:
    """Сырые факты наблюдения для объяснения (не для модели).

    Возвращает то, что можно показать пользователю дословно: сработавшие маркеры,
    расшифровку стадии и тренда, балл датасета по формуле §1.
    """
    return {
        "stage": obs.stage,
        "stage_label": STAGE_LABELS.get(obs.stage, "неизвестно"),
        "stage_inferred": obs.stage_inferred,
        "trend": obs.trend,
        "trend_label": TREND_LABELS.get(obs.trend, "неизвестно"),
        "trend_inferred": obs.trend_inferred,
        "points": obs.stage + obs.trend,
        "n_companies": len(obs.companies),
        "n_sources": len(obs.sources),
        "n_high_trust": sum(1 for _, t in obs.sources if t == TrustLevel.HIGH),
        "n_low_trust": sum(1 for _, t in obs.sources if t == TrustLevel.LOW),
        "weakness_hits": lexicon.weakness_hits(obs.text),
        "maturity_hits": lexicon.maturity_hits(obs.text),
        "hype_hits": lexicon.hype_hits(obs.text),
    }


__all__ = [
    "Observation", "FEATURE_NAMES", "FEATURE_LABELS", "FEATURE_SPECS",
    "observation_from_row", "observation_from_doc",
    "build_features", "feature_vector", "describe",
]
