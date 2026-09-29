"""Фильтр зрелости и хайпа (CLAUDE.md §2.10, §2.8, §2.3).

Требование ТЗ: зрелые технологии, массовое внедрение, отраслевые стандарты,
сформированные рынки с лидерами, маркетинговый хайп и инфошум — исключаются,
и **по каждому кандидату должна быть явная причина отклонения**. Жюри отдельно
оценивает обоснованность исключения, поэтому причина пишется не шаблоном, а с
подстановкой конкретных фактов: сколько источников, какие маркеры сработали.

Молчаливый фильтр — красный флаг CLAUDE.md §7. Ни одна ветка здесь не возвращает
«отклонено» без текста причины.

Порядок правил = приоритет. Первое сработавшее правило и есть причина отклонения:
пользователю нужна одна понятная формулировка, а не список из шести.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ml.app.dataset import STAGE_LABELS, TREND_LABELS
from ml.app.features import Observation, describe

#: Сколько маркеров зрелости в тексте считаем достаточным основанием.
MATURITY_MARKERS_MIN = 2
#: Сколько маркеров хайпа считаем достаточным основанием.
HYPE_MARKERS_MIN = 2
#: Доля источников высокой доверенности, ниже которой хайп не уравновешен фактурой.
HYPE_HIGH_TRUST_MAX = 0.34
#: Число компаний, при котором рынок уже нельзя назвать зарождающимся.
CROWDED_MARKET_MIN = 8


@dataclass
class MaturityVerdict:
    """Результат фильтра. rejected=False означает «кандидат проходит дальше»."""

    rejected: bool
    rule: Optional[str] = None                      # машинный код правила — для логов и статистики
    reason: Optional[str] = None                    # русская формулировка — уходит в rejected_reason
    evidence: List[str] = field(default_factory=list)  # факты, на которых основано решение


def _fmt(items: List[str]) -> str:
    return ", ".join(items) if items else "—"


def check(obs: Observation) -> MaturityVerdict:
    """Проверка кандидата фильтром зрелости и хайпа.

    Возвращает вердикт с явной причиной. Правила идут от «технология зрелая» к
    «оснований для включения недостаточно» — то есть от содержательных к формальным.
    """
    facts = describe(obs)
    maturity_hits: List[str] = facts["maturity_hits"]        # type: ignore[assignment]
    hype_hits: List[str] = facts["hype_hits"]                # type: ignore[assignment]
    n_sources: int = facts["n_sources"]                      # type: ignore[assignment]
    n_low: int = facts["n_low_trust"]                        # type: ignore[assignment]
    n_high: int = facts["n_high_trust"]                      # type: ignore[assignment]
    n_companies: int = facts["n_companies"]                  # type: ignore[assignment]
    share_high = (float(n_high) / n_sources) if n_sources else 0.0

    # --- §2.3: без подтверждённых источников кандидат не существует ---------
    if n_sources == 0:
        return MaturityVerdict(
            rejected=True,
            rule="no_sources",
            reason=(
                "Отклонено: у кандидата нет ни одного подтверждённого источника. "
                "Включение технологии в выдачу без реально найденного источника запрещено "
                "(требование о работе только по проверенным источникам)."
            ),
            evidence=["источников найдено: 0"],
        )

    # --- §2.8: соцсети, блоги и пресс-релизы не могут быть единственным основанием ---
    if n_low == n_sources:
        return MaturityVerdict(
            rejected=True,
            rule="low_trust_only",
            reason=(
                "Отклонено: все {} источник(ов) имеют пониженную доверенность "
                "(соцсети, личные блоги, агрегаторы или пресс-релизы). Такие источники не могут "
                "быть единственным основанием для включения технологии в выдачу.".format(n_sources)
            ),
            evidence=["источников: {}".format(n_sources), "из них пониженной доверенности: {}".format(n_low)],
        )

    # --- §2.10: зрелая технология ------------------------------------------
    if len(maturity_hits) >= MATURITY_MARKERS_MIN:
        return MaturityVerdict(
            rejected=True,
            rule="mature_technology",
            reason=(
                "Отклонено фильтром зрелости: технология вышла из стадии зарождения. "
                "Признаки зрелости в описании и источниках: {}. Такие технологии исключаются "
                "как массово внедрённые и стандартизованные.".format(_fmt(maturity_hits))
            ),
            evidence=maturity_hits,
        )

    # --- §2.10: сформированный рынок с лидерами -----------------------------
    if obs.stage >= 4 and obs.trend <= 1:
        return MaturityVerdict(
            rejected=True,
            rule="formed_market",
            reason=(
                "Отклонено фильтром зрелости: стадия «{}» при характеристике тренда «{}». "
                "Рынок сформирован, рост упоминаний отсутствует — признаков зарождающегося "
                "тренда нет.".format(
                    STAGE_LABELS.get(obs.stage, "неизвестно"),
                    TREND_LABELS.get(obs.trend, "неизвестно"),
                )
            ),
            evidence=[
                "стадия: {}".format(STAGE_LABELS.get(obs.stage, "неизвестно")),
                "тренд: {}".format(TREND_LABELS.get(obs.trend, "неизвестно")),
            ],
        )

    if n_companies >= CROWDED_MARKET_MIN and obs.stage >= 3:
        return MaturityVerdict(
            rejected=True,
            rule="crowded_market",
            reason=(
                "Отклонено фильтром зрелости: на стадии «{}» вокруг технологии уже {} участников "
                "рынка. Это признак сформированного рынка с установившимися лидерами, а не "
                "зарождающегося тренда.".format(STAGE_LABELS.get(obs.stage, "неизвестно"), n_companies)
            ),
            evidence=["компаний-участников: {}".format(n_companies)],
        )

    # --- §2.10: маркетинговый хайп и инфошум --------------------------------
    if len(hype_hits) >= HYPE_MARKERS_MIN and share_high < HYPE_HIGH_TRUST_MAX:
        return MaturityVerdict(
            rejected=True,
            rule="hype",
            reason=(
                "Отклонено как маркетинговый хайп: медийная риторика не подкреплена фактурой. "
                "Сработавшие признаки: {}. При этом источников высокой доверенности (наука, патенты, "
                "госреестры) — {} из {}.".format(_fmt(hype_hits), n_high, n_sources)
            ),
            evidence=hype_hits + ["высокая доверенность: {} из {}".format(n_high, n_sources)],
        )

    return MaturityVerdict(rejected=False, evidence=[
        "источников: {} (высокой доверенности: {})".format(n_sources, n_high),
        "маркеров зрелости: {}".format(len(maturity_hits)),
        "маркеров хайпа: {}".format(len(hype_hits)),
    ])


__all__ = ["MaturityVerdict", "check"]
