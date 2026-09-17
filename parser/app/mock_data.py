"""МОК-фикстуры парсера. Заменяются реальными коннекторами (Участник 3).

ВАЖНО (CLAUDE.md §2.3): это данные-заглушки исключительно для сквозного прогона скелета.
Реальная выдача обязана строиться ТОЛЬКО по фактически найденным и проверенным источникам.
Каждый сервис помечает мок-режим полем `mock: true` в /health.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from shared.schema import SignalDoc, Source, SourceType, TrustLevel

AREAS = [
    "Индустриальный ИИ",
    "Инфраструктура ИИ",
    "Роботы",
    "Финтех",
    "Защита ИИ",
    "Edge",
]

# (title, area, companies, stage, trend, raw_text)
_RAW = [
    ("Нейроморфные ускорители для предиктивного обслуживания станков", "Индустриальный ИИ",
     ["Innatera", "SynSense"], 2, 2,
     "Спайковые процессоры начинают применять для вибродиагностики промышленного оборудования: "
     "энергопотребление на два порядка ниже классических DSP при сопоставимой точности."),
    ("Цифровые двойники техпроцессов с онлайн-дообучением", "Индустриальный ИИ",
     ["Neural Concept"], 3, 2,
     "Пилоты цифровых двойников, дообучающихся на потоке телеметрии линии без остановки производства."),
    ("Оптические межсоединения chiplet-to-chiplet", "Инфраструктура ИИ",
     ["Ayar Labs", "Lightmatter"], 2, 3,
     "Кремниевая фотоника переходит из лабораторий в упаковку ускорителей: со-упакованная оптика "
     "снимает ограничение по пропускной способности между чиплетами."),
    ("Дисагрегированная память CXL 3.0 для инференса", "Инфраструктура ИИ",
     ["Panmnesia", "Unifabrix"], 2, 2,
     "Пулы памяти по CXL позволяют держать KV-кэш вне узла; появились первые измерения на инференс-кластерах."),
    ("Жидкостное иммерсионное охлаждение модульных ЦОД", "Инфраструктура ИИ",
     ["Submer", "Asperitas"], 4, 1,
     "Иммерсионное охлаждение массово внедряется в гиперскейл-ЦОД, рынок сформирован, лидеры известны."),
    ("Тактильные сенсоры с самовосстанавливающейся кожей для захватов", "Роботы",
     ["BeBop Sensors"], 1, 2,
     "Полимерные покрытия, восстанавливающие проводящие дорожки после пореза, испытываются "
     "на манипуляторах для пищевой промышленности."),
    ("Мультиагентные рои дронов для инвентаризации складов", "Роботы",
     ["Verity", "Corvus Robotics"], 3, 2,
     "Пилоты автономных роёв внутри помещений без GPS с распределённым планированием маршрутов."),
    ("Мягкие актуаторы на жидкокристаллических эластомерах", "Роботы",
     ["Artimus Robotics"], 1, 2,
     "Актуаторы, меняющие форму при нагреве, дают высокий удельный момент при малой массе."),
    ("Гуманоидные роботы общего назначения", "Роботы",
     ["Figure", "Tesla"], 2, 3,
     "Тема перегрета медийно: объём инфошума кратно превышает объём верифицированных внедрений."),
    ("Программируемые деньги на смарт-контрактах ЦФА", "Финтех",
     ["Мастерчейн"], 2, 2,
     "Целевое расходование средств, зашитое в смарт-контракт цифрового финансового актива."),
    ("Приватный скоринг на гомоморфном шифровании", "Финтех",
     ["Zama", "Duality"], 1, 2,
     "Кредитный скоринг по зашифрованным данным заёмщика без расшифровки на стороне банка."),
    ("Агентные платежи между ИИ-ассистентами", "Финтех",
     ["Skyfire"], 1, 3,
     "Протоколы, в которых ИИ-агент самостоятельно инициирует микроплатёж за данные или вычисления."),
    ("Водяные знаки в весах моделей для доказательства происхождения", "Защита ИИ",
     ["Mithril Security"], 2, 2,
     "Встраивание проверяемой метки прямо в веса, устойчивой к дообучению и квантованию."),
    ("Детекторы отравления обучающих выборок на этапе препроцессинга", "Защита ИИ",
     ["Robust Intelligence"], 2, 2,
     "Статистические фильтры, отлавливающие бэкдор-триггеры до начала обучения."),
    ("Аппаратные анклавы для доверенного инференса", "Защита ИИ",
     ["Edgeless Systems"], 3, 2,
     "Инференс внутри доверенной среды исполнения с удалённой аттестацией — пилоты в регулируемых отраслях."),
    ("Компиляторы под микроконтроллеры для однобитных моделей", "Edge",
     ["Nota AI"], 2, 3,
     "Тернарная квантизация даёт запуск языковых моделей на микроконтроллерах с сотнями килобайт ОЗУ."),
    ("Федеративное дообучение на абонентских устройствах оператора", "Edge",
     ["Flower Labs"], 2, 2,
     "Дообучение персональных моделей без выгрузки данных с устройства, агрегация на стороне оператора."),
    ("Энергонезависимые вычисления в памяти на ReRAM", "Edge",
     ["Weebit Nano", "TetraMem"], 1, 2,
     "Матричные умножения прямо в массиве памяти снимают фон-неймановское бутылочное горлышко."),
]

# Шаблоны источников: полный набор метаданных обязателен (CLAUDE.md §2.7)
_SOURCE_TEMPLATES = [
    dict(url="https://arxiv.org/abs/2608.{n:05d}", source_type=SourceType.SCIENTIFIC,
         language="en", trust_level=TrustLevel.HIGH, translated=True,
         title="Препринт arXiv (автоперевод названия): {title}", date="2026-04-17"),
    dict(url="https://patents.google.com/patent/US1234{n:03d}B2", source_type=SourceType.PATENT,
         language="en", trust_level=TrustLevel.HIGH, translated=True,
         title="Патент US (автоперевод названия): {title}", date="2026-02-05"),
    dict(url="https://www.nature.com/articles/s41586-026-{n:05d}", source_type=SourceType.ANALYTICS,
         language="en", trust_level=TrustLevel.MEDIUM, translated=True,
         title="Отраслевой обзор (автоперевод резюме): {title}", date="2026-05-30"),
    dict(url="https://habr.com/ru/articles/{n:06d}/", source_type=SourceType.BLOG,
         language="ru", trust_level=TrustLevel.LOW, translated=False,
         title="Блог-разбор: {title}", date="2026-06-11"),
    dict(url="https://example-vendor.ru/press/{n:04d}/", source_type=SourceType.PRESS,
         language="ru", trust_level=TrustLevel.LOW, translated=False,
         title="Пресс-релиз вендора: {title}", date="2026-07-01"),
]


def _sources_for(idx: int, title: str) -> List[Source]:
    """Детерминированный набор источников: 2–3 штуки на кандидата.

    Кандидат idx=8 («гуманоидные роботы») намеренно получает ТОЛЬКО источники
    пониженной доверенности — чтобы в скелете было видно срабатывание trust-слоя (§2.8).
    """
    picks = [3, 4] if idx == 8 else [idx % 3, (idx + 1) % 3, 3]
    out: List[Source] = []
    for k, p in enumerate(picks):
        tpl: Dict = dict(_SOURCE_TEMPLATES[p])
        n = idx * 7 + k + 11
        out.append(
            Source(
                title=tpl["title"].format(title=title),
                url=tpl["url"].format(n=n),
                date=tpl["date"],
                source_type=tpl["source_type"],
                language=tpl["language"],
                trust_level=tpl["trust_level"],
                translated=tpl["translated"],
            )
        )
    return out


def build_documents(query: str, area: Optional[str] = None, limit: int = 30) -> List[SignalDoc]:
    """МОК «собранных» документов. Реальная реализация — коннекторы arXiv/OpenAlex/GDELT/PatentsView."""
    docs: List[SignalDoc] = []
    for idx, (title, doc_area, companies, stage, trend, text) in enumerate(_RAW):
        if area and area != doc_area:
            continue
        docs.append(
            SignalDoc(
                id="mock-{:03d}".format(idx),
                title=title,
                area=doc_area,
                companies=list(companies),
                raw_text="{} [МОК-данные, запрос: «{}»]".format(text, query),
                stage=stage,
                trend=trend,
                sources=_sources_for(idx, title),
            )
        )
    return docs[:limit]


def all_sources() -> List[Source]:
    """Плоский список всех источников мок-корпуса — для GET /api/sources."""
    out: List[Source] = []
    for doc in build_documents("проверка источников", limit=len(_RAW)):
        out.extend(doc.sources)
    return out
