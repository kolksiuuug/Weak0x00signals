"""Наполнение БД датасетом организаторов и контрольной выборкой (CLAUDE.md §6, Участник 2).

Запуск:
    python -m ml.db.seed                          # датасет из data/raw + контроль
    python -m ml.db.seed --dataset data/raw/x.xlsx
    python -m ml.db.seed --truncate               # сначала очистить размеченные строки

Что кладём:
  * documents с origin='dataset' (label=1) — строки датасета организаторов;
  * documents с origin='control' (label=0) — контрольная выборка зрелых и хайповых;
  * sources — по одной строке на ссылку, с полным набором метаданных (CLAUDE.md §2.7).

Живые документы парсера (origin='parser') этот скрипт не трогает: он идемпотентен
и работает только со своими двумя источниками данных.

Тип и доверенность источника восстанавливаются из домена (ml/app/source_meta.py):
в датасете организаторов в колонке «Источники» есть только название и ссылка.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ml.app import dataset as ds
from ml.app.source_meta import classify_url, domain_of, guess_language
from ml.db.migrate import database_url

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("ml.db.seed")

DEFAULT_CONTROL = Path("data/control")   # каталог: складываются все *.csv отрицательного класса

#: Выражение для companies подставляется по фактическому типу колонки — см.
#: companies_expression(). В схеме /ml это TEXT[], в db/init.sql — JSONB, и один
#: и тот же параметр в оба типа не вставить.
UPSERT_DOCUMENT = """
INSERT INTO documents (id, title, area, companies, raw_text, stage, trend, points, origin, label)
VALUES (%(id)s, %(title)s, %(area)s, {companies}, %(raw_text)s,
        %(stage)s, %(trend)s, %(points)s, %(origin)s, %(label)s)
ON CONFLICT (id) DO UPDATE SET
    title     = EXCLUDED.title,
    area      = EXCLUDED.area,
    companies = EXCLUDED.companies,
    raw_text  = EXCLUDED.raw_text,
    stage     = EXCLUDED.stage,
    trend     = EXCLUDED.trend,
    points    = EXCLUDED.points,
    origin    = EXCLUDED.origin,
    label     = EXCLUDED.label;
"""

UPSERT_SOURCE = """
INSERT INTO sources (document_id, title, url, published_at, source_type, language,
                     trust_level, translated, domain)
VALUES (%(document_id)s, %(title)s, %(url)s, %(published_at)s, %(source_type)s,
        %(language)s, %(trust_level)s, %(translated)s, %(domain)s)
ON CONFLICT (document_id, url) DO UPDATE SET
    title       = EXCLUDED.title,
    source_type = EXCLUDED.source_type,
    language    = EXCLUDED.language,
    trust_level = EXCLUDED.trust_level,
    domain      = EXCLUDED.domain;
"""


def companies_expression(cur) -> str:
    """Как вставлять companies: как массив или как JSONB.

    Зачем эта развилка. Колонку documents.companies создают два разных файла:
    ml/db/migrations/001 объявляет её TEXT[], а db/init.sql — JSONB. psycopg
    адаптирует список Python в массив PostgreSQL, и в JSONB-колонку такая вставка
    падает с «column companies is of type jsonb but expression is of type text[]».
    Спрашиваем фактический тип у базы и подставляем подходящее выражение, чтобы сид
    работал на любой из двух схем без правок (см. ml/db/README.md).
    """
    cur.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'documents' "
        "AND column_name = 'companies'"
    )
    row = cur.fetchone()
    data_type = (row[0] if row else "") or ""
    if data_type.lower() == "jsonb":
        logger.info("documents.companies имеет тип JSONB (схема db/init.sql) — пишу JSON")
        return "%(companies)s::jsonb"
    return "%(companies)s"


def companies_value(names: List[str], expression: str) -> Any:
    """Значение companies под выбранное выражение: JSON-строка или список."""
    if "jsonb" in expression:
        return json.dumps(names, ensure_ascii=False)
    return names


def make_id(origin: str, title: str) -> str:
    """Детерминированный id: повторный сид обновляет те же строки, а не плодит копии."""
    digest = hashlib.sha1("{}|{}".format(origin, title).encode("utf-8")).hexdigest()[:12]
    return "{}-{}".format(origin, digest)


def rows_to_records(frame, origin: str, label: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    documents: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    for _, row in frame.iterrows():
        title = str(row["technology"]).strip()
        if not title:
            continue
        doc_id = make_id(origin, title)
        points = row.get("points")
        documents.append({
            "id": doc_id,
            "title": title,
            "area": (str(row.get("area") or "").strip() or None),
            "companies": list(row.get("companies") or []),
            "raw_text": str(row.get("why_raw") or ""),
            "stage": int(row["stage"]),
            "trend": int(row["trend"]),
            "points": int(points) if points == points and points is not None else None,  # NaN-safe
            "origin": origin,
            "label": label,
        })

        for source in (row.get("sources") or []):
            url = source.get("url", "")
            if not url:
                continue
            source_type, trust_level = classify_url(url)
            source_title = source.get("title") or url
            sources.append({
                "document_id": doc_id,
                "title": source_title,
                "url": url,
                "published_at": None,          # в датасете организаторов даты публикации нет
                "source_type": source_type.value,
                "language": guess_language(source_title, url),
                "trust_level": trust_level.value,
                "translated": False,
                "domain": domain_of(url),
            })

    return documents, sources


def seed(dataset_path: Optional[Path], control_path: Path, truncate: bool = False) -> int:
    try:
        import psycopg
    except ImportError:
        raise SystemExit("Не установлен psycopg. Установите: pip install -r ml/requirements.txt")

    positives = ds.load_dataset(dataset_path)
    negatives = ds.load_negatives(control_path) if control_path.exists() else None
    if negatives is None:
        logger.warning("Контрольная выборка %s не найдена — заливаю только датасет организаторов.", control_path)

    doc_rows, src_rows = rows_to_records(positives, "dataset", 1)
    if negatives is not None:
        neg_docs, neg_srcs = rows_to_records(negatives, "control", 0)
        doc_rows += neg_docs
        src_rows += neg_srcs

    with psycopg.connect(database_url(), autocommit=False) as conn:
        with conn.cursor() as cur:
            if truncate:
                logger.info("Очищаю размеченные строки (origin IN ('dataset','control'))")
                cur.execute("DELETE FROM documents WHERE origin IN ('dataset', 'control')")

            # Тип companies спрашиваем у базы: схему могли создать миграции (TEXT[])
            # или db/init.sql (JSONB). Значения приводим под выбранное выражение.
            expression = companies_expression(cur)
            statement = UPSERT_DOCUMENT.format(companies=expression)
            for row in doc_rows:
                row["companies"] = companies_value(row["companies"], expression)

            cur.executemany(statement, doc_rows)
            logger.info("documents: записано %d", len(doc_rows))

            if src_rows:
                cur.executemany(UPSERT_SOURCE, src_rows)
                logger.info("sources: записано %d", len(src_rows))

            cur.execute(
                "SELECT origin, count(*) FROM documents "
                "WHERE origin IN ('dataset','control') GROUP BY origin ORDER BY origin"
            )
            for origin, count in cur.fetchall():
                logger.info("  в базе: %-8s %d", origin, count)
        conn.commit()

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Сид БД датасетом и контрольной выборкой")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Путь к датасету организаторов (по умолчанию — первый файл в data/raw)")
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--truncate", action="store_true",
                        help="Удалить существующие размеченные строки перед заливкой")
    args = parser.parse_args(argv)
    return seed(args.dataset, args.control, args.truncate)


if __name__ == "__main__":
    sys.exit(main())
