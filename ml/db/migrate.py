"""Применение SQL-миграций к PostgreSQL (CLAUDE.md §6, Участник 2).

Запуск:
    python -m ml.db.migrate                 # применить непринятые миграции
    python -m ml.db.migrate --status        # показать состояние, ничего не менять
    python -m ml.db.migrate --dry-run       # показать, что будет применено

Строка подключения берётся из переменной окружения DATABASE_URL (её задаёт
docker-compose). Ключи и пароли в коде не хранятся (CLAUDE.md §5).

Почему свой раннер, а не Alembic: миграций несколько, они на чистом SQL и должны
читаться жюри без знания Python-ORM. Alembic здесь добавил бы слой абстракции,
ничего не упростив.

Идемпотентность: применённые файлы отмечаются в таблице schema_migrations по имени
и контрольной сумме. Если файл изменился после применения — запуск падает с явной
ошибкой, а не переписывает историю молча.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("ml.db.migrate")

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

#: Перевод строки отдельной константой: в .format() внутри тройных кавычек
#: обратный слэш читается хуже, чем явное имя.
NEWLINE = "\n"

CREATE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name        TEXT PRIMARY KEY,
    checksum    TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise SystemExit(
            "Не задана переменная окружения DATABASE_URL. "
            "Локально: DATABASE_URL=postgresql://signals:<пароль>@localhost:5432/signals"
        )
    # psycopg не понимает префикс SQLAlchemy postgresql+psycopg://
    return url.replace("postgresql+psycopg://", "postgresql://")


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def discover() -> List[Tuple[str, str, str]]:
    """Все миграции в порядке имени: (имя, SQL, контрольная сумма)."""
    if not MIGRATIONS_DIR.exists():
        raise SystemExit("Каталог миграций не найден: {}".format(MIGRATIONS_DIR))
    out = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        out.append((path.name, sql, _checksum(sql)))
    if not out:
        raise SystemExit("В {} нет ни одного .sql".format(MIGRATIONS_DIR))
    return out


#: Колонки, без которых сид и обучающая выборка работать не могут. Их добавляет
#: 001_core_schema.sql — либо создавая таблицу с нуля, либо блоком совместимости
#: (ALTER ... ADD COLUMN IF NOT EXISTS) поверх таблицы, созданной db/init.sql.
REQUIRED_DOCUMENT_COLUMNS = ("origin", "label", "points")

#: Колонки /ml сверх db/init.sql в остальных таблицах.
REQUIRED_SOURCE_COLUMNS = ("domain",)


def _columns_of(cur, table: str) -> set:
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s",
        (table,),
    )
    return {row[0] for row in cur.fetchall()}


def _detect_foreign_schema(cur) -> Optional[List[str]]:
    """Создал ли таблицы не наш файл (почти наверняка db/init.sql).

    Возвращает список недостающих колонок или None, если проверять нечего.
    Раньше здесь был отказ работать. Отказ снят осознанно: 001_core_schema.sql
    теперь содержит блок совместимости и сам добавляет недостающие колонки, не
    трогая чужие. Значит накатывать миграции поверх схемы init.sql безопасно —
    расхождение закрывается, а живые вставки парсера продолжают работать.
    """
    cur.execute("SELECT to_regclass('public.documents') IS NOT NULL")
    (exists,) = cur.fetchone()
    if not exists:
        return None  # чистая база — миграции создадут схему сами

    present = _columns_of(cur, "documents")
    missing = [name for name in REQUIRED_DOCUMENT_COLUMNS if name not in present]
    return missing or None


def _verify_schema_after_migrate(cur) -> None:
    """Проверка ПОСЛЕ применения: колонки, нужные сиду, действительно появились.

    Это замена прежнего отказа на входе. Раньше раннер решал заранее, что схема
    чужая, и не запускался; теперь он запускается и проверяет результат. Так ошибка
    ловится по факту, а не по догадке, и сообщение указывает на настоящую причину.
    """
    problems = []

    present_docs = _columns_of(cur, "documents")
    missing_docs = [n for n in REQUIRED_DOCUMENT_COLUMNS if n not in present_docs]
    if missing_docs:
        problems.append("documents: нет колонок {}".format(", ".join(missing_docs)))

    present_src = _columns_of(cur, "sources")
    missing_src = [n for n in REQUIRED_SOURCE_COLUMNS if n not in present_src]
    if missing_src:
        problems.append("sources: нет колонок {}".format(", ".join(missing_src)))

    if not problems:
        return

    raise SystemExit(
        """Миграции применены, но схема всё ещё не пригодна для ml.db.seed:
  {}
Блок совместимости в 001_core_schema.sql должен был добавить эти колонки.
Если 001 помечена как применённая, а колонок нет — база инициализирована
db/init.sql уже ПОСЛЕ применения миграций (пересоздан том postgres_data).
Лечится повторным прогоном с чистой историей: DROP TABLE schema_migrations;
затем python -m ml.db.migrate.""".format((NEWLINE + '  ').join(problems))
    )


def run(dry_run: bool = False, status_only: bool = False) -> int:
    try:
        import psycopg
    except ImportError:
        raise SystemExit("Не установлен psycopg. Установите зависимости: pip install -r ml/requirements.txt")

    migrations = discover()

    with psycopg.connect(database_url(), autocommit=False) as conn:
        with conn.cursor() as cur:
            foreign = _detect_foreign_schema(cur)
            if foreign:
                logger.info(
                    "Таблица documents уже существует и создана не миграциями (нет колонок: %s) — "
                    "почти наверняка это db/init.sql из /docker-entrypoint-initdb.d/. Работаю "
                    "в режиме совместимости: 001_core_schema.sql добавит недостающие колонки, "
                    "чужие не тронет.",
                    ", ".join(foreign),
                )
            cur.execute(CREATE_HISTORY_TABLE)
            conn.commit()
            cur.execute("SELECT name, checksum FROM schema_migrations")
            applied = {name: checksum for name, checksum in cur.fetchall()}

        if status_only:
            logger.info("Состояние миграций:")
            for name, _, checksum in migrations:
                if name not in applied:
                    state = "НЕ ПРИМЕНЕНА"
                elif applied[name] != checksum:
                    state = "ИЗМЕНЕНА ПОСЛЕ ПРИМЕНЕНИЯ"
                else:
                    state = "применена"
                logger.info("  %-28s %s", name, state)
            return 0

        pending = []
        for name, sql, checksum in migrations:
            if name not in applied:
                pending.append((name, sql, checksum))
            elif applied[name] != checksum:
                raise SystemExit(
                    "Миграция {} изменена после применения (контрольная сумма {} != {}). "
                    "Не переписываю историю: создайте новую миграцию вместо правки старой.".format(
                        name, checksum, applied[name])
                )

        if not pending:
            logger.info("Непринятых миграций нет — схема актуальна.")
            # Проверяем даже здесь: если том postgres_data пересоздали, init.sql мог
            # заново создать таблицы ПОСЛЕ того, как миграции уже отмечены применёнными.
            with conn.cursor() as cur:
                _verify_schema_after_migrate(cur)
            return 0

        for name, sql, checksum in pending:
            if dry_run:
                logger.info("[dry-run] применил бы %s (%d символов SQL)", name, len(sql))
                continue
            logger.info("Применяю %s", name)
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (name, checksum) VALUES (%s, %s)",
                    (name, checksum),
                )
            conn.commit()
            logger.info("  ок")

        if not dry_run:
            with conn.cursor() as cur:
                _verify_schema_after_migrate(cur)
            logger.info("Схема пригодна для ml.db.seed: обязательные колонки на месте.")

    logger.info("Готово: %s миграций", "проверено {}".format(len(pending)) if dry_run else "применено {}".format(len(pending)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Миграции схемы PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Показать, что будет применено")
    parser.add_argument("--status", action="store_true", help="Показать состояние и выйти")
    args = parser.parse_args(argv)
    return run(dry_run=args.dry_run, status_only=args.status)


if __name__ == "__main__":
    sys.exit(main())
