"""Ручная проверка коннекторов: python -m parser.tests.smoke_arxiv

Запускает arXiv-коннектор по запросу, нормализует в SignalDoc и печатает результат.
Не требует сетевых сервисов, кроме интернета.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from parser.app.connectors import discover_connectors, registry  # noqa: E402
from parser.app.normalize import normalize_all  # noqa: E402


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "optical interconnects chiplet"
    area = sys.argv[2] if len(sys.argv) > 2 else None

    discover_connectors()
    print("Коннекторы:", registry.names())

    connector = registry.by_name("arxiv")
    if connector is None:
        print("arxiv не зарегистрирован")
        return

    raw = await connector.search(query, limit=5)
    print("Сырых результатов:", len(raw))
    if not raw:
        print("(пусто — проверить сеть/запрос)")
        return

    docs = normalize_all(raw, query=query, area=area)
    print("Документов после нормализации:", len(docs))
    for d in docs[:5]:
        print("-" * 70)
        print("id:", d.id)
        print("title:", d.title)
        print("area:", d.area)
        print("sources:", len(d.sources))
        for s in d.sources:
            print("  ", s.source_type.value, "|", s.trust_level.value, "|", s.language, "|", s.date, "|", s.url)


if __name__ == "__main__":
    asyncio.run(main())