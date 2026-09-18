"""Smoke-тест /collect: python -m parser.tests.smoke_collect.

Проверяет весь путь оркестратора: фан-аут по коннекторам -> нормализация
-> дедуп -> CollectResponse (контакт /shared/contracts.py не менять).
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from parser.app.connectors import discover_connectors  # noqa: E402
from parser.app.orchestrator import collect  # noqa: E402


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "optical interconnect"
    area = sys.argv[2] if len(sys.argv) > 2 else None

    discover_connectors()
    resp = await collect(query, area=area, limit=10)
    print("query:", resp.query)
    print("sources_processed:", resp.sources_processed)
    print("documents:", len(resp.documents))
    for d in resp.documents[:5]:
        print("-", d.title, "|", d.area, "|", len(d.sources), "src")


if __name__ == "__main__":
    asyncio.run(main())