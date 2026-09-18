"""Smoke-тест FastAPI: python -m parser.tests.smoke_api.

Проверяет /health, /models, POST /collect, GET /sources без поднятия uvicorn.
Контракт ответов не меняется (shared/contracts.py).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi.testclient import TestClient  # noqa: E402

from parser.app.connectors import discover_connectors  # noqa: E402
from parser.app.main import app  # noqa: E402


def main() -> None:
    discover_connectors()
    client = TestClient(app)

    h = client.get("/health")
    print("GET /health ->", h.status_code, h.json())

    m = client.get("/models")
    print("GET /models ->", m.status_code, "selected:", m.json().get("selected"))

    r = client.post("/collect", json={"query": "optical interconnect", "limit": 5})
    print("POST /collect ->", r.status_code)
    body = r.json()
    print("  sources_processed:", body["sources_processed"], "| documents:", len(body["documents"]))
    for d in body["documents"][:3]:
        print("   -", d["title"], "|", len(d["sources"]), "src")

    s = client.get("/sources?limit=5")
    print("GET /sources ->", s.status_code, "| total:", s.json().get("total"))


if __name__ == "__main__":
    main()