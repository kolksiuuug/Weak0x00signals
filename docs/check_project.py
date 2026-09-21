from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

required = [
    "docker-compose.yml", "db/init.sql", "shared/schema.py", "shared/contracts.py",
    "api/app/main.py", "api/app/pipeline.py", "api/app/trust.py",
    "parser/app/main.py", "parser/app/connectors.py", "parser/app/llm.py",
    "parser/app/database.py", "parser/app/embeddings.py", "ml/app/main.py",
    "ml/app/model.py", "ml/app/train.py", "frontend/src/App.jsx",
]
missing = [p for p in required if not (ROOT / p).exists()]
print(json.dumps({"missing": missing, "ok": not missing}, ensure_ascii=False, indent=2))
raise SystemExit(1 if missing else 0)
