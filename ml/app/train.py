"""Однократное обучение: официальный XLSX + собственные негативные кандидаты."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ml.app.model import _negative_csv, train_if_possible
from parser.app.dataset import load_docs


def main() -> None:
    path = Path(os.getenv("MODEL_PATH", "/app/model/weak_signal.joblib"))
    positives = load_docs(100)
    negatives = _negative_csv(Path(os.getenv("NEGATIVE_DATA_PATH", "/app/data/negative_candidates.csv")))
    result = train_if_possible(positives, negatives, path)
    metrics_path = path.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
