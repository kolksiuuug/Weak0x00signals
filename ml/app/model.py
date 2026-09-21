"""Интерпретируемая модель этапа 1: LogisticRegression на инженерных признаках."""
from __future__ import annotations

import csv
import json
import logging
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.schema import SignalDoc

logger = logging.getLogger("ml.model")
MODEL_VERSION = "weak-signal-logreg-v1"
FEATURE_NAMES = [
    "stage",
    "trend",
    "earliness",
    "trend_acceleration",
    "source_count",
    "company_count",
    "high_trust_ratio",
    "news_ratio",
    "scientific_ratio",
    "recent_ratio",
    "text_length_log",
    "hype_terms",
    "research_terms",
    "deployment_terms",
    "seed_terms",
    "player_concentration",
]

STAGE_MAP = {1: 1.0, 2: 0.67, 3: 0.33, 4: 0.0}
TREND_MAP = {1: 0.0, 2: 0.5, 3: 1.0}
HYPE_TERMS = ("hype", "overhyped", "маркетинг", "хайп", "перегрет", "инфошум")
RESEARCH_TERMS = ("research", "study", "paper", "lab", "исследован", "лаборатор", "концепц")
DEPLOYMENT_TERMS = ("production", "deployment", "deployed", "commercial", "массово", "внедрен", "промышлен")


def features(doc: SignalDoc) -> List[float]:
    text = (doc.title + " " + doc.raw_text).lower()
    total = max(1, len(doc.sources))
    high = sum(1 for s in doc.sources if s.trust_level.value == "высокий")
    news = sum(1 for s in doc.sources if s.source_type.value == "новость")
    scientific = sum(1 for s in doc.sources if s.source_type.value == "научная_статья")
    dated = [s.date for s in doc.sources if s.date]
    cutoff = (date.today() - timedelta(days=180)).isoformat()
    recent = sum(1 for d in dated if str(d) >= cutoff)
    return [
        float(doc.stage or 1),
        float(doc.trend or 1),
        STAGE_MAP.get(doc.stage or 1, 0.0),
        TREND_MAP.get(doc.trend or 1, 0.0),
        float(len(doc.sources)),
        float(len(doc.companies)),
        high / total,
        news / total,
        scientific / total,
        recent / max(1, len(dated)),
        float(np.log1p(len(text))),
        float(sum(t in text for t in HYPE_TERMS)),
        float(sum(t in text for t in RESEARCH_TERMS)),
        float(sum(t in text for t in DEPLOYMENT_TERMS)),
        float(sum(t in text for t in ("seed round", "seed", "посев", "pre-seed", "венчур"))),
        1.0 / max(1, len(doc.companies)),
    ]


def _negative_csv(path: Path) -> List[SignalDoc]:
    if not path.exists():
        return []
    docs = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            docs.append(SignalDoc(
                id="negative-" + row.get("id", str(len(docs))),
                title=row.get("title", "").strip(),
                area=row.get("area") or None,
                companies=[x.strip() for x in (row.get("companies") or "").split(";") if x.strip()],
                raw_text=row.get("raw_text", ""),
                stage=int(row["stage"]) if row.get("stage") else None,
                trend=int(row["trend"]) if row.get("trend") else None,
                rejected_reason=row.get("reason") or "Размечено командой как негативный кандидат.",
            ))
    return docs


def train_if_possible(positive_docs: List[SignalDoc], negative_docs: List[SignalDoc], model_path: Path) -> Dict:
    negatives = list(negative_docs)
    allow_synth = os.getenv("ALLOW_SYNTHETIC_NEGATIVES", "false").lower() == "true"
    if not negatives and allow_synth:
        for doc in positive_docs:
            negatives.append(doc.model_copy(update={
                "id": "synthetic-" + doc.id,
                "stage": 4,
                "trend": 1,
                "raw_text": doc.raw_text + " commercial deployment marketing hype",
                "rejected_reason": "synthetic negative for development only",
            }))
    samples = positive_docs + negatives
    if len(positive_docs) < 10 or len(negatives) < 10:
        return {
            "mode": "heuristic",
            "model_version": "heuristic-v2",
            "train_samples": len(samples),
            "positive_samples": len(positive_docs),
            "negative_samples": len(negatives),
            "reason": "Для LogisticRegression нужны официальный positive train и собственные negative candidates. "
                      "Сейчас используется прозрачная fallback-эвристика; синтетические негативы не считаются финальными метриками.",
        }

    x = np.asarray([features(d) for d in samples], dtype=float)
    y = np.asarray([1] * len(positive_docs) + [0] * len(negatives))
    stratify = y if min(np.sum(y == 0), np.sum(y == 1)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=stratify)
    pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))])
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    metrics = {
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "report": classification_report(y_test, pred, zero_division=0, output_dict=True),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "feature_names": FEATURE_NAMES, "metrics": metrics, "version": MODEL_VERSION}, model_path)
    return {
        "mode": "logreg",
        "model_version": MODEL_VERSION,
        "train_samples": len(samples),
        "positive_samples": len(positive_docs),
        "negative_samples": len(negatives),
        **metrics,
    }


class WeakSignalModel:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.bundle = None
        if self.model_path.exists():
            try:
                self.bundle = joblib.load(self.model_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Не удалось загрузить ML-модель: %s", exc)

    @property
    def mode(self) -> str:
        return "logreg" if self.bundle else "heuristic"

    @property
    def version(self) -> str:
        return self.bundle.get("version", MODEL_VERSION) if self.bundle else "heuristic-v2"

    @property
    def metrics(self) -> Dict:
        return self.bundle.get("metrics", {}) if self.bundle else {}

    def predict(self, doc: SignalDoc) -> Tuple[float, List[str]]:
        if self.bundle:
            x = np.asarray([features(doc)], dtype=float)
            probability = float(self.bundle["pipeline"].predict_proba(x)[0, 1])
            coefs = self.bundle["pipeline"].named_steps["model"].coef_[0]
            pairs = sorted(zip(FEATURE_NAMES, coefs), key=lambda p: abs(p[1]), reverse=True)[:5]
            predictors = ["{}: {:+.2f}".format(name, weight) for name, weight in pairs]
            return round(max(0.0, min(1.0, probability)), 3), predictors
        # Детерминированная прозрачная fallback-модель.
        stage = doc.stage or 1
        trend = doc.trend or 1
        high_ratio = sum(1 for s in doc.sources if s.trust_level.value == "высокий") / max(1, len(doc.sources))
        earliness = {1: 1.0, 2: 0.7, 3: 0.35, 4: 0.0}.get(stage, 0.0)
        momentum = {1: 0.0, 2: 0.55, 3: 1.0}.get(trend, 0.0)
        sparse = 0.5 if len(doc.companies) <= 3 else 0.0
        evidence = 0.4 if len(doc.sources) <= 3 else 0.1
        trust = 0.15 * high_ratio
        hype = 0.4 if any(t in (doc.raw_text or "").lower() for t in HYPE_TERMS) else 0.0
        score = 0.36 * momentum + 0.34 * earliness + 0.12 * sparse + 0.12 * evidence + trust - hype
        score = max(0.0, min(1.0, score))
        predictors = [
            "стадия={} → ранность={:.2f}".format(stage, earliness),
            "тренд={} → импульс={:.2f}".format(trend, momentum),
            "источников={}".format(len(doc.sources)),
            "компаний={}".format(len(doc.companies)),
            "источники высокого доверия={:.0f}%".format(high_ratio * 100),
        ]
        return round(score, 3), predictors
