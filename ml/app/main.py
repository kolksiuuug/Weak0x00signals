"""ML-сервис: интерпретируемая scikit-learn модель + прозрачная fallback-эвристика."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI

from ml.app.model import WeakSignalModel
from shared.contracts import HealthResponse, ScoreBatchRequest, ScoreBatchResponse, ScoreRequest, ScoreResponse
from shared.schema import SignalDoc

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("ml")
app = FastAPI(title="Слабые сигналы — ML", version="1.0.0")

STAGE_LABELS = {1: "Концепция/Исследование", 2: "Прототип/PoC", 3: "Пилот", 4: "Раннее внедрение"}
TREND_LABELS = {1: "упоминания стабильны", 2: "упоминания растут", 3: "упоминания растут быстро"}
HYPE_KEYWORDS = ("хайп", "инфошум", "перегрет", "overhyped", "marketing buzz")
WEAK_THRESHOLD = float(os.getenv("WEAK_SIGNAL_THRESHOLD", "0.55"))
MATURE_STAGE = int(os.getenv("MATURE_STAGE", "4"))
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/weak_signal.joblib")
model = WeakSignalModel(MODEL_PATH)


def maturity_reason(doc: SignalDoc) -> Optional[str]:
    text = (doc.raw_text or "").lower()
    if doc.stage == MATURE_STAGE and (doc.trend or 1) <= 1:
        return "Отклонено фильтром зрелости: раннее внедрение при стабильном тренде — признаков зарождения недостаточно."
    if any(k in text for k in HYPE_KEYWORDS):
        return "Отклонено как маркетинговый хайп/инфошум: в контексте есть явные признаки перегретой информационной повестки."
    if len(doc.sources) == 1 and doc.sources[0].trust_level.value == "пониженный":
        return "Отклонено: единственное основание имеет пониженную доверенность."
    return None


def score_document(doc: SignalDoc, query: Optional[str]) -> ScoreResponse:
    score, predictors = model.predict(doc)
    rejected = maturity_reason(doc)
    stage = doc.stage or 1
    trend = doc.trend or 1
    dataset_score = stage + trend
    weak = rejected is None and score >= WEAK_THRESHOLD
    why = (
        "Уверенность {:.0f} %. Прозрачные признаки: {}. "
        "Балльная часть датасета воспроизводится как стадия {} + тренд {} = {}/7."
    ).format(score * 100, "; ".join(predictors), STAGE_LABELS.get(stage, "?"), TREND_LABELS.get(trend, "?"), dataset_score)
    return ScoreResponse(
        id=doc.id, score=score, is_weak_signal=weak, why=why,
        rejected_reason=rejected, predictors=predictors, dataset_score=dataset_score,
        model_version=model.version, model_mode=model.mode,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(service="ml", version="1.0.0", mock=False, details={"mode": model.mode, "model_version": model.version, "metrics": model.metrics})


@app.get("/metrics")
def metrics() -> dict:
    return {"mode": model.mode, "model_version": model.version, "metrics": model.metrics}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    result = score_document(req.document, req.query)
    return result


@app.post("/score/batch", response_model=ScoreBatchResponse)
def score_batch(req: ScoreBatchRequest) -> ScoreBatchResponse:
    items = [score_document(doc, req.query) for doc in req.documents]
    logger.info("score/batch: documents=%d model=%s", len(items), model.version)
    return ScoreBatchResponse(items=items)
