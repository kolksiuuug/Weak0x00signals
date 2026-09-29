"""Инференс модели этапа 1: score(doc) -> (score, why) + фильтр зрелости.

Контракт наружу не меняется: ScoreRequest -> ScoreResponse из shared/contracts.py
(CLAUDE.md §3). Меняется только начинка — вместо правила-заглушки работает обученная
scikit-learn модель.

Интерпретируемость (CLAUDE.md §2.2) реализована честно, а не пересказом:
основная модель — логистическая регрессия на стандартизованных признаках, поэтому
вклад каждого признака в решение считается точно:

    вклад_j = coef_j * (x_j - mean_j) / scale_j
    logit   = intercept + sum(вклад_j)

Эти вклады и попадают в `predictors` и в текст `why`. Никакого post-hoc пересказа:
числа в объяснении — это ровно те числа, из которых сложилось решение.

Деградация: если артефакт модели не обучен (ml/artifacts/model.joblib отсутствует),
сервис поднимается на прозрачных правилах с заранее заданными весами и честно
сообщает об этом в /health (mock=true) и в тексте объяснения. Демо не падает.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ml.app import maturity
from ml.app.features import (
    FEATURE_LABELS, FEATURE_NAMES, Observation, build_features, describe,
    observation_from_doc,
)
from shared.contracts import ScoreResponse
from shared.schema import SignalDoc

logger = logging.getLogger("ml.model")

MODEL_VERSION = "logreg-l2-v4"

ARTIFACT_PATH = Path(os.getenv("MODEL_PATH", "ml/artifacts/model.joblib"))

def _env_threshold(default: float) -> float:
    """`WEAK_SIGNAL_THRESHOLD` из окружения, с защитой от мусора в переменной.

    Влияет ТОЛЬКО на запасной режим. Порог обученного артефакта переопределять нельзя:
    он откалиброван на out-of-fold предсказаниях под требование точности (§1), и
    подмена его числом из compose тихо ломает заявленные Precision/Recall.
    """
    raw = os.getenv("WEAK_SIGNAL_THRESHOLD")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("WEAK_SIGNAL_THRESHOLD=%r — не число, беру %.2f", raw, default)
        return default
    if not 0.0 < value < 1.0:
        logger.warning("WEAK_SIGNAL_THRESHOLD=%.3f вне (0, 1), беру %.2f", value, default)
        return default
    return value


#: Порог отнесения к слабому сигналу, если артефакт не обучен.
FALLBACK_THRESHOLD = _env_threshold(0.55)

#: Прозрачные веса запасного режима. Знак отражает содержательную гипотезу:
#: слабый сигнал — ранняя стадия, растущие упоминания, мало игроков, мало публикаций,
#: научная фактура вместо новостного шума.
FALLBACK_WEIGHTS: Dict[str, float] = {
    "stage": -0.55,
    "trend": 0.70,
    "n_companies_log": -0.30,
    "few_players": 0.45,
    "n_sources_log": -0.25,
    "sparse_pubs": 0.40,
    "n_high_trust_log": 0.20,
    "n_low_trust_log": -0.20,
    "share_high_trust": 0.35,
    "share_low_trust": -0.45,
    "has_scientific": 0.50,
    "has_patent": 0.35,
    "share_news": -0.30,
    "weakness_markers": 0.90,
    "maturity_markers": -1.20,
    "hype_markers": -0.80,
}
FALLBACK_INTERCEPT = -0.20

#: Сколько признаков показывать в объяснении с каждой стороны.
TOP_POSITIVE = 3
TOP_NEGATIVE = 2


# --- артефакт ---------------------------------------------------------------

@dataclass
class ModelArtifact:
    """Всё, что нужно для воспроизводимого инференса и для отчёта на защите."""

    pipeline: Any                                   # sklearn Pipeline: scaler + классификатор
    feature_names: Tuple[str, ...]
    threshold: float                                # порог, откалиброванный на hold-out
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_kind: str = "logistic_regression"
    trained_at: str = ""
    n_train: int = 0
    n_positive: int = 0
    n_negative: int = 0
    notes: str = ""


def save_artifact(artifact: ModelArtifact, path: Path = ARTIFACT_PATH) -> Path:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    logger.info("Артефакт модели сохранён: %s", path)
    return path


def load_artifact(path: Path = ARTIFACT_PATH) -> Optional[ModelArtifact]:
    """Артефакт или None, если он не обучен либо несовместим с текущим набором признаков."""
    path = Path(path)
    if not path.exists():
        logger.warning(
            "Артефакт модели не найден (%s). ML-сервис работает на прозрачных правилах — "
            "обучите модель: python -m ml.app.train", path,
        )
        return None
    try:
        import joblib

        artifact = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 — битый артефакт не должен ронять сервис
        logger.error("Не удалось загрузить артефакт %s: %s. Работаю на правилах.", path, exc)
        return None

    if tuple(artifact.feature_names) != FEATURE_NAMES:
        logger.error(
            "Артефакт обучен на другом наборе признаков (%d против %d). Переобучите модель. "
            "Работаю на правилах.", len(artifact.feature_names), len(FEATURE_NAMES),
        )
        return None
    logger.info(
        "Модель загружена: %s, обучена %s, порог %.3f, F1=%s",
        artifact.model_kind, artifact.trained_at or "—", artifact.threshold,
        artifact.metrics.get("holdout", {}).get("f1"),
    )
    return artifact


# --- вклады признаков -------------------------------------------------------

def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _contributions_from_pipeline(artifact: ModelArtifact,
                                 values: Dict[str, float]) -> Optional[Dict[str, float]]:
    """Точные аддитивные вклады признаков для линейной модели. None — если модель нелинейная."""
    pipeline = artifact.pipeline
    try:
        scaler = pipeline.named_steps.get("scaler")
        clf = pipeline.named_steps.get("clf")
        coefs = clf.coef_[0]
    except (AttributeError, IndexError, KeyError):
        return None

    means = getattr(scaler, "mean_", None)
    scales = getattr(scaler, "scale_", None)
    out: Dict[str, float] = {}
    for index, name in enumerate(artifact.feature_names):
        raw = values.get(name, 0.0)
        if means is not None and scales is not None:
            scale = float(scales[index]) or 1.0
            standardized = (raw - float(means[index])) / scale
        else:
            standardized = raw
        out[name] = float(coefs[index]) * standardized
    return out


def _fallback_contributions(values: Dict[str, float]) -> Dict[str, float]:
    """Вклады запасного режима. Центрируем порядковые шкалы, чтобы знак вклада читался."""
    centers = {"stage": 2.0, "trend": 2.0, "n_companies_log": 1.0, "n_sources_log": 1.0}
    return {
        name: FALLBACK_WEIGHTS.get(name, 0.0) * (values.get(name, 0.0) - centers.get(name, 0.0))
        for name in FEATURE_NAMES
    }


# --- объяснение -------------------------------------------------------------

def _format_predictors(contributions: Dict[str, float]) -> List[str]:
    """Признаки, отсортированные по абсолютному вкладу, с русской подписью и знаком."""
    ordered = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    out = []
    for name, value in ordered:
        if abs(value) < 0.01:
            continue
        out.append("{}: {:+.2f}".format(FEATURE_LABELS.get(name, name), value))
    return out


def _build_why(probability: float,
               facts: Dict[str, Any],
               contributions: Dict[str, float],
               model_kind: str,
               verdict: maturity.MaturityVerdict) -> str:
    """Текст объяснения на русском (CLAUDE.md §2.1) — то, что видит пользователь."""
    positives = [(n, v) for n, v in sorted(contributions.items(), key=lambda kv: kv[1], reverse=True) if v > 0.01]
    negatives = [(n, v) for n, v in sorted(contributions.items(), key=lambda kv: kv[1]) if v < -0.01]

    parts = ["Уверенность модели в слабости сигнала: {:.1f} %.".format(probability * 100)]

    stage_note = " (восстановлена из текста)" if facts.get("stage_inferred") else ""
    trend_note = " (восстановлен из текста)" if facts.get("trend_inferred") else ""
    parts.append(
        "Стадия развития: {}{} — {} балл(а); тренд упоминаний: {}{} — {} балл(а); "
        "итоговый балл по методике датасета: {}/7.".format(
            facts["stage_label"], stage_note, facts["stage"],
            facts["trend_label"], trend_note, facts["trend"],
            facts["points"],
        )
    )

    if positives:
        parts.append("В пользу слабого сигнала: " + "; ".join(
            "{} ({:+.2f})".format(FEATURE_LABELS.get(n, n), v) for n, v in positives[:TOP_POSITIVE]
        ) + ".")
    if negatives:
        parts.append("Против: " + "; ".join(
            "{} ({:+.2f})".format(FEATURE_LABELS.get(n, n), v) for n, v in negatives[:TOP_NEGATIVE]
        ) + ".")

    if facts.get("weakness_hits"):
        parts.append("Текстовые признаки ранней стадии: {}.".format(", ".join(facts["weakness_hits"])))
    if facts.get("maturity_hits"):
        parts.append("Найдены признаки зрелости: {}.".format(", ".join(facts["maturity_hits"])))
    if facts.get("hype_hits"):
        parts.append("Найдены признаки хайпа: {}.".format(", ".join(facts["hype_hits"])))

    parts.append(
        "Фактура: источников {} (из них высокой доверенности {}, пониженной {}), "
        "компаний-участников {}.".format(
            facts["n_sources"], facts["n_high_trust"], facts["n_low_trust"], facts["n_companies"],
        )
    )

    if verdict.rejected and verdict.reason:
        parts.append(verdict.reason)

    if model_kind == "rule_fallback":
        parts.append(
            "Внимание: обученная модель недоступна, оценка получена прозрачными правилами "
            "с фиксированными весами."
        )
    return " ".join(parts)


# --- скорер -----------------------------------------------------------------

class WeakSignalScorer:
    """Обёртка над артефактом. Один экземпляр на процесс, создаётся при старте сервиса."""

    def __init__(self, artifact: Optional[ModelArtifact] = None) -> None:
        self.artifact = artifact
        self.threshold = artifact.threshold if artifact else FALLBACK_THRESHOLD
        self.model_kind = artifact.model_kind if artifact else "rule_fallback"

    @property
    def is_trained(self) -> bool:
        return self.artifact is not None

    @classmethod
    def load(cls, path: Path = ARTIFACT_PATH) -> "WeakSignalScorer":
        return cls(load_artifact(path))

    def info(self) -> Dict[str, Any]:
        """Паспорт модели — отдаётся в /model/info для логирования и защиты."""
        if not self.artifact:
            return {
                "model_kind": "rule_fallback",
                "trained": False,
                "threshold": self.threshold,
                "features": list(FEATURE_NAMES),
                "note": "Модель не обучена. Запустите: python -m ml.app.train",
            }
        return {
            "model_kind": self.artifact.model_kind,
            "trained": True,
            "trained_at": self.artifact.trained_at,
            "threshold": self.artifact.threshold,
            "features": list(self.artifact.feature_names),
            "n_train": self.artifact.n_train,
            "n_positive": self.artifact.n_positive,
            "n_negative": self.artifact.n_negative,
            "metrics": self.artifact.metrics,
            "notes": self.artifact.notes,
        }

    def predict_proba(self, values: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Вероятность слабого сигнала + вклады признаков."""
        if self.artifact is not None:
            vector = [[values[name] for name in self.artifact.feature_names]]
            probability = float(self.artifact.pipeline.predict_proba(vector)[0][1])
            contributions = _contributions_from_pipeline(self.artifact, values)
            if contributions is None:
                # нелинейная модель: показываем прозрачные веса как приближение вклада
                contributions = _fallback_contributions(values)
            return probability, contributions

        contributions = _fallback_contributions(values)
        logit = FALLBACK_INTERCEPT + sum(contributions.values())
        return _sigmoid(logit), contributions

    def score_observation(self, obs: Observation) -> ScoreResponse:
        """Главный контракт инференса: наблюдение -> балл + объяснение + причина отклонения."""
        values = build_features(obs)
        facts = describe(obs)
        probability, contributions = self.predict_proba(values)
        verdict = maturity.check(obs)

        why = _build_why(probability, facts, contributions, self.model_kind, verdict)
        predictors = _format_predictors(contributions)

        return ScoreResponse(
            id=obs.doc_id or obs.title,
            score=round(probability, 3),
            is_weak_signal=bool(not verdict.rejected and probability >= self.threshold),
            why=why,
            rejected_reason=verdict.reason if verdict.rejected else None,
            predictors=predictors,
            model_version=MODEL_VERSION if self.artifact else "rule-fallback-v3",
            model_mode="trained" if self.artifact else "rule_fallback",
        )

    def score_document(self, doc: SignalDoc, query: Optional[str] = None) -> ScoreResponse:
        """SignalDoc -> ScoreResponse. query пока не влияет на балл: релевантность
        запросу обеспечивает поиск на стороне парсера, а модель оценивает слабость сигнала."""
        obs = observation_from_doc(doc)
        response = self.score_observation(obs)
        response.id = doc.id
        return response


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "ARTIFACT_PATH", "MODEL_VERSION", "ModelArtifact", "WeakSignalScorer",
    "save_artifact", "load_artifact", "utc_now_iso",
]
