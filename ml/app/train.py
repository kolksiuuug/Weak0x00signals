"""Обучение модели этапа 1 + метрики + важность признаков (CLAUDE.md §6, Участник 2).

Запуск:
    python -m ml.app.train                                   # автопоиск датасета в data/raw
    python -m ml.app.train --dataset data/official_dataset.xlsx   # явный путь
    python -m ml.app.train --min-precision 0.8               # требование ТЗ по точности

Что делает:
  1. Читает датасет организаторов (класс 1) и контрольную выборку зрелых/хайповых
     технологий (класс 0, см. data/control/README.md).
  2. Проверяет инсайт CLAUDE.md §1: Балл == стадия + тренд, и печатает долю совпадений.
  3. Обучает две модели — логистическую регрессию и градиентный бустинг — и выбирает
     по F1 на кросс-валидации. При сопоставимом качестве побеждает логистическая
     регрессия: интерпретируемость важнее долей процента (CLAUDE.md §2.2).
  4. Подбирает порог по out-of-fold предсказаниям на обучающей части, а не на hold-out
     (иначе порог подогнан под тестовую выборку и метрики завышены).
  5. Считает Precision / Recall / F1 на hold-out и на 5-фолдовой кросс-валидации.
  6. Выгружает важность признаков: коэффициенты линейной модели и permutation importance.
  7. Сохраняет артефакт ml/artifacts/model.joblib и отчёты ml/reports/.

Почему scikit-learn, а не нейросеть: 100 размеченных строк, требование интерпретируемости
и прямой запрет CLAUDE.md §2.2 / §7.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.app import dataset as ds
from ml.app import maturity
from ml.app.features import FEATURE_LABELS, FEATURE_NAMES, build_features, observation_from_row
from ml.app.model import ModelArtifact, save_artifact, utc_now_iso

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("ml.train")

#: Куда сохранить артефакт. Берём тот же `MODEL_PATH`, из которого артефакт ЧИТАЕТ
#: `ml.app.model.ARTIFACT_PATH`. Раньше здесь стоял литерал, и половины модуля
#: расходились: в Docker тренер писал внутрь образа, а сервис читал том — модели
#: не находил и молча уходил на запасные веса (`/health` отдавал `mock=true`).
DEFAULT_ARTIFACT = Path(os.getenv("MODEL_PATH", "ml/artifacts/model.joblib"))

#: Отрицательный класс (CLAUDE.md §0.2). Каталог или один .csv.
DEFAULT_CONTROL = Path(os.getenv("NEGATIVE_DATA_PATH", "data/control"))

#: Датасет организаторов. В Docker задаётся как /app/data/official_dataset.xlsx.
#: Переменная нужна для явного пути к официальному датасету в контейнере.
_DATASET_ENV = os.getenv("OFFICIAL_DATASET_PATH") or None
DEFAULT_DATASET = Path(_DATASET_ENV) if _DATASET_ENV else None

DEFAULT_REPORTS = Path(os.getenv("ML_REPORTS_PATH", "ml/reports"))

#: Насколько бустинг должен обогнать логистическую регрессию по F1, чтобы победить.
#: Меньший отрыв не окупает потерю интерпретируемости.
INTERPRETABILITY_MARGIN = 0.02


# --- данные -----------------------------------------------------------------

def build_matrix(frame: pd.DataFrame) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """DataFrame -> матрица признаков в порядке FEATURE_NAMES."""
    rows = []
    for _, row in frame.iterrows():
        rows.append(build_features(observation_from_row(row)))
    matrix = np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)
    return matrix, rows


def load_training_data(dataset_path: Optional[Path],
                       control_path: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    positives = ds.load_dataset(dataset_path)
    if positives.empty:
        raise SystemExit("Датасет организаторов пуст — обучать нечего.")

    if not control_path.exists():
        raise SystemExit(
            "Не найдена контрольная выборка отрицательного класса: {}. "
            "Без неё Precision/Recall/F1 не определены — см. data/control/README.md.".format(control_path)
        )
    negatives = ds.load_negatives(control_path)

    logger.info("Положительный класс (слабые сигналы): %d строк", len(positives))
    logger.info("Отрицательный класс (зрелые и хайповые): %d строк", len(negatives))

    x_pos, _ = build_matrix(positives)
    x_neg, _ = build_matrix(negatives)
    features = np.vstack([x_pos, x_neg])
    labels = np.hstack([np.ones(len(positives)), np.zeros(len(negatives))]).astype(int)
    return features, labels, positives, negatives


# --- диагностика признаков --------------------------------------------------

def log_feature_distribution(features: np.ndarray) -> None:
    logger.info("=== FEATURE DISTRIBUTION ===")
    for index, name in enumerate(FEATURE_NAMES):
        column = features[:, index]
        logger.info(
            "%s: min=%.3f max=%.3f mean=%.3f std=%.3f unique=%d",
            name, float(column.min()), float(column.max()), float(column.mean()),
            float(column.std()), int(len(np.unique(column))),
        )


def log_model_coefficients(pipeline: Pipeline) -> None:
    clf = pipeline.named_steps.get("clf")
    coefs = getattr(clf, "coef_", None)
    if coefs is None:
        return
    logger.info("=== MODEL COEFFICIENTS ===")
    for name, coef in zip(FEATURE_NAMES, coefs[0]):
        logger.info("%-24s %+0.6f", name, float(coef))
    logger.info("=== FEATURE IMPORTANCE (|coefficient|) ===")
    for name, coef in sorted(zip(FEATURE_NAMES, coefs[0]), key=lambda item: abs(item[1]), reverse=True):
        logger.info("%-24s %+0.6f", name, float(coef))


# --- модели -----------------------------------------------------------------

def candidate_models() -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    """Кандидаты и сетки гиперпараметров. Обе модели — из scikit-learn (CLAUDE.md §2.2)."""
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            penalty="l2",
            C=1.0,
        )),
    ])
    # Подбираем только силу L2-регуляризации. Выбор между L1/L2 запрещён
    # намеренно: L1 зануляет признаки и противоречит требованию использовать
    # весь инженерный вектор для интерпретируемого live-инференса.
    logreg_grid = {"clf__C": [0.1, 0.3, 1.0, 3.0, 10.0]}

    boosting = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=42)),
    ])
    boosting_grid = {
        "clf__n_estimators": [80, 150],
        "clf__max_depth": [2, 3],
        "clf__learning_rate": [0.05, 0.1],
    }
    return {"logistic_regression": (logreg, logreg_grid),
            "gradient_boosting": (boosting, boosting_grid)}


def select_model(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> Tuple[str, Pipeline, Dict[str, float]]:
    """Перебор кандидатов по F1 на стратифицированной 5-фолдовой кросс-валидации."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores: Dict[str, float] = {}
    fitted: Dict[str, Pipeline] = {}

    for name, (pipeline, grid) in candidate_models().items():
        search = GridSearchCV(pipeline, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(x_train, y_train)
        scores[name] = float(search.best_score_)
        fitted[name] = search.best_estimator_
        logger.info("%-20s F1(CV) = %.4f, параметры: %s", name, search.best_score_, search.best_params_)

    linear_score = scores.get("logistic_regression", 0.0)
    best_name = max(scores, key=lambda key: scores[key])
    if best_name != "logistic_regression" and scores[best_name] - linear_score < INTERPRETABILITY_MARGIN:
        logger.info(
            "Бустинг обогнал регрессию лишь на %.4f (< %.2f) — оставляю логистическую регрессию "
            "ради интерпретируемости (CLAUDE.md §2.2).",
            scores[best_name] - linear_score, INTERPRETABILITY_MARGIN,
        )
        best_name = "logistic_regression"

    return best_name, fitted[best_name], scores


# --- порог ------------------------------------------------------------------

def tune_threshold(pipeline: Pipeline, x_train: np.ndarray, y_train: np.ndarray,
                   seed: int, min_precision: float) -> Tuple[float, Dict[str, float]]:
    """Порог по out-of-fold предсказаниям на обучающей части.

    Требование ТЗ — точность не ниже 75-80 %. Поэтому среди порогов, дающих
    Precision >= min_precision, берём тот, что максимизирует F1. Если таких нет —
    берём максимум F1 и явно предупреждаем: молча занижать требование нельзя.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(pipeline, x_train, y_train, cv=cv, method="predict_proba")[:, 1]

    grid = np.round(np.arange(0.20, 0.91, 0.01), 2)
    rows = []
    for threshold in grid:
        predicted = (proba >= threshold).astype(int)
        if predicted.sum() == 0:
            continue
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_train, predicted, zero_division=0)),
            "recall": float(recall_score(y_train, predicted, zero_division=0)),
            "f1": float(f1_score(y_train, predicted, zero_division=0)),
        })

    eligible = [row for row in rows if row["precision"] >= min_precision]
    if eligible:
        best = max(eligible, key=lambda row: (row["f1"], row["recall"]))
    else:
        best = max(rows, key=lambda row: row["f1"])
        logger.warning(
            "Ни один порог не даёт Precision >= %.2f на кросс-валидации. Взят максимум F1 "
            "(precision=%.3f). Требование ТЗ по точности НЕ выполнено — нужны признаки или данные.",
            min_precision, best["precision"],
        )
    logger.info("Порог: %.2f (OOF precision=%.3f, recall=%.3f, F1=%.3f)",
                best["threshold"], best["precision"], best["recall"], best["f1"])
    return best["threshold"], best


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, Any]:
    predicted = (proba >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    true_neg, false_pos, false_neg, true_pos = matrix.ravel()
    metrics: Dict[str, Any] = {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "confusion_matrix": {
            "true_negative": int(true_neg), "false_positive": int(false_pos),
            "false_negative": int(false_neg), "true_positive": int(true_pos),
        },
    }
    if len(set(y_true.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        metrics["average_precision"] = float(average_precision_score(y_true, proba))
    return metrics


# --- важность признаков -----------------------------------------------------

def feature_importance(pipeline: Pipeline, x_test: np.ndarray, y_test: np.ndarray,
                       seed: int) -> Dict[str, Any]:
    """Коэффициенты линейной модели + permutation importance на hold-out."""
    out: Dict[str, Any] = {}

    clf = pipeline.named_steps.get("clf")
    coefs = getattr(clf, "coef_", None)
    if coefs is not None:
        out["coefficients"] = sorted(
            [
                {
                    "feature": name,
                    "label": FEATURE_LABELS.get(name, name),
                    "coef": float(coefs[0][index]),
                }
                for index, name in enumerate(FEATURE_NAMES)
            ],
            key=lambda item: abs(item["coef"]), reverse=True,
        )
        out["intercept"] = float(getattr(clf, "intercept_", [0.0])[0])
    elif hasattr(clf, "feature_importances_"):
        out["tree_importances"] = sorted(
            [
                {
                    "feature": name,
                    "label": FEATURE_LABELS.get(name, name),
                    "importance": float(clf.feature_importances_[index]),
                }
                for index, name in enumerate(FEATURE_NAMES)
            ],
            key=lambda item: item["importance"], reverse=True,
        )

    result = permutation_importance(pipeline, x_test, y_test, n_repeats=30,
                                    random_state=seed, scoring="f1")
    out["permutation"] = sorted(
        [
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "importance_mean": float(result.importances_mean[index]),
                "importance_std": float(result.importances_std[index]),
            }
            for index, name in enumerate(FEATURE_NAMES)
        ],
        key=lambda item: item["importance_mean"], reverse=True,
    )
    return out


# --- диагностика утечки -----------------------------------------------------

#: Одиночный признак с таким AUC разделяет классы почти идеально. На выборке, где
#: отрицательный класс собран руками, это почти всегда артефакт сборки, а не открытие.
LEAK_AUC_THRESHOLD = 0.95


def class_profile(features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Профиль признаков по классам + поиск признаков-«одиночек».

    Зачем. Отрицательный класс собран командой, а не выдан организаторами. Если при
    сборке в него попал систематический артефакт (например, у всех зрелых технологий
    ровно два источника), модель выучит артефакт, покажет отличный F1 и развалится на
    живых данных этапа 2. Проверка ловит это до защиты, а не после.
    """
    rows: List[Dict[str, Any]] = []
    suspicious: List[str] = []
    positive = labels == 1
    negative = labels == 0

    for index, name in enumerate(FEATURE_NAMES):
        column = features[:, index]
        auc = float(roc_auc_score(labels, column)) if len(set(column.tolist())) > 1 else 0.5
        separation = max(auc, 1.0 - auc)   # признак может разделять и «в обратную сторону»
        rows.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "mean_positive": float(column[positive].mean()),
            "mean_negative": float(column[negative].mean()),
            "std_positive": float(column[positive].std()),
            "std_negative": float(column[negative].std()),
            "single_feature_auc": separation,
        })
        if separation >= LEAK_AUC_THRESHOLD:
            suspicious.append(name)
            logger.warning(
                "Признак «%s» в одиночку разделяет классы с AUC=%.3f. Похоже на артефакт "
                "сборки отрицательного класса, а не на содержательный признак — проверьте "
                "data/control/mature_control.csv.", FEATURE_LABELS.get(name, name), separation,
            )

    rows.sort(key=lambda item: item["single_feature_auc"], reverse=True)
    return {"features": rows, "suspicious": suspicious, "auc_threshold": LEAK_AUC_THRESHOLD}


# --- абляция: сколько качества держится на словарях -------------------------

#: Признаки, которые считаются по словарям ml/app/lexicon.py.
LEXICON_FEATURES = ("weakness_markers", "maturity_markers", "hype_markers")


def ablation(features: np.ndarray, labels: np.ndarray, threshold: float, seed: int) -> Dict[str, Any]:
    """Переобучение без словарных признаков.

    Зачем. Негативный класс написан нами, и в обосновании каждой зрелой технологии
    прямо стоят слова «массовое внедрение», «отраслевой стандарт». Модель может выучить
    не свойства технологии, а наш стиль письма — и показать отличный F1, который на
    живых документах этапа 2 не воспроизведётся.

    Проверка простая: убираем словарные признаки и смотрим, что осталось. Если F1
    рушится — качество держалось на словаре, и метрику нельзя предъявлять без оговорки.
    """
    keep = [index for index, name in enumerate(FEATURE_NAMES) if name not in LEXICON_FEATURES]
    reduced = features[:, keep]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", penalty="l2", C=1.0)),
    ])
    proba = cross_val_predict(pipeline, reduced, labels, cv=cv, method="predict_proba")[:, 1]
    without = evaluate(labels, proba, threshold)

    full_proba = cross_val_predict(
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", penalty="l2", C=1.0)),
        ]),
        features, labels, cv=cv, method="predict_proba",
    )[:, 1]
    full = evaluate(labels, full_proba, threshold)

    drop = full["f1"] - without["f1"]
    logger.info("Абляция без словарей: F1 %.3f -> %.3f (падение %.3f)",
                full["f1"], without["f1"], drop)
    if drop > 0.15:
        logger.warning(
            "Больше 15 %% F1 держится на словарных признаках. Негативный класс написан нами, "
            "поэтому слова про зрелость в нём встречаются по построению — метрику нельзя "
            "предъявлять без этой оговорки. Ориентируйтесь на оценку открытого запроса: "
            "python -m ml.app.evaluate"
        )
    return {
        "features_removed": list(LEXICON_FEATURES),
        "f1_full": full["f1"],
        "f1_without_lexicon": without["f1"],
        "f1_drop": round(drop, 4),
        "precision_without_lexicon": without["precision"],
        "recall_without_lexicon": without["recall"],
    }


# --- диагностика фильтра зрелости ------------------------------------------

def maturity_diagnostics(positives: pd.DataFrame, negatives: pd.DataFrame) -> Dict[str, Any]:
    """Насколько фильтр зрелости согласован с разметкой.

    Фильтр работает ДО и ПОСЛЕ модели, поэтому его ошибки важны отдельно:
      * сколько настоящих слабых сигналов он отклоняет (это потери выдачи);
      * сколько зрелых и хайповых технологий он ловит сам, без модели.
    """
    def _run(frame: pd.DataFrame) -> Tuple[int, Dict[str, int], List[str]]:
        rejected = 0
        by_rule: Dict[str, int] = {}
        examples: List[str] = []
        for _, row in frame.iterrows():
            verdict = maturity.check(observation_from_row(row))
            if verdict.rejected:
                rejected += 1
                by_rule[verdict.rule or "?"] = by_rule.get(verdict.rule or "?", 0) + 1
                if len(examples) < 5:
                    examples.append("{} -> {}".format(row["technology"], verdict.rule))
        return rejected, by_rule, examples

    pos_rejected, pos_rules, pos_examples = _run(positives)
    neg_rejected, neg_rules, _ = _run(negatives)
    return {
        "positives_total": int(len(positives)),
        "positives_rejected_by_filter": pos_rejected,
        "positives_rejection_rate": round(pos_rejected / len(positives), 4) if len(positives) else 0.0,
        "positives_rules": pos_rules,
        "positives_examples": pos_examples,
        "negatives_total": int(len(negatives)),
        "negatives_caught_by_filter": neg_rejected,
        "negatives_catch_rate": round(neg_rejected / len(negatives), 4) if len(negatives) else 0.0,
        "negatives_rules": neg_rules,
    }


# --- отчёты -----------------------------------------------------------------

def _fmt_pct(value: Optional[float]) -> str:
    return "—" if value is None else "{:.1f} %".format(value * 100)


def write_reports(report: Dict[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    holdout = report["metrics"]["holdout"]
    cv_metrics = report["metrics"]["cross_validation"]
    insight = report["insight"]
    diag = report["maturity_filter"]

    lines: List[str] = [
        "# Отчёт по метрикам модели этапа 1",
        "",
        "Сгенерировано автоматически: `python -m ml.app.train`. Дата: {}.".format(report["trained_at"]),
        "",
        "## 1. Данные",
        "",
        "| Класс | Источник | Строк |",
        "|---|---|---|",
        "| 1 — слабый сигнал | датасет организаторов | {} |".format(report["n_positive"]),
        "| 0 — зрелая технология или хайп | `data/control/` | {} |".format(report["n_negative"]),
        "",
        "Отрицательный класс собран командой — этого прямо требует CLAUDE.md §0.2: скрытого "
        "датасета нет, в выданном файле только положительные примеры. Метрики самодекларируемые "
        "(§0.4). Состав и правила ведения — `data/control/README.md`.",
        "",
        "Происхождение отрицательных примеров:",
        "",
        "| Файл | Строк |",
        "|---|---|",
    ]
    for file_name, count in report["negative_sources"].items():
        lines.append("| `{}` | {} |".format(file_name, count))
    lines += [
        "",
        "## 2. Проверка инсайта «Балл = стадия + тренд» (CLAUDE.md §1)",
        "",
        "Строк с проставленным баллом: **{}**. Формула воспроизводится на **{}** наблюдений.".format(
            insight["rows_with_points"], _fmt_pct(insight["match_rate"]),
        ),
        "",
    ]
    if insight["mismatches"]:
        lines += [
            "Расхождения ({} шт.) — их стоит проверить вручную:".format(len(insight["mismatches"])),
            "",
            "| Технология | Стадия | Тренд | Ожидаемый балл | Балл в файле |",
            "|---|---|---|---|---|",
        ]
        for item in insight["mismatches"][:15]:
            lines.append("| {} | {} ({}) | {} ({}) | {} | {} |".format(
                item["technology"], item["stage_raw"], item["stage"],
                item["trend_raw"], item["trend"], item["points_expected"], item["points_actual"],
            ))
        lines.append("")
    else:
        lines += ["Расхождений нет: формула воспроизводится полностью.", ""]

    lines += [
        "## 3. Модель",
        "",
        "Выбрана: **{}**. Порог отнесения к слабому сигналу: **{:.2f}** "
        "(подобран по out-of-fold предсказаниям на обучающей части, не на hold-out).".format(
            report["model_kind"], report["threshold"],
        ),
        "",
        "F1 на кросс-валидации по кандидатам:",
        "",
        "| Модель | F1 (CV) |",
        "|---|---|",
    ]
    for name, score in report["candidate_scores"].items():
        lines.append("| {} | {:.4f} |".format(name, score))
    lines += [
        "",
        "## 4. Метрики",
        "",
        "| Метрика | Hold-out ({} набл.) | 5-фолдовая CV (все данные) |".format(holdout["n"]),
        "|---|---|---|",
        "| Precision | {} | {} |".format(_fmt_pct(holdout["precision"]), _fmt_pct(cv_metrics["precision"])),
        "| Recall | {} | {} |".format(_fmt_pct(holdout["recall"]), _fmt_pct(cv_metrics["recall"])),
        "| F1 | {} | {} |".format(_fmt_pct(holdout["f1"]), _fmt_pct(cv_metrics["f1"])),
        "| Accuracy | {} | {} |".format(_fmt_pct(holdout["accuracy"]), _fmt_pct(cv_metrics["accuracy"])),
        "| ROC-AUC | {} | {} |".format(_fmt_pct(holdout.get("roc_auc")), _fmt_pct(cv_metrics.get("roc_auc"))),
        "",
        "Матрица ошибок на hold-out: истинно-положительных {}, ложно-положительных {}, "
        "ложно-отрицательных {}, истинно-отрицательных {}.".format(
            holdout["confusion_matrix"]["true_positive"],
            holdout["confusion_matrix"]["false_positive"],
            holdout["confusion_matrix"]["false_negative"],
            holdout["confusion_matrix"]["true_negative"],
        ),
        "",
        "Требование ТЗ — точность не ниже 75-80 %. Достигнутая точность на hold-out: **{}** — {}.".format(
            _fmt_pct(holdout["precision"]),
            "требование выполнено" if holdout["precision"] >= report["min_precision"] else "ТРЕБОВАНИЕ НЕ ВЫПОЛНЕНО",
        ),
        "",
        "## 5. Важность признаков",
        "",
    ]

    importance = report["feature_importance"]
    if "coefficients" in importance:
        lines += [
            "Коэффициенты логистической регрессии на стандартизованных признаках. "
            "Знак читается напрямую: плюс — признак повышает вероятность слабого сигнала.",
            "",
            "| Признак | Коэффициент |",
            "|---|---|",
        ]
        for item in importance["coefficients"]:
            lines.append("| {} | {:+.3f} |".format(item["label"], item["coef"]))
        lines.append("")
    if "tree_importances" in importance:
        lines += ["| Признак | Важность (дерево) |", "|---|---|"]
        for item in importance["tree_importances"]:
            lines.append("| {} | {:.3f} |".format(item["label"], item["importance"]))
        lines.append("")

    lines += [
        "Permutation importance на hold-out (падение F1 при перемешивании признака):",
        "",
        "| Признак | Падение F1 | σ |",
        "|---|---|---|",
    ]
    for item in importance["permutation"][:10]:
        lines.append("| {} | {:+.4f} | {:.4f} |".format(
            item["label"], item["importance_mean"], item["importance_std"]))

    profile = report["class_profile"]
    lines += [
        "",
        "## 6. Проверка на артефакты выборки",
        "",
        "Отрицательный класс собран командой, поэтому отдельно проверяется, не разделяет ли "
        "классы один-единственный признак: это почти всегда артефакт сборки, а не открытие. "
        "Порог подозрительности по AUC одиночного признака — {:.2f}.".format(profile["auc_threshold"]),
        "",
    ]
    if profile["suspicious"]:
        lines += [
            "**Подозрительные признаки: {}.** Их стоит проверить в "
            "`data/control/mature_control.csv` до защиты.".format(
                ", ".join(FEATURE_LABELS.get(name, name) for name in profile["suspicious"])),
            "",
        ]
    else:
        lines += ["Признаков-«одиночек» не обнаружено.", ""]

    lines += [
        "| Признак | Среднее (слабые сигналы) | Среднее (зрелые и хайп) | AUC в одиночку |",
        "|---|---|---|---|",
    ]
    for item in profile["features"]:
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} |".format(
            item["label"], item["mean_positive"], item["mean_negative"], item["single_feature_auc"]))

    abl = report["ablation"]
    lines += [
        "",
        "### Абляция: сколько качества держится на словарях",
        "",
        "Негативный класс написан нами, и в обосновании каждой зрелой технологии по "
        "построению стоят слова «массовое внедрение», «отраслевой стандарт». Поэтому "
        "отдельно считается, что останется от модели без словарных признаков ({}).".format(
            ", ".join(FEATURE_LABELS.get(f, f) for f in abl["features_removed"])),
        "",
        "| Набор признаков | F1 (кросс-валидация) |",
        "|---|---|",
        "| Все признаки | {} |".format(_fmt_pct(abl["f1_full"])),
        "| Без словарных признаков | {} |".format(_fmt_pct(abl["f1_without_lexicon"])),
        "",
        "Падение F1: **{}**. {}".format(
            _fmt_pct(abl["f1_drop"]),
            "Существенная часть качества держится на словаре — метрику нельзя предъявлять "
            "без этой оговорки, ориентируйтесь на оценку открытого запроса "
            "(`python -m ml.app.evaluate`)."
            if abl["f1_drop"] > 0.15 else
            "Модель опирается не только на словарь — признаки фактуры и шкалы работают сами.",
        ),
        "",
        "## 7. Диагностика фильтра зрелости",
        "",
        "Фильтр (CLAUDE.md §2.10) работает независимо от модели, поэтому его ошибки считаются отдельно.",
        "",
        "* Слабых сигналов ошибочно отклонено фильтром: **{} из {}** ({}).".format(
            diag["positives_rejected_by_filter"], diag["positives_total"],
            _fmt_pct(diag["positives_rejection_rate"]),
        ),
        "* Зрелых и хайповых технологий фильтр ловит сам, без модели: **{} из {}** ({}).".format(
            diag["negatives_caught_by_filter"], diag["negatives_total"],
            _fmt_pct(diag["negatives_catch_rate"]),
        ),
        "",
        "Сработавшие правила на отрицательном классе: {}.".format(
            ", ".join("{} — {}".format(k, v) for k, v in diag["negatives_rules"].items()) or "—"),
        "",
    ]
    if diag["positives_examples"]:
        lines += [
            "Слабые сигналы, отклонённые фильтром (проверить вручную — это прямые потери выдачи):",
            "",
        ] + ["* {}".format(item) for item in diag["positives_examples"]] + [""]

    (reports_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Отчёты записаны: %s", reports_dir)


# --- точка входа ------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Обучение модели этапа 1 (слабые сигналы)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Путь к датасету организаторов. По умолчанию — OFFICIAL_DATASET_PATH, "
                             "иначе используется /app/data/official_dataset.xlsx")
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL,
                        help="Отрицательный класс: файл .csv или каталог с несколькими (CLAUDE.md §0.2)")
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT, help="Куда сохранить артефакт")
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS, help="Каталог отчётов")
    parser.add_argument("--test-size", type=float, default=0.25, help="Доля hold-out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-precision", type=float, default=0.80,
                        help="Требуемая точность (CLAUDE.md §1: 75-80 %%)")
    args = parser.parse_args(argv)

    features, labels, positives, negatives = load_training_data(args.dataset, args.control)
    log_feature_distribution(features)

    insight = ds.insight_report(positives)
    if insight["match_rate"] is not None:
        logger.info("Инсайт §1 «Балл = стадия + тренд» воспроизводится на %.1f %% строк",
                    insight["match_rate"] * 100)
        if insight["mismatches"]:
            logger.warning("Расхождений: %d (см. отчёт)", len(insight["mismatches"]))

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=args.test_size, random_state=args.seed, stratify=labels,
    )
    logger.info("Обучающая часть: %d, hold-out: %d", len(y_train), len(y_test))

    model_kind, pipeline, candidate_scores = select_model(x_train, y_train, args.seed)
    threshold, oof = tune_threshold(pipeline, x_train, y_train, args.seed, args.min_precision)

    pipeline.fit(x_train, y_train)
    log_model_coefficients(pipeline)
    holdout = evaluate(y_test, pipeline.predict_proba(x_test)[:, 1], threshold)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_proba = cross_val_predict(pipeline, features, labels, cv=cv, method="predict_proba")[:, 1]
    cv_metrics = evaluate(labels, cv_proba, threshold)

    importance = feature_importance(pipeline, x_test, y_test, args.seed)
    profile = class_profile(features, labels)
    ablation_report = ablation(features, labels, threshold, args.seed)
    diagnostics = maturity_diagnostics(positives, negatives)

    logger.info("HOLD-OUT: precision=%.3f recall=%.3f F1=%.3f",
                holdout["precision"], holdout["recall"], holdout["f1"])
    logger.info("CV(5):    precision=%.3f recall=%.3f F1=%.3f",
                cv_metrics["precision"], cv_metrics["recall"], cv_metrics["f1"])

    report: Dict[str, Any] = {
        "trained_at": utc_now_iso(),
        "model_kind": model_kind,
        "threshold": threshold,
        "min_precision": args.min_precision,
        "seed": args.seed,
        "n_train": int(len(y_train)),
        "n_holdout": int(len(y_test)),
        "n_positive": int(len(positives)),
        "n_negative": int(len(negatives)),
        "negative_sources": (
            {str(k): int(v) for k, v in negatives["source_file"].value_counts().items()}
            if "source_file" in negatives.columns else {}
        ),
        "features": list(FEATURE_NAMES),
        "candidate_scores": candidate_scores,
        "insight": insight,
        "metrics": {"holdout": holdout, "cross_validation": cv_metrics, "threshold_oof": oof},
        "feature_importance": importance,
        "class_profile": profile,
        "ablation": ablation_report,
        "maturity_filter": diagnostics,
    }

    # финальная модель обучается на всех данных: hold-out уже сыграл свою роль
    pipeline.fit(features, labels)
    save_artifact(
        ModelArtifact(
            pipeline=pipeline,
            feature_names=FEATURE_NAMES,
            threshold=threshold,
            metrics={"holdout": holdout, "cross_validation": cv_metrics},
            model_kind=model_kind,
            trained_at=report["trained_at"],
            n_train=int(len(labels)),
            n_positive=int(len(positives)),
            n_negative=int(len(negatives)),
            notes="Отрицательный класс — data/control/mature_control.csv (см. README там же).",
        ),
        args.out,
    )
    write_reports(report, args.reports)

    if holdout["precision"] < args.min_precision:
        logger.warning("Точность на hold-out ниже требуемой (%.3f < %.2f).",
                       holdout["precision"], args.min_precision)
        return 1
    return 0


def _cli() -> int:
    """Точка входа с человеческим сообщением вместо трейсбека.

    Самый вероятный сценарий первого запуска — датасета организаторов нет на месте
    (он в .gitignore, в репозиторий не коммитится). Показывать в этом случае
    двадцать строк стека — значит заставлять читателя искать в них единственную
    содержательную строку. Текст ошибки уже написан по-русски и говорит, что делать.
    """
    try:
        return main()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
