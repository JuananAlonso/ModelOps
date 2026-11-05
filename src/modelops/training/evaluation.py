"""Métricas por tarea; en clasificación binaria la clase positiva es 1."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, positive_scores: np.ndarray
) -> dict[str, float | None]:
    """Retornar None cuando un denominador sea cero o AUC no esté definida."""
    actual, predicted, scores = map(np.asarray, (y_true, y_pred, positive_scores))
    if (
        actual.ndim != 1
        or predicted.shape != actual.shape
        or scores.shape != actual.shape
        or not actual.size
    ):
        raise ValueError("Las etiquetas y probabilidades deben ser vectores alineados.")
    if not np.isin(actual, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Las etiquetas de evaluación deben ser 0 o 1.")
    try:
        scores = scores.astype(float)
    except (ValueError, TypeError) as error:
        raise ValueError("Las probabilidades deben ser numéricas.") from error
    if not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError("Las probabilidades deben ser finitas y estar entre 0 y 1.")
    actual_positives = int((actual == 1).sum())
    predicted_positives = int((predicted == 1).sum())
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": (
            float(precision_score(actual, predicted, pos_label=1))
            if predicted_positives
            else None
        ),
        "recall": (
            float(recall_score(actual, predicted, pos_label=1))
            if actual_positives
            else None
        ),
        "f1": (
            float(f1_score(actual, predicted, pos_label=1))
            if actual_positives + predicted_positives
            else None
        ),
        "roc_auc": (
            float(roc_auc_score(actual, scores))
            if np.unique(actual).size == 2
            else None
        ),
    }


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    task: str = "classification",
) -> dict[str, float | None]:
    if task == "regression":
        predicted = model.predict(features)
        r2 = float(r2_score(target, predicted))
        return {
            "mae": float(mean_absolute_error(target, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(target, predicted))),
            "r2": r2 if np.isfinite(r2) else None,
        }
    classes = list(model.classes_)
    if len(classes) > 2:
        predicted = model.predict(features)
        return {
            "accuracy": float(accuracy_score(target, predicted)),
            "precision_macro": float(
                precision_score(target, predicted, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(target, predicted, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(target, predicted, average="macro", zero_division=0)
            ),
            "roc_auc_ovr_macro": float(
                roc_auc_score(
                    target,
                    model.predict_proba(features),
                    labels=classes,
                    multi_class="ovr",
                    average="macro",
                )
            ),
        }
    if set(classes) != {0, 1}:
        raise ValueError("El modelo debe haberse entrenado con las clases 0 y 1.")
    return binary_metrics(
        target.to_numpy(),
        model.predict(features),
        model.predict_proba(features)[:, classes.index(1)],
    )
