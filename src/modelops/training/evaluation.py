"""Evaluación común; la clase positiva siempre es la etiqueta entera 1."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
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
    model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> dict[str, float | None]:
    classes = list(model.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("El modelo debe haberse entrenado con las clases 0 y 1.")
    return binary_metrics(
        target.to_numpy(),
        model.predict(features),
        model.predict_proba(features)[:, classes.index(1)],
    )
