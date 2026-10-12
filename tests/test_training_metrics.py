"""Métricas contrastadas con casos calculables a mano y casos indefinidos."""

import numpy as np
import pytest

from modelops.training.evaluation import binary_metrics


def test_metrics_match_manual_confusion_matrix() -> None:
    # TN=1, FP=1, FN=1, TP=1; tres de cuatro pares positivos/negativos ordenados.
    metrics = binary_metrics(
        np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), np.array([0.1, 0.8, 0.4, 0.9])
    )
    assert metrics == pytest.approx(
        {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.75}
    )


def test_auc_uses_scores_instead_of_predictions() -> None:
    metrics = binary_metrics(np.array([0, 1]), np.array([0, 0]), np.array([0.1, 0.4]))
    assert metrics["accuracy"] == 0.5
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] is None
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_undefined_metrics_use_none() -> None:
    metrics = binary_metrics(np.array([0, 0]), np.array([0, 0]), np.array([0.1, 0.2]))
    assert metrics == {
        "accuracy": 1.0,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
    }


@pytest.mark.parametrize(
    ("actual", "predicted", "scores"),
    [
        ([], [], []),
        ([0, 1], [0], [0.2, 0.8]),
        ([0, 1], [0, 1], [0.2]),
        ([[0, 1]], [[0, 1]], [[0.2, 0.8]]),
        ([0, 2], [0, 1], [0.2, 0.8]),
        ([0, 1], [0, 2], [0.2, 0.8]),
        ([0, 1], [0, 1], [float("nan"), 0.8]),
        ([0, 1], [0, 1], [0.2, 1.1]),
        ([0, 1], [0, 1], ["bad", "bad"]),
    ],
)
def test_reject_invalid_metric_inputs(
    actual: list, predicted: list, scores: list
) -> None:
    with pytest.raises(ValueError):
        binary_metrics(np.array(actual), np.array(predicted), np.array(scores))
