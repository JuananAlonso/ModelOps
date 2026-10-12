"""Ambos modelos, reproducibilidad y pruebas causales de ausencia de leakage."""

from dataclasses import replace

import numpy as np
import pytest

from modelops.training.config import TrainingConfig
from modelops.training.data import TabularDataset
from modelops.training.runner import TrainingResult, train_dataset


@pytest.mark.parametrize("name", ["baseline", "lightgbm"])
def test_both_models_are_fitted_and_evaluated(
    trained_result: TrainingResult, name: str
) -> None:
    model = trained_result.models[name]
    assert set(model.classes_) == {0, 1}
    assert model.predict(trained_result.split.x_test).shape == (114,)
    assert model.predict_proba(trained_result.split.x_test).shape == (114, 2)
    metrics = trained_result.metrics[name]
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(value is not None and 0 <= value <= 1 for value in metrics.values())


def test_scaler_learns_training_statistics_only(trained_result: TrainingResult) -> None:
    scaler = (
        trained_result.models["baseline"]
        .named_steps["preprocess"]
        .named_transformers_["numeric"]
    )
    np.testing.assert_allclose(
        scaler.mean_, trained_result.split.x_train.mean().to_numpy(), rtol=1e-12
    )
    assert scaler.n_samples_seen_ == len(trained_result.split.x_train)
    assert not np.allclose(scaler.mean_, trained_result.dataset.features.mean())


def test_changing_test_features_cannot_change_fitted_models(
    trained_result: TrainingResult,
) -> None:
    features = trained_result.dataset.features.copy()
    features.iloc[trained_result.split.test_positions] += 1_000_000
    perturbed = replace(trained_result.dataset, features=features)
    result = train_dataset(perturbed, trained_result.config)
    np.testing.assert_array_equal(
        trained_result.split.train_positions, result.split.train_positions
    )
    original = trained_result.models["baseline"].named_steps["model"]
    changed = result.models["baseline"].named_steps["model"]
    np.testing.assert_allclose(original.coef_, changed.coef_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(original.intercept_, changed.intercept_, atol=1e-12)
    assert (
        trained_result.models["lightgbm"]
        .named_steps["model"]
        .booster_.model_to_string()
        == result.models["lightgbm"].named_steps["model"].booster_.model_to_string()
    )


def test_repeated_training_reproduces_predictions_and_metrics(
    demo_dataset: TabularDataset, trained_result: TrainingResult
) -> None:
    repeated = train_dataset(demo_dataset, TrainingConfig())
    assert repeated.dataset_sha256 == trained_result.dataset_sha256
    for name, model in repeated.models.items():
        assert repeated.metrics[name] == pytest.approx(
            trained_result.metrics[name], rel=1e-9, abs=1e-12
        )
        np.testing.assert_allclose(
            model.predict_proba(repeated.split.x_test),
            trained_result.models[name].predict_proba(trained_result.split.x_test),
            rtol=1e-9,
            atol=1e-12,
        )
