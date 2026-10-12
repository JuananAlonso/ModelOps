"""División reproducible, estratificación y validación de configuración."""

import numpy as np
import pandas as pd
import pytest

from modelops.training.config import TrainingConfig
from modelops.training.data import TabularDataset
from modelops.training.splitting import split_dataset


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True, "42"])
def test_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="semilla"):
        TrainingConfig(seed=seed)


@pytest.mark.parametrize(
    "test_size", [0, 1, -0.2, float("nan"), float("inf"), True, "x"]
)
def test_invalid_fraction(test_size: object) -> None:
    with pytest.raises(ValueError, match="test_size"):
        TrainingConfig(test_size=test_size)


def test_split_is_disjoint_stratified_and_reproducible(
    demo_dataset: TabularDataset,
) -> None:
    first = split_dataset(demo_dataset, TrainingConfig(seed=17))
    second = split_dataset(demo_dataset, TrainingConfig(seed=17))
    different = split_dataset(demo_dataset, TrainingConfig(seed=18))
    np.testing.assert_array_equal(first.train_positions, second.train_positions)
    np.testing.assert_array_equal(first.test_positions, second.test_positions)
    assert not np.array_equal(first.test_positions, different.test_positions)
    train, test = set(first.train_positions), set(first.test_positions)
    assert not train & test
    assert train | test == set(range(len(demo_dataset.features)))
    assert first.x_train.index.equals(first.y_train.index)
    assert first.x_test.index.equals(first.y_test.index)
    assert len(first.x_test) == 114
    for subset in (first.y_train, first.y_test):
        assert set(subset.unique()) == {0, 1}
        assert subset.mean() == pytest.approx(demo_dataset.target.mean(), abs=0.01)


@pytest.mark.parametrize("test_size", [0.001, 0.999])
def test_split_requires_two_classes_per_partition(
    demo_dataset: TabularDataset, test_size: float
) -> None:
    with pytest.raises(ValueError, match="dos filas"):
        split_dataset(demo_dataset, TrainingConfig(test_size=test_size))


def test_stratification_rejects_a_partition_without_minority_class() -> None:
    dataset = TabularDataset(
        name="imbalanced-toy",
        features=pd.DataFrame({"x": np.arange(10, dtype=float)}),
        target=pd.Series([0] * 8 + [1] * 2, name="target"),
        feature_names=("x",),
        target_names=("negative", "positive"),
    )
    with pytest.raises(ValueError, match="ambas clases"):
        split_dataset(dataset, TrainingConfig(test_size=0.2))
