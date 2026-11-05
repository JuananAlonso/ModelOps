"""Partición común; estratificada en clasificación, aleatoria en regresión."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modelops.training.config import TrainingConfig
from modelops.training.data import TabularDataset
from modelops.training.validation import validate_dataset


@dataclass(frozen=True)
class DataSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_positions: np.ndarray
    test_positions: np.ndarray


def split_dataset(dataset: TabularDataset, config: TrainingConfig) -> DataSplit:
    validate_dataset(dataset)
    test_count = math.ceil(len(dataset.features) * config.test_size)
    if min(test_count, len(dataset.features) - test_count) < 2:
        raise ValueError("test_size debe dejar al menos dos filas en cada partición.")
    if dataset.task == "classification" and min(
        test_count, len(dataset.features) - test_count
    ) < len(dataset.target_names):
        raise ValueError(
            "test_size debe dejar al menos una fila por clase en cada partición; "
            "ajusta --test-size o añade datos."
        )
    train_positions, test_positions = train_test_split(
        np.arange(len(dataset.features)),
        test_size=config.test_size,
        random_state=config.seed,
        stratify=dataset.target if dataset.task == "classification" else None,
    )
    y_train = dataset.target.iloc[train_positions].copy()
    y_test = dataset.target.iloc[test_positions].copy()
    if dataset.task == "classification" and (
        set(y_train.unique()) != set(dataset.target.unique())
        or set(y_test.unique()) != set(dataset.target.unique())
    ):
        raise ValueError(
            "La partición no conserva ambas clases o todas las clases multiclase; "
            "cambia test_size o añade datos."
        )
    return DataSplit(
        x_train=dataset.features.iloc[train_positions].copy(),
        x_test=dataset.features.iloc[test_positions].copy(),
        y_train=y_train,
        y_test=y_test,
        train_positions=train_positions,
        test_positions=test_positions,
    )
