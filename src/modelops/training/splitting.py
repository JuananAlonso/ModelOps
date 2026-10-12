"""Una única partición estratificada compartida por todos los modelos."""

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
    train_positions, test_positions = train_test_split(
        np.arange(len(dataset.features)),
        test_size=config.test_size,
        random_state=config.seed,
        stratify=dataset.target,
    )
    y_train = dataset.target.iloc[train_positions].copy()
    y_test = dataset.target.iloc[test_positions].copy()
    if set(y_train.unique()) != {0, 1} or set(y_test.unique()) != {0, 1}:
        raise ValueError(
            "La partición no conserva ambas clases; cambia test_size o añade datos."
        )
    return DataSplit(
        x_train=dataset.features.iloc[train_positions].copy(),
        x_test=dataset.features.iloc[test_positions].copy(),
        y_train=y_train,
        y_test=y_test,
        train_positions=train_positions,
        test_positions=test_positions,
    )
