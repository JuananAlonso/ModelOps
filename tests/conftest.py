"""Fixtures pequeñas y protección contra acceso de red durante los tests."""

import os
import socket
from collections.abc import Iterator

import pytest

from modelops.training.config import TrainingConfig
from modelops.training.data import TabularDataset, load_demo_dataset
from modelops.training.runner import TrainingResult, train_dataset

# Antes de la colección/importación de los tests que usan MLflow.
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"


@pytest.fixture(scope="session", autouse=True)
def forbid_network() -> Iterator[None]:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("Los tests de ModelOps no deben abrir conexiones de red.")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", reject_connection)
        patch.setattr(socket.socket, "connect_ex", reject_connection)
        yield


@pytest.fixture(scope="session")
def demo_dataset() -> TabularDataset:
    return load_demo_dataset()


@pytest.fixture(scope="session")
def trained_result(demo_dataset: TabularDataset) -> TrainingResult:
    return train_dataset(demo_dataset, TrainingConfig())
