"""Contrato de datos y cargador del ejemplo incluido en scikit-learn."""

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer

from modelops.tabular import DEMO_FEATURE_NAMES, DEMO_TARGET_NAMES


@dataclass(frozen=True)
class TabularDataset:
    """Tabla y objetivo supervisado, alineados por índice único.

    Otro cargador puede producir este contrato sin cambiar entrenamiento.
    Los nombres del esquema son explícitos, independientes de la tabla recibida.
    """

    name: str
    features: pd.DataFrame
    target: pd.Series
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    target_name: str = "target"
    task: str = "classification"
    categorical_features: tuple[str, ...] = ()
    allow_missing: bool = False
    source_options: dict | None = None


def load_demo_dataset() -> TabularDataset:
    """Cargar datos empaquetados, sin red ni interpretación clínica.

    Se conserva la codificación original: 0='malignant', 1='benign'.
    La clase positiva matemática será 1; no representa una recomendación clínica.
    """
    source = load_breast_cancer(as_frame=True)
    # El contrato expone strings de Python, no escalares numpy.str_.
    features = source.data.copy()
    features.columns = [str(name) for name in source.feature_names]
    return TabularDataset(
        name="sklearn.datasets.load_breast_cancer",
        features=features,
        target=source.target.copy(),
        feature_names=DEMO_FEATURE_NAMES,
        target_names=DEMO_TARGET_NAMES,
    )
