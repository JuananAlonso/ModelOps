"""Contrato de datos y cargador del ejemplo incluido en scikit-learn."""

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer


@dataclass(frozen=True)
class TabularDataset:
    """Tabla numérica y objetivo entero {0, 1}, alineados por índice único.

    Otro cargador puede producir este contrato sin cambiar entrenamiento.
    Los nombres del esquema son explícitos, independientes de la tabla recibida.
    """

    name: str
    features: pd.DataFrame
    target: pd.Series
    feature_names: tuple[str, ...]
    target_names: tuple[str, str]
    target_name: str = "target"


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
        feature_names=tuple(str(name) for name in source.feature_names),
        target_names=tuple(str(name) for name in source.target_names),
    )
