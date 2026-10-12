"""Validación estructural sin estimar estadísticas ni transformar los datos."""

import hashlib
import json

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

from modelops.training.data import TabularDataset


def validate_dataset(dataset: TabularDataset) -> None:
    """Rechazar esquemas ambiguos, datos no finitos y objetivos no binarios."""
    if not isinstance(dataset, TabularDataset):
        raise ValueError("Se esperaba un TabularDataset.")
    if not isinstance(dataset.name, str) or not dataset.name.strip():
        raise ValueError("El dataset debe tener un identificador no vacío.")
    names = dataset.feature_names
    if (
        not isinstance(names, tuple)
        or not names
        or any(type(name) is not str or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("El esquema requiere nombres únicos de columnas de tipo str.")
    if (
        not isinstance(dataset.target_name, str)
        or not dataset.target_name.strip()
        or dataset.target_name in names
    ):
        raise ValueError("El nombre del objetivo debe ser distinto de las features.")
    if (
        not isinstance(dataset.target_names, tuple)
        or len(dataset.target_names) != 2
        or any(
            not isinstance(name, str) or not name.strip()
            for name in dataset.target_names
        )
        or len(set(dataset.target_names)) != 2
    ):
        raise ValueError("Se necesitan dos nombres de clase distintos para 0 y 1.")
    features, target = dataset.features, dataset.target
    if not isinstance(features, pd.DataFrame) or not isinstance(target, pd.Series):
        raise ValueError("Features debe ser DataFrame y target debe ser Series.")
    if tuple(features.columns) != names:
        raise ValueError("Las columnas y su orden deben coincidir con el esquema.")
    if target.name != dataset.target_name:
        raise ValueError("El nombre del target no coincide con el esquema.")
    if len(features) < 4:
        raise ValueError("Se necesitan al menos cuatro filas para la división binaria.")
    if not features.index.is_unique or not target.index.is_unique:
        raise ValueError("Los índices de las filas deben ser únicos.")
    if not features.index.equals(target.index):
        raise ValueError(
            "Features y target deben tener índices alineados y igual tamaño."
        )
    if features.isna().any().any() or target.isna().any():
        raise ValueError("No se admiten valores ausentes en features ni en target.")
    for name, dtype in features.dtypes.items():
        if dtype.kind not in "iuf":
            raise ValueError(f"La columna {name!r} debe ser numérica real, no {dtype}.")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("Las features deben ser finitas; no se admiten infinitos.")
    if not is_integer_dtype(target.dtype) or is_bool_dtype(target.dtype):
        raise ValueError("El target debe tener tipo entero con etiquetas 0 y 1.")
    if set(target.unique()) != {0, 1}:
        raise ValueError("El target debe contener exactamente las clases 0 y 1.")
    if target.value_counts().min() < 2:
        raise ValueError("Cada clase necesita al menos dos filas para estratificar.")


def dataset_fingerprint(dataset: TabularDataset) -> str:
    """SHA-256 de identidad, esquema, tipos, valores, índices y orden de filas.

    La versión de pandas se registra: su hash de filas es parte del algoritmo.
    """
    validate_dataset(dataset)
    schema = {
        "name": dataset.name,
        "feature_names": dataset.feature_names,
        "feature_dtypes": [str(dtype) for dtype in dataset.features.dtypes],
        "target_name": dataset.target_name,
        "target_dtype": str(dataset.target.dtype),
        "target_names": dataset.target_names,
    }
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8"))
    for table in (dataset.features, dataset.target):
        row_hashes = pd.util.hash_pandas_object(table, index=True)
        digest.update(row_hashes.to_numpy(dtype="<u8").tobytes())
    return digest.hexdigest()
