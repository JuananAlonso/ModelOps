"""Contrato tabular, errores útiles e identidad del dataset."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from modelops.training.data import TabularDataset, load_demo_dataset
from modelops.training.validation import dataset_fingerprint, validate_dataset


def test_demo_loader() -> None:
    dataset = load_demo_dataset()
    validate_dataset(dataset)
    assert dataset.features.shape == (569, 30)
    assert dataset.target_names == ("malignant", "benign")
    assert all(type(name) is str for name in dataset.feature_names)
    assert set(dataset.target.unique()) == {0, 1}
    assert dataset.name == "sklearn.datasets.load_breast_cancer"


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_column", "columnas"),
        ("extra_column", "columnas"),
        ("reordered_columns", "orden"),
        ("duplicate_columns", "columnas"),
        ("string_feature", "numérica"),
        ("bool_feature", "numérica"),
        ("complex_feature", "numérica"),
        ("missing_feature", "ausentes"),
        ("missing_target", "ausentes"),
        ("infinite_feature", "finitas"),
        ("float_target", "entero"),
        ("bool_target", "entero"),
        ("third_class", "clases 0 y 1"),
        ("single_class", "clases 0 y 1"),
        ("rare_class", "dos filas"),
        ("misaligned_target", "alineados"),
        ("duplicate_index", "únicos"),
        ("empty_table", "cuatro filas"),
        ("wrong_target_name", "nombre del target"),
        ("empty_id", "identificador"),
        ("empty_schema", "esquema"),
        ("duplicate_schema", "esquema"),
        ("numpy_schema", "esquema"),
        ("target_in_features", "distinto"),
        ("invalid_class_names", "nombres de clase"),
        ("wrong_features_type", "DataFrame"),
        ("wrong_target_type", "Series"),
    ],
)
def test_reject_invalid_dataset(
    demo_dataset: TabularDataset, case: str, message: str
) -> None:
    features = demo_dataset.features.copy()
    target = demo_dataset.target.copy()
    changes = {}
    first = features.columns[0]
    match case:
        case "missing_column":
            features = features.drop(columns=first)
        case "extra_column":
            features["unexpected"] = 0.0
        case "reordered_columns":
            features = features[features.columns[::-1]]
        case "duplicate_columns":
            features.columns = [first] * len(features.columns)
        case "string_feature":
            features[first] = "invalid"
        case "bool_feature":
            features[first] = True
        case "complex_feature":
            features[first] = 1j
        case "missing_feature":
            features.iloc[0, 0] = np.nan
        case "missing_target":
            target = target.astype("Int64")
            target.iloc[0] = pd.NA
        case "infinite_feature":
            features.iloc[0, 0] = np.inf
        case "float_target":
            target = target.astype(float)
        case "bool_target":
            target = target.astype(bool)
        case "third_class":
            target.iloc[0] = 2
        case "single_class":
            target[:] = 0
        case "rare_class":
            target[:] = 0
            target.iloc[0] = 1
        case "misaligned_target":
            target = target.iloc[::-1]
        case "duplicate_index":
            features.index = [0] * len(features)
        case "empty_table":
            features, target = features.iloc[:0], target.iloc[:0]
        case "wrong_target_name":
            target.name = "other"
        case "empty_id":
            changes["name"] = ""
        case "empty_schema":
            changes["feature_names"] = ()
        case "duplicate_schema":
            changes["feature_names"] = (first, first)
        case "numpy_schema":
            changes["feature_names"] = tuple(
                np.str_(name) for name in demo_dataset.feature_names
            )
        case "target_in_features":
            changes["target_name"] = first
        case "invalid_class_names":
            changes["target_names"] = ("same", "same")
        case "wrong_features_type":
            features = features.to_numpy()
        case "wrong_target_type":
            target = target.to_numpy()
    invalid = replace(demo_dataset, features=features, target=target, **changes)
    with pytest.raises(ValueError, match=message):
        validate_dataset(invalid)


def test_fingerprint_tracks_values_schema_and_order(
    demo_dataset: TabularDataset,
) -> None:
    original = dataset_fingerprint(demo_dataset)
    assert len(original) == 64
    assert original == dataset_fingerprint(load_demo_dataset())
    features = demo_dataset.features.copy()
    features.iloc[0, 0] += 1
    assert original != dataset_fingerprint(replace(demo_dataset, features=features))
    assert original != dataset_fingerprint(replace(demo_dataset, name="new-version"))
    reordered = replace(
        demo_dataset,
        features=demo_dataset.features.iloc[::-1],
        target=demo_dataset.target.iloc[::-1],
    )
    assert original != dataset_fingerprint(reordered)


def test_another_numeric_loader_can_use_contract() -> None:
    dataset = TabularDataset(
        name="toy-v1",
        features=pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}),
        target=pd.Series([0, 1, 0, 1], name="label"),
        feature_names=("x",),
        target_names=("negative", "positive"),
        target_name="label",
    )
    validate_dataset(dataset)
