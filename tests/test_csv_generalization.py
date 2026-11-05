"""CSV heterogéneos: contratos, ausencia de fuga y recorrido Registry/CLI/HTTP."""

import json
import shlex
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from modelops import cli
from modelops.csv_data import CsvOptions, csv_dataset, inspect_csv, read_csv
from modelops.model_contract import ModelContract, pipeline_contract
from modelops.serving.app import create_app
from modelops.serving.config import ServingConfig
from modelops.serving.predictor import Predictor
from modelops.tracking.adapter import load_registered_pipeline
from modelops.tracking.config import resolve_config
from modelops.training.artifacts import persist_result
from modelops.training.config import TrainingConfig
from modelops.training.runner import train_dataset


def mixed_csv(path: Path, task: str = "classification") -> Path:
    frame = pd.DataFrame(
        {
            "id": np.arange(90),
            "importe [EUR]": [
                np.nan if i % 7 == 0 else float(i) + 0.5 for i in range(90)
            ],
            "ciudad": [
                None if i % 11 == 0 else ["Bilbao", "León", "Cádiz"][i % 3]
                for i in range(90)
            ],
            "postal": ["001", "020", "300"] * 30,
            "vacia": [np.nan] * 90,
            "objetivo": ["01", "1", "otro"] * 30
            if task == "classification"
            else np.arange(90) * 2.5 + 3,
        }
    )
    frame.to_csv(path, index=False, sep=";", decimal=",", encoding="latin-1")
    return path


@pytest.mark.parametrize(
    "task,algorithm",
    [
        ("classification", "logistic-regression"),
        ("classification", "lightgbm"),
        ("regression", "linear-regression"),
        ("regression", "lightgbm"),
    ],
)
def test_mixed_csv_registry_cli_http_parity(
    task,
    algorithm,
    tmp_path,
    capsys,
):
    path = mixed_csv(tmp_path / "mi_tabla.csv", task)
    storage = tmp_path / "registry"
    args = [
        "train",
        str(path),
        "--target",
        "objetivo",
        "--task",
        task,
        "--algorithm",
        algorithm,
        "--name",
        "mi-modelo",
        "--exclude",
        "id",
        "--categorical",
        "postal",
        "--delimiter",
        ";",
        "--decimal",
        ",",
        "--encoding",
        "latin-1",
        "--storage-dir",
        str(storage),
        "--output-dir",
        str(tmp_path / "runs"),
        "--json",
    ]
    assert cli.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["task"] == task
    assert summary["categorical_features"] == ["ciudad", "postal"]
    assert summary["class_names"] == (
        ["01", "1", "otro"] if task == "classification" else []
    )
    assert set(summary["metrics"]) == (
        {"accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro"}
        if task == "classification"
        else {"mae", "rmse", "r2"}
    )
    pipeline = load_registered_pipeline(resolve_config(storage), "mi-modelo", "1")
    contract = pipeline_contract(pipeline)
    assert contract.schema_version == 2 and contract.allow_missing
    run = Path(summary["local_run_dir"])
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["dataset"]["source_options"]["decimal"] == ","
    assert manifest["dataset"]["positive_label"] is None
    assert (
        manifest["split"]["train_class_counts"] is None
        if task == "regression"
        else True
    )
    local = joblib.load(
        run
        / "models"
        / ("lightgbm.joblib" if algorithm == "lightgbm" else "baseline.joblib")
    )
    rows = pd.DataFrame(
        {
            "vacia": [None, 10.0],
            "postal": ["001", None],
            "ciudad": ["ciudad desconocida", None],
            "importe [EUR]": [None, 5.5],
        }
    )
    new = tmp_path / "nuevos.csv"
    rows.to_csv(new, index=False, sep=";", decimal=",", encoding="latin-1")
    # El comando real conserva el dialecto y la versión del entrenamiento.
    predict_args = shlex.split(summary["predict_command"])[4:]
    predict_args[1] = str(new)
    assert cli.main([*predict_args, "--json"]) == 0
    predicted = json.loads(capsys.readouterr().out)
    canonical = contract.parse_csv_features(
        read_csv(new, CsvOptions(";", "latin-1", ",")), ","
    )
    native = pipeline.predict(canonical)
    np.testing.assert_allclose(local.predict(canonical), native)
    key = "label" if task == "classification" else "value"
    np.testing.assert_allclose([p[key] for p in predicted["predictions"]], native)
    if task == "classification":
        assert predicted["classes"] == [0, 1, 2]
        np.testing.assert_allclose(
            [p["probabilities"] for p in predicted["predictions"]],
            pipeline.predict_proba(canonical),
        )
    else:
        assert predicted["task"] == "regression" and predicted["schema_version"] == "2"
    config = ServingConfig(resolve_config(storage), "mi-modelo", "1")
    with TestClient(create_app(config)) as api:
        assert api.get("/health/ready").status_code == 200
        assert api.get("/model/schema").json()["task"] == task
        schema = api.get("/openapi.json").json()["components"]["schemas"]["FeatureRow"]
        assert schema["properties"]["ciudad"]["anyOf"] == [
            {"type": "string"},
            {"type": "null"},
        ]
        instances = json.loads(rows.to_json(orient="records"))
        response = api.post("/predict", json={"instances": instances})
        assert response.status_code == 200, response.text
        assert response.json() == predicted
        for field, invalid in [
            ("ciudad", 5),
            ("importe [EUR]", "5.5"),
            ("importe [EUR]", True),
        ]:
            bad = {**instances[0], field: invalid}
            assert api.post("/predict", json={"instances": [bad]}).status_code == 422
        incomplete = {k: v for k, v in instances[0].items() if k != "vacia"}
        assert api.post("/predict", json={"instances": [incomplete]}).status_code == 422
        assert api.get("/health/ready").status_code == 200
    assert cli.main(predict_args) == 0
    assert (
        "valor predicho" if task == "regression" else "clase original"
    ) in capsys.readouterr().out


def test_imputation_scaling_categories_learn_only_training_partition(tmp_path):
    path = mixed_csv(tmp_path / "data.csv")
    dataset = csv_dataset(
        path,
        "objetivo",
        ["id"],
        options=CsvOptions(";", "latin-1", ","),
        categorical=["postal"],
    )
    original = train_dataset(dataset, TrainingConfig(), candidates=("baseline",))
    pre = original.models["baseline"].named_steps["preprocess"]
    numeric = pre.named_transformers_["numeric"]
    np.testing.assert_allclose(
        numeric.named_steps["impute"].statistics_,
        [original.split.x_train["importe [EUR]"].median(), 0.0],
    )
    encoder = pre.named_transformers_["categorical"].named_steps["encode"]
    assert "exclusivo-test" not in encoder.categories_[0]
    features = dataset.features.copy()
    features.loc[original.split.x_test.index, "ciudad"] = "exclusivo-test"
    features.loc[original.split.x_test.index, "importe [EUR]"] = 1e8
    changed = train_dataset(
        replace(dataset, features=features), original.config, candidates=("baseline",)
    )
    modified = changed.models["baseline"].named_steps["preprocess"]
    np.testing.assert_allclose(
        numeric.named_steps["impute"].statistics_,
        modified.named_transformers_["numeric"].named_steps["impute"].statistics_,
    )
    np.testing.assert_allclose(
        original.models["baseline"].named_steps["model"].coef_,
        changed.models["baseline"].named_steps["model"].coef_,
    )
    assert (
        "exclusivo-test"
        not in modified.named_transformers_["categorical"]
        .named_steps["encode"]
        .categories_[0]
    )


@pytest.mark.parametrize("algorithm", ["baseline", "lightgbm"])
def test_categorical_only_and_sparse_inputs_persist(tmp_path, algorithm):
    path = tmp_path / "categories.csv"
    pd.DataFrame(
        {"codigo": [f"c-{i}" for i in range(240)], "y": ["no", "sí"] * 120}
    ).to_csv(path, index=False)
    data = csv_dataset(path, "y", [])
    result = train_dataset(data, TrainingConfig(), candidates=(algorithm,))
    model = result.models[algorithm]
    assert model.named_steps["preprocess"].sparse_output_
    run = persist_result(result, tmp_path / "runs")
    loaded = joblib.load(run / "models" / f"{algorithm}.joblib")
    prediction = Predictor(loaded).predict(pd.DataFrame({"codigo": ["nueva", None]}))
    assert len(prediction) == 2


@pytest.mark.parametrize(
    "options", [CsvOptions(";", "latin-1", ","), CsvOptions("\t"), CsvOptions("|")]
)
def test_dialects_quoted_headers_and_multiline_fields(tmp_path, options):
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {"nombre, completo": ["a\nb", "León"], "x": [1.5, np.nan], "y": ["01", "1"]}
    ).to_csv(
        path,
        index=False,
        sep=options.delimiter,
        decimal=options.decimal,
        encoding=options.encoding,
    )
    parsed = read_csv(path, options)
    assert parsed["nombre, completo"].tolist() == ["a\nb", "León"]
    assert parsed["y"].tolist() == ["01", "1"]
    info = inspect_csv(path, options)
    assert info["rows"] == 2 and info["columns"][1]["missing"] == 1
    data = csv_dataset(path, "y", [], options=options)
    assert data.features["x"].iloc[0] == 1.5


@pytest.mark.parametrize(
    "changes",
    [
        {"delimiter": ""},
        {"delimiter": "xx"},
        {"delimiter": "\n"},
        {"delimiter": '"'},
        {"decimal": ","},
        {"decimal": "x"},
    ],
)
def test_invalid_dialect_rejected(changes):
    with pytest.raises(ValueError):
        CsvOptions(**changes)


@pytest.mark.parametrize(
    "task,algorithm",
    [("regression", "logistic-regression"), ("classification", "linear-regression")],
)
def test_incompatible_algorithm_rejected_before_persistence(
    tmp_path, capsys, task, algorithm
):
    path = tmp_path / "does-not-exist.csv"
    output = tmp_path / "runs"
    assert (
        cli.main(
            [
                "train",
                str(path),
                "--target",
                "y",
                "--name",
                "m",
                "--algorithm",
                algorithm,
                "--task",
                task,
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    assert "incompatible" in capsys.readouterr().err
    assert not output.exists()


def test_forced_types_and_invalid_numeric_values(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x,postal,y\n1,001,a\nbad,020,b\n3,001,a\n4,020,b\n")
    data = csv_dataset(path, "y", [], categorical=["postal"])
    assert data.categorical_features == ("x", "postal")
    for overrides in [
        {"numeric": ["x"]},
        {"categorical": ["missing"]},
        {"numeric": ["postal"], "categorical": ["postal"]},
        {"categorical": ["y"]},
    ]:
        with pytest.raises(ValueError):
            csv_dataset(path, "y", [], **overrides)
    with pytest.raises(ValueError, match="regresión"):
        csv_dataset(path, "y", [], task="regression")
    path.write_text("x,y\ninf,a\n2,b\n3,a\n4,b\n")
    with pytest.raises(ValueError, match="finitos"):
        csv_dataset(path, "y", [])


def test_csv_prediction_uses_saved_types_and_rejects_numeric_text(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": np.arange(20), "y": ["no", "sí"] * 10}).to_csv(path, index=False)
    result = train_dataset(
        csv_dataset(path, "y", []), TrainingConfig(), candidates=("baseline",)
    )
    contract = pipeline_contract(result.models["baseline"])
    for invalid in ["bad", "inf"]:
        with pytest.raises(ValueError):
            contract.parse_csv_features(pd.DataFrame({"x": [invalid]}))
    parsed = contract.parse_csv_features(pd.DataFrame({"x": [None]}))
    assert len(Predictor(result.models["baseline"]).predict(parsed)) == 1
    for invalid_contract in [
        {"schema_version": 99},
        {"schema_version": 1},
        {"categorical_features": ["other"]},
        {"target_name": "x"},
        {"task": "regression"},
    ]:
        with pytest.raises(ValueError):
            ModelContract.model_validate({**contract.model_dump(), **invalid_contract})
