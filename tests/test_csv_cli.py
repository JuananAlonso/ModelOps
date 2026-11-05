"""CSV propios → CLI → Registry → CLI/API, usando SQLite real y sin red."""

import copy
import json
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException

from modelops import cli
from modelops.csv_data import csv_dataset, inspect_csv, read_csv
from modelops.model_contract import pipeline_contract
from modelops.serving.app import create_app
from modelops.serving.config import ServingConfig
from modelops.tracking.adapter import load_registered_pipeline, track_result
from modelops.tracking.config import resolve_config
from modelops.training.artifacts import persist_result
from modelops.training.runner import TrainingResult


@pytest.fixture
def csv_files(tmp_path: Path):
    features = pd.DataFrame({"edad": np.arange(40) + 20, "saldo": np.arange(40) ** 2})
    frame = features.assign(id_cliente=np.arange(40), churn=["no", "sí"] * 20)
    train = tmp_path / "clientes.csv"
    frame.to_csv(train, index=False)
    new = tmp_path / "nuevos.csv"
    features.iloc[:5].to_csv(new, index=False)
    return train, new, features


@pytest.mark.parametrize("algorithm", ["logistic-regression", "lightgbm"])
def test_cli_and_http_match_registered_pipeline_for_distinct_schemas(
    algorithm: str,
    csv_files,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, new, features = csv_files
    if algorithm == "logistic-regression":
        # También admite nombres que colisionarían al sustituir espacios por _.
        features.columns = ["a b", "a_b"]
        features.assign(id_cliente=np.arange(40), churn=["no", "sí"] * 20).to_csv(
            train, index=False
        )
        features.iloc[:5].to_csv(new, index=False)
    else:
        # El selector conserva los nombres externos; LightGBM recibe una matriz.
        features.columns = ["antiguedad", "importe [EUR]"]
        features.assign(id_cliente=np.arange(40), churn=["no", "sí"] * 20).to_csv(
            train, index=False
        )
        features.iloc[:5].to_csv(new, index=False)
    storage = tmp_path / "registry"
    common = ["--storage-dir", str(storage)]
    name = "clientes-" + algorithm
    train_args = [
        "train",
        str(train),
        "--target",
        "churn",
        "--algorithm",
        algorithm,
        "--name",
        name,
        "--exclude",
        "id_cliente",
        "--output-dir",
        str(tmp_path / "runs"),
        "--json",
        *common,
    ]
    assert cli.main(train_args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["name"] == name and summary["version"] == "1"
    assert summary["class_names"] == ["no", "sí"]
    assert shlex.split(summary["predict_command"])[-3:] == [
        "1",
        "--tracking-uri",
        resolve_config(storage).tracking_uri,
    ]
    assert cli.main(train_args) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["version"] == "2" and "--version 2" in second["predict_command"]
    pipeline = load_registered_pipeline(resolve_config(storage), name, "1")
    contract = pipeline_contract(pipeline)
    assert contract.feature_names == tuple(features.columns)
    assert contract.class_names == ("no", "sí")
    local = Path(summary["local_run_dir"])
    assert len(list((local / "models").iterdir())) == 1
    assert json.loads((local / "contracts.json").read_text())
    candidate = "baseline" if algorithm == "logistic-regression" else "lightgbm"
    if candidate == "baseline":
        positions = json.loads((local / "split.json").read_text())["train_positions"]
        scaler = (
            pipeline.named_steps["preprocess"]
            .named_transformers_["numeric"]
            .named_steps["scale"]
        )
        np.testing.assert_allclose(scaler.mean_, features.iloc[positions].mean())
        assert not np.allclose(scaler.mean_, features.mean())

    predict_args = [
        "predict",
        str(new),
        "--model",
        name,
        "--version",
        "1",
        "--json",
        *common,
    ]
    assert cli.main(predict_args) == 0
    expected = json.loads(capsys.readouterr().out)
    rows = features.iloc[:5]
    np.testing.assert_array_equal(
        [p["label"] for p in expected["predictions"]], pipeline.predict(rows)
    )
    np.testing.assert_allclose(
        [p["probabilities"] for p in expected["predictions"]],
        pipeline.predict_proba(rows),
        rtol=1e-9,
        atol=1e-12,
    )
    # Predecir no puede entrenar, ni siquiera desde el objeto recuperado.
    monkeypatch.setattr(
        type(pipeline), "fit", Mock(side_effect=AssertionError("no fit"))
    )
    with TestClient(
        create_app(ServingConfig(resolve_config(storage), name, "1"))
    ) as api:
        assert api.get("/model/schema").json()["feature_names"] == list(
            features.columns
        )
        required = api.get("/openapi.json").json()["components"]["schemas"][
            "FeatureRow"
        ]["required"]
        assert required == list(features.columns)
        for ordered in (rows, rows.iloc[:, ::-1]):
            ordered.to_csv(new, index=False)
            assert cli.main(predict_args) == 0
            assert json.loads(capsys.readouterr().out) == expected
            response = api.post(
                "/predict", json={"instances": ordered.to_dict("records")}
            )
            assert response.status_code == 200
            assert response.json() == expected
        for invalid, field in (
            (rows.iloc[:, 1:], features.columns[0]),
            (rows.assign(extra=1), "extra"),
        ):
            invalid.to_csv(new, index=False)
            assert cli.main(predict_args) == 2
            captured = capsys.readouterr()
            assert captured.out == "" and field in captured.err
            response = api.post(
                "/predict", json={"instances": invalid.to_dict("records")}
            )
            assert response.status_code == 422 and field in response.text
        assert api.get("/health/ready").status_code == 200
    rows.to_csv(new, index=False)
    missing_args = predict_args.copy()
    missing_args[missing_args.index("--version") + 1] = "999"
    assert cli.main(missing_args) == 2
    assert capsys.readouterr().out == ""
    with TestClient(
        create_app(ServingConfig(resolve_config(storage), name, "999"))
    ) as api:
        assert api.get("/health/ready").status_code == 503
        assert (
            api.post(
                "/predict", json={"instances": rows.to_dict("records")}
            ).status_code
            == 503
        )


@pytest.mark.parametrize(
    "content,expected",
    [
        ("a,a,y\n1,2,no\n", "duplicadas"),
        ("a,y\n1,no,extra\n", "campos"),
        ("a,y\n1\n", "campos"),
        ('a,y\n"unfinished', "CSV"),
        ("a,y\n1,no\n2,no\n", "dos clases"),
        ("a,y\n1,no\n2,sí\n", "cuatro filas"),
        ("a,y\n1,no\n2,sí\n3,no\n4,sí\n", "test_size"),
        (
            "a,y\n1,no\n2,sí\n3,sí\n4,sí\n5,sí\n6,sí\n7,sí\n8,sí\n9,sí\n10,sí\n",
            "dos filas",
        ),
        ("a,y\ninf,no\n2,sí\n3,no\n4,sí\n", "finitos"),
        ("a,y\n1,\n2,sí\n3,no\n4,sí\n", "objetivo tiene ausentes"),
        ("y\nno\nsí\nno\nsí\n", "variables"),
    ],
)
def test_rejected_csv_is_actionable_and_never_registers(
    content: str,
    expected: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(content)
    register = Mock(side_effect=AssertionError("must not register"))
    monkeypatch.setattr(cli, "track_result", register)
    assert (
        cli.main(
            [
                "train",
                str(path),
                "--target",
                "y",
                "--algorithm",
                "lightgbm",
                "--name",
                "invalid",
                "--json",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert expected.lower() in captured.err.lower()
    assert "--help" in captured.err and "Traceback" not in captured.err
    register.assert_not_called()


def test_inspection_and_invalid_target_or_exclusion(
    csv_files, capsys: pytest.CaptureFixture
) -> None:
    train, _, _ = csv_files
    info = inspect_csv(train)
    assert info["rows"] == 40
    target = next(col for col in info["columns"] if col["name"] == "churn")
    assert target["binary_candidate"] and target["missing"] == 0
    assert cli.main(["inspect", str(train), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == info
    for target, exclude in (
        ("absent", []),
        ("churn", ["absent"]),
        ("churn", ["churn"]),
        ("churn", ["edad", "saldo", "id_cliente"]),
    ):
        with pytest.raises(ValueError):
            csv_dataset(train, target, exclude)


def test_registry_failure_preserves_local_artifacts(
    csv_files,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, _, _ = csv_files
    monkeypatch.setattr(
        cli, "track_result", Mock(side_effect=MlflowException("backend unavailable"))
    )
    root = tmp_path / "runs"
    assert (
        cli.main(
            [
                "train",
                str(train),
                "--target",
                "churn",
                "--algorithm",
                "logistic-regression",
                "--name",
                "churn",
                "--exclude",
                "id_cliente",
                "--output-dir",
                str(root),
                "--json",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == "" and "registro" in captured.err
    runs = list(root.iterdir())
    assert len(runs) == 1 and str(runs[0]) in captured.err
    assert (runs[0] / "manifest.json").is_file()


def test_legacy_pipeline_without_embedded_contract_still_serves(
    trained_result: TrainingResult, tmp_path: Path
) -> None:
    model = copy.deepcopy(trained_result.models["baseline"])
    del model.modelops_contract_
    legacy = replace(
        trained_result,
        models={"baseline": model},
        metrics={"baseline": trained_result.metrics["baseline"]},
    )
    tracking = resolve_config(tmp_path / "registry")
    local = persist_result(legacy, tmp_path / "runs")
    reference = track_result(legacy, local, tracking).candidates["baseline"]
    restored = load_registered_pipeline(tracking, reference.name, reference.version)
    assert not hasattr(restored, "modelops_contract_")
    config = ServingConfig(tracking, reference.name, reference.version)
    with TestClient(create_app(config)) as api:
        rows = trained_result.split.x_test.iloc[:3]
        response = api.post("/predict", json={"instances": rows.to_dict("records")})
        assert response.status_code == 200
        np.testing.assert_array_equal(
            [p["label"] for p in response.json()["predictions"]], model.predict(rows)
        )


@pytest.mark.parametrize("command", ["inspect", "train", "predict"])
def test_help_has_example(command: str, capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as exit:
        cli.main([command, "--help"])
    assert exit.value.code == 0
    assert "Ejemplo:" in capsys.readouterr().out


def test_installed_cli_from_other_directory(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("x,y\n1,no\n2,sí\n")
    completed = subprocess.run(
        [sys.executable, "-I", "-m", "modelops.cli", "inspect", str(path), "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(completed.stdout)["rows"] == 2
    assert read_csv(path).columns.tolist() == ["x", "y"]


@pytest.mark.parametrize(
    "argument,environment,expected",
    [
        (None, None, "modelops-lightgbm-audit-model"),
        (None, "configured-env", "configured-env"),
        ("configured-cli", "configured-env", "configured-cli"),
    ],
)
def test_experiment_default_and_override_priority(
    csv_files, tmp_path, capsys, monkeypatch, argument, environment, expected
):
    train, _, _ = csv_files
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    if environment is not None:
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", environment)
    options = ["--experiment-name", argument] if argument is not None else []
    assert (
        cli.main(
            [
                "train",
                str(train),
                "--target",
                "churn",
                "--algorithm",
                "lightgbm",
                "--name",
                "audit-model",
                "--exclude",
                "id_cliente",
                "--storage-dir",
                str(tmp_path / "registry"),
                "--output-dir",
                str(tmp_path / "runs"),
                "--json",
                *options,
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["experiment_name"] == expected
    config = resolve_config(tmp_path / "registry")
    from modelops.tracking.adapter import make_client

    experiment = make_client(config).get_experiment_by_name(expected)
    assert (
        experiment is not None and experiment.artifact_location == config.artifact_uri
    )


def test_inspect_keeps_separators_aligned_for_wrapped_names():
    info = {
        "rows": 10,
        "columns": [
            {
                "name": "variable_con_un_nombre_muy_largo_que_ocupa_varias_lineas",
                "dtype": "float64",
                "missing": 0,
                "unique": 10,
                "classification_candidate": False,
                "regression_candidate": True,
            }
        ],
        "issues": [],
    }
    output = cli._format_inspection(info, terminal_width=80)
    lines = [line for line in output.splitlines() if "│" in line]
    assert len(lines) > 2
    boundaries = [[i for i, value in enumerate(line) if value == "│"] for line in lines]
    assert all(value == boundaries[0] for value in boundaries)
    assert all(len(line) <= 80 for line in lines)
