"""Contrato HTTP, salud y ciclo de vida con un pipeline real en memoria."""

import copy
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from modelops.serving import app as serving
from modelops.serving.config import ServingConfig
from modelops.serving.contracts import MAX_BATCH_SIZE
from modelops.tabular import DEMO_FEATURE_NAMES
from modelops.tracking.config import TrackingConfig
from modelops.training.runner import TrainingResult


@pytest.fixture
def config(tmp_path: Path) -> ServingConfig:
    database = tmp_path / "mlflow.db"
    database.touch()
    return ServingConfig(
        TrackingConfig(database, tmp_path / "artifacts"), "modelops-baseline", "7"
    )


@pytest.fixture
def row(trained_result: TrainingResult) -> dict:
    return trained_result.split.x_test.iloc[0].to_dict()


@pytest.fixture
def app(
    config: ServingConfig,
    trained_result: TrainingResult,
    monkeypatch: pytest.MonkeyPatch,
) -> FastAPI:
    monkeypatch.setattr(
        serving,
        "load_registered_pipeline",
        lambda *args: copy.deepcopy(trained_result.models["baseline"]),
    )
    return serving.create_app(config)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client


def test_health_and_shutdown(app: FastAPI) -> None:
    assert app.state.predictor is None
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "alive"}
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["model"]["version"] == "7"
    assert app.state.predictor is None


@pytest.mark.parametrize("count", [1, 5, MAX_BATCH_SIZE])
def test_valid_batch_preserves_parity_and_column_order(
    client: TestClient, trained_result: TrainingResult, count: int
) -> None:
    features = trained_result.dataset.features.iloc[:count]
    # La inserción de claves en JSON no cambia la selección del pipeline.
    rows = [dict(reversed(list(row.items()))) for row in features.to_dict("records")]
    response = client.post("/predict", json={"instances": rows})
    assert response.status_code == 200
    value = response.json()
    assert value["schema_version"] == "1"
    assert value["model"] == {
        "name": "modelops-baseline",
        "version": "7",
        "uri": "models:/modelops-baseline/7",
    }
    assert value["classes"] == [0, 1]
    assert value["class_names"] == ["malignant", "benign"]
    pipeline = trained_result.models["baseline"]
    np.testing.assert_array_equal(
        [p["label"] for p in value["predictions"]], pipeline.predict(features)
    )
    np.testing.assert_allclose(
        [p["probabilities"] for p in value["predictions"]],
        pipeline.predict_proba(features),
        rtol=1e-9,
        atol=1e-12,
    )


def test_integer_features_are_numbers(client: TestClient, row: dict) -> None:
    row["mean radius"] = 18
    assert client.post("/predict", json={"instances": [row]}).status_code == 200


@pytest.mark.parametrize(
    "invalid",
    [None, True, False, "17.99", [], {}, float("nan"), float("inf"), -float("inf")],
)
def test_bad_feature_is_rejected_without_echoing_input(
    client: TestClient, row: dict, invalid: object
) -> None:
    row["mean radius"] = invalid
    response = client.post(
        "/predict",
        content=json.dumps({"instances": [row]}),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "instances", 0, "mean radius"]
    assert set(error) == {"loc", "msg", "type"}
    assert client.get("/health/ready").status_code == 200


@pytest.mark.parametrize("kind", ["missing", "extra", "python_name"])
def test_exact_feature_names(client: TestClient, row: dict, kind: str) -> None:
    if kind == "missing":
        del row["mean radius"]
    elif kind == "extra":
        row["unexpected"] = 1
    else:
        row["mean_radius"] = row.pop("mean radius")
    assert client.post("/predict", json={"instances": [row]}).status_code == 422


@pytest.mark.parametrize(
    "body", [{}, [], {"instances": []}, {"instances": {}}, {"instances": [None]}]
)
def test_invalid_body(client: TestClient, body: object) -> None:
    assert client.post("/predict", json=body).status_code == 422


def test_invalid_json_and_batch_limits(client: TestClient, row: dict) -> None:
    assert (
        client.post(
            "/predict", content="{", headers={"content-type": "application/json"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/predict", json={"instances": [row], "unexpected": True}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/predict", json={"instances": [row] * (MAX_BATCH_SIZE + 1)}
        ).status_code
        == 422
    )


def test_openapi_documents_all_required_features(client: TestClient) -> None:
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    feature_schema = schemas["FeatureRow"]
    assert feature_schema["required"] == list(DEMO_FEATURE_NAMES)
    assert feature_schema["additionalProperties"] is False
    assert schemas["PredictRequest"]["properties"]["instances"]["maxItems"] == 256


def test_load_once_and_no_fit_or_registry_access_per_request(
    config: ServingConfig,
    trained_result: TrainingResult,
    row: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = Mock(return_value=trained_result.models["baseline"])
    monkeypatch.setattr(serving, "load_registered_pipeline", loader)
    monkeypatch.setattr(Pipeline, "fit", Mock(side_effect=AssertionError("no fit")))
    app = serving.create_app(config)
    loader.assert_not_called()
    with TestClient(app) as client:
        loader.assert_called_once_with(config.tracking, config.model_name, "7")
        loader.side_effect = AssertionError("No Registry access after startup")
        for _ in range(3):
            assert client.post("/predict", json={"instances": [row]}).status_code == 200
            assert client.get("/health/ready").status_code == 200
        assert loader.call_count == 1


@pytest.mark.parametrize("failure", [FileNotFoundError, ValueError, RuntimeError])
def test_failed_load_has_live_process_but_no_readiness_or_predictions(
    config: ServingConfig,
    row: dict,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: type[Exception],
) -> None:
    loader = Mock(side_effect=failure("SECRET private storage path"))
    monkeypatch.setattr(serving, "load_registered_pipeline", loader)
    with (
        caplog.at_level(logging.ERROR),
        TestClient(serving.create_app(config)) as client,
    ):
        assert client.get("/health/live").status_code == 200
        ready = client.get("/health/ready")
        assert ready.status_code == 503
        assert "SECRET" not in ready.text
        assert client.post("/predict", json={"instances": [row]}).status_code == 503
    assert "SECRET" not in caplog.text
    assert "reinicia" in caplog.text
    assert loader.call_count == 1


def test_missing_database_does_not_initialize_registry(
    tmp_path: Path, row: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ServingConfig(
        TrackingConfig(tmp_path / "absent" / "mlflow.db", tmp_path / "artifacts"),
        "modelops-baseline",
        "1",
    )
    loader = Mock()
    monkeypatch.setattr(serving, "load_registered_pipeline", loader)
    with TestClient(serving.create_app(config)) as client:
        assert client.get("/health/ready").status_code == 503
    loader.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["exception", "nan", "shape", "sum", "label"])
def test_runtime_failure_is_sanitized_and_clears_readiness(
    client: TestClient,
    app: FastAPI,
    row: dict,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    pipeline = app.state.predictor.pipeline
    if failure == "exception":
        monkeypatch.setattr(
            pipeline, "predict", Mock(side_effect=RuntimeError("SECRET"))
        )
    elif failure == "label":
        monkeypatch.setattr(pipeline, "predict", lambda _: np.array([2]))
    else:
        values = {
            "nan": [[float("nan"), 0.0]],
            "shape": [[1.0]],
            "sum": [[0.1, 0.1]],
        }
        monkeypatch.setattr(
            pipeline, "predict_proba", lambda _: np.array(values[failure])
        )
    response = client.post("/predict", json={"instances": [row]})
    assert response.status_code == 500
    assert response.json() == {"detail": "No se pudo completar la inferencia."}
    assert client.get("/health/ready").status_code == 503
    assert client.get("/health/live").status_code == 200
    assert client.post("/predict", json={"instances": [row]}).status_code == 503


def test_classes_follow_native_pipeline_order(
    config: ServingConfig,
    trained_result: TrainingResult,
    row: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = copy.deepcopy(trained_result.models["baseline"])
    pipeline.named_steps["model"].classes_ = np.array([1, 0])
    monkeypatch.setattr(serving, "load_registered_pipeline", lambda *args: pipeline)
    with TestClient(serving.create_app(config)) as client:
        value = client.post("/predict", json={"instances": [row]}).json()
    assert value["classes"] == [1, 0]
    assert value["class_names"] == ["benign", "malignant"]
    features = pd.DataFrame([row], columns=DEMO_FEATURE_NAMES)
    assert value["predictions"][0]["label"] == pipeline.predict(features)[0]
    np.testing.assert_allclose(
        value["predictions"][0]["probabilities"], pipeline.predict_proba(features)[0]
    )


@pytest.mark.parametrize("kind", ["schema", "classes", "not_pipeline", "unfitted"])
def test_incompatible_model_is_never_ready(
    config: ServingConfig,
    trained_result: TrainingResult,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    pipeline = copy.deepcopy(trained_result.models["baseline"])
    if kind == "schema":
        pipeline.named_steps["preprocess"].feature_names_in_[0] = "wrong"
    elif kind == "classes":
        pipeline.named_steps["model"].classes_ = np.array([1, 2])
    elif kind == "not_pipeline":
        pipeline = object()
    else:
        pipeline = Pipeline([])
    monkeypatch.setattr(serving, "load_registered_pipeline", lambda *args: pipeline)
    with TestClient(serving.create_app(config)) as client:
        assert client.get("/health/ready").status_code == 503
