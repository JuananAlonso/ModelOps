"""Integración real con SQLite y archivos temporales, sin servidor ni red."""

import json
from dataclasses import replace
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pytest
from mlflow.exceptions import MlflowException
from mlflow.models import Model

from modelops.tracking import adapter
from modelops.tracking.adapter import (
    compare_runs,
    ensure_experiment,
    load_registered_pipeline,
    make_client,
    track_result,
)
from modelops.tracking.config import TrackingConfig
from modelops.training.artifacts import persist_result
from modelops.training.config import TrainingConfig
from modelops.training.runner import TrainingResult, train_dataset


@pytest.fixture
def tracking_config(tmp_path: Path) -> TrackingConfig:
    root = tmp_path / "mlflow with spaces"
    return TrackingConfig(root / "mlflow.db", root / "artifacts", "test-tabular")


def test_two_executions_tracking_and_registry_versions(
    tracking_config: TrackingConfig, trained_result: TrainingResult, tmp_path: Path
) -> None:
    result_123 = train_dataset(trained_result.dataset, TrainingConfig(seed=123))
    tracked = []
    for result in (trained_result, result_123):
        local = persist_result(result, tmp_path / "local", project_root=Path.cwd())
        tracked.append(track_result(result, local, tracking_config))
    client = make_client(tracking_config)
    assert tracked[0].experiment_id == tracked[1].experiment_id
    assert tracked[0].parent_run_id != tracked[1].parent_run_id
    runs = client.search_runs([tracked[0].experiment_id])
    assert len(runs) == 6
    assert all(run.info.status == "FINISHED" for run in runs)
    for execution, result in zip(tracked, (trained_result, result_123), strict=True):
        parent = client.get_run(execution.parent_run_id)
        assert not parent.data.metrics
        assert parent.data.params["seed"] == str(result.config.seed)
        for candidate, reference in execution.candidates.items():
            child = client.get_run(reference.run_id)
            assert child.data.tags["mlflow.parentRunId"] == execution.parent_run_id
            assert child.data.tags["modelops.candidate"] == candidate
            assert child.data.params["candidate"] == candidate
            assert child.data.params["seed"] == str(result.config.seed)
            assert child.data.params["test_size"] == "0.2"
            assert child.data.params["dataset_id"] == result.dataset.name
            assert child.data.params["dataset_sha256"] == result.dataset_sha256
            assert child.data.params["positive_label"] == "1"
            assert child.data.params["python_version"].startswith("3.12.")
            assert child.data.params["version.mlflow"] == "3.16.1"
            assert "modelops.git_revision" in child.data.tags
            assert child.data.metrics == pytest.approx(result.metrics[candidate])
            registered = client.get_model_version(reference.name, reference.version)
            assert registered.run_id == reference.run_id
            assert registered.source == reference.source_uri
            assert registered.tags["modelops.dataset_sha256"] == result.dataset_sha256
            assert not client.get_registered_model(reference.name).aliases
    for candidate in ("baseline", "lightgbm"):
        assert tracked[0].candidates[candidate].version == "1"
        assert tracked[1].candidates[candidate].version == "2"
    assert (
        client.get_run(tracked[0].candidates["baseline"].run_id).data.params[
            "model.solver"
        ]
        == "lbfgs"
    )
    assert (
        client.get_run(tracked[0].candidates["lightgbm"].run_id).data.params[
            "model.deterministic"
        ]
        == "true"
    )
    comparison = compare_runs(tracking_config)
    assert len(comparison) == 4
    assert {row["seed"] for row in comparison} == {42, 123}
    assert {row["candidate"] for row in comparison} == {"baseline", "lightgbm"}
    assert len({row["dataset_sha256"] for row in comparison}) == 1


def test_recover_artifacts_signature_and_equivalent_predictions(
    tracking_config: TrackingConfig, trained_result: TrainingResult, tmp_path: Path
) -> None:
    local = persist_result(trained_result, tmp_path / "local")
    tracked = track_result(trained_result, local, tracking_config)
    client = make_client(tracking_config)
    destination = tmp_path / "downloads"
    destination.mkdir()
    downloaded = Path(
        client.download_artifacts(
            tracked.parent_run_id, "training/manifest.json", str(destination)
        )
    )
    assert (
        json.loads(downloaded.read_text())["dataset"]["sha256"]
        == trained_result.dataset_sha256
    )
    for candidate, reference in tracked.candidates.items():
        model_dir = Path(
            client.download_artifacts(reference.run_id, "pipeline", str(destination))
        )
        metadata = Model.load(str(model_dir / "MLmodel"))
        assert metadata.signature.inputs.input_names() == list(
            trained_result.dataset.feature_names
        )
        assert metadata.signature.outputs is not None
        assert metadata.saved_input_example_info is not None
        assert (model_dir / "input_example.json").is_file()
        assert (model_dir / "requirements.txt").is_file()
        loaded = load_registered_pipeline(
            tracking_config, reference.name, reference.version
        )
        original = trained_result.models[candidate]
        features = trained_result.split.x_test
        np.testing.assert_array_equal(
            loaded.predict(features), original.predict(features)
        )
        np.testing.assert_allclose(
            loaded.predict_proba(features.iloc[:, ::-1]),
            original.predict_proba(features),
            rtol=1e-9,
            atol=1e-12,
        )
        pyfunc = mlflow.pyfunc.load_model(reference.source_uri)
        np.testing.assert_array_equal(
            pyfunc.predict(features), original.predict(features)
        )
        with pytest.raises(MlflowException, match="missing|required|Missing"):
            pyfunc.predict(features.iloc[:, 1:])
        if candidate == "baseline":
            scaler = loaded.named_steps["preprocess"].named_transformers_["numeric"]
            np.testing.assert_allclose(
                scaler.mean_, trained_result.split.x_train.mean(), rtol=1e-12
            )


def test_undefined_metrics_are_null_artifacts_and_absent_numeric_metrics(
    tracking_config: TrackingConfig, trained_result: TrainingResult, tmp_path: Path
) -> None:
    metrics = {key: dict(value) for key, value in trained_result.metrics.items()}
    metrics["baseline"]["precision"] = None
    result = replace(trained_result, metrics=metrics)
    local = persist_result(result, tmp_path / "local")
    tracked = track_result(result, local, tracking_config)
    client = make_client(tracking_config)
    child = client.get_run(tracked.candidates["baseline"].run_id)
    assert "precision" not in child.data.metrics
    assert json.loads(child.data.tags["modelops.undefined_metrics"]) == ["precision"]
    artifact = Path(client.download_artifacts(child.info.run_id, "metrics.json"))
    assert json.loads(artifact.read_text())["precision"] is None
    assert (
        next(
            row
            for row in compare_runs(tracking_config)
            if row["candidate"] == "baseline"
        )["metrics"]["precision"]
        is None
    )


def test_existing_experiment_rejects_a_different_artifact_root(
    tracking_config: TrackingConfig, tmp_path: Path
) -> None:
    client = make_client(tracking_config)
    experiment_id = ensure_experiment(client, tracking_config)
    different = replace(tracking_config, artifact_dir=tmp_path / "different")
    with pytest.raises(ValueError, match="otro artifact location"):
        ensure_experiment(client, different)
    assert not client.search_runs([experiment_id])


def test_failure_marks_runs_and_keeps_local_result(
    tracking_config: TrackingConfig,
    trained_result: TrainingResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = persist_result(trained_result, tmp_path / "local")

    original_save = adapter.mlflow.sklearn.save_model
    calls = 0

    def fail_second_model(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("serialization failed")
        original_save(*args, **kwargs)

    monkeypatch.setattr(adapter.mlflow.sklearn, "save_model", fail_second_model)
    with pytest.raises(OSError, match="serialization failed"):
        track_result(trained_result, local, tracking_config)
    client = make_client(tracking_config)
    experiment = client.get_experiment_by_name(tracking_config.experiment_name)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 3
    assert sorted(run.info.status for run in runs) == ["FAILED", "FAILED", "FINISHED"]
    assert client.get_model_version("modelops-baseline", "1").status == "READY"
    assert not compare_runs(tracking_config)
    assert (local / "models" / "baseline.joblib").is_file()


def test_corrupt_local_artifact_is_rejected_before_tracking(
    tracking_config: TrackingConfig, trained_result: TrainingResult, tmp_path: Path
) -> None:
    local = persist_result(trained_result, tmp_path / "local")
    (local / "models" / "baseline.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        track_result(trained_result, local, tracking_config)
    assert not tracking_config.database_path.exists()


def test_explicit_client_does_not_use_or_change_global_tracking(
    tracking_config: TrackingConfig,
    trained_result: TrainingResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "http://127.0.0.1:9")
    previous = (
        mlflow.get_tracking_uri(),
        mlflow.get_registry_uri(),
        mlflow.active_run(),
    )
    local = persist_result(trained_result, tmp_path / "local")
    tracked = track_result(trained_result, local, tracking_config)
    assert tracked.candidates
    assert previous == (
        mlflow.get_tracking_uri(),
        mlflow.get_registry_uri(),
        mlflow.active_run(),
    )
