"""Persistencia cargable, procedencia y protección de ejecuciones anteriores."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from modelops.training import artifacts
from modelops.training.artifacts import persist_result
from modelops.training.runner import TrainingResult


def test_persist_load_and_manifest(
    trained_result: TrainingResult, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[1]
    run_dir = persist_result(trained_result, tmp_path, project_root=root)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    split = json.loads((run_dir / "split.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["run_id"] == run_dir.name
    assert manifest["seed"] == config["seed"] == trained_result.config.seed
    assert manifest["dataset"]["sha256"] == trained_result.dataset_sha256
    assert manifest["dataset"]["class_names"]["1"] == "benign"
    assert manifest["split"]["train_rows"] == 455
    assert manifest["split"]["test_rows"] == 114
    assert manifest["environment"]["python"].startswith("3.12.")
    assert {"numpy", "pandas", "scikit-learn", "lightgbm", "scipy", "joblib"} <= set(
        manifest["environment"]["dependencies"]
    )
    assert len(manifest["git"]["revision"]) == 40
    assert isinstance(manifest["git"]["dirty"], bool)
    assert len(manifest["project_files_sha256"]["uv.lock"]) == 64
    assert config["native_thread_limit"] == 1
    assert config["models"]["lightgbm"]["parameters"]["deterministic"] is True
    assert config["models"]["baseline"]["preprocessing"]["numeric"]["type"] == (
        "StandardScaler"
    )
    assert config["models"]["lightgbm"]["preprocessing"]["numeric"]["type"] == (
        "passthrough"
    )
    assert metrics == trained_result.metrics
    assert split["train_positions"] == trained_result.split.train_positions.tolist()
    assert split["test_positions"] == trained_result.split.test_positions.tolist()
    for relative, digest in manifest["artifacts_sha256"].items():
        assert hashlib.sha256((run_dir / relative).read_bytes()).hexdigest() == digest
    for name, model in trained_result.models.items():
        loaded = joblib.load(run_dir / "models" / f"{name}.joblib")
        np.testing.assert_array_equal(
            loaded.predict(trained_result.split.x_test),
            model.predict(trained_result.split.x_test),
        )
        # El selector persistido mantiene el orden aprendido aunque llegue otro orden.
        reversed_features = trained_result.split.x_test.iloc[:, ::-1]
        np.testing.assert_allclose(
            loaded.predict_proba(reversed_features),
            model.predict_proba(trained_result.split.x_test),
            rtol=1e-12,
        )
        with pytest.raises(ValueError, match="columns are missing"):
            loaded.predict(trained_result.split.x_test.iloc[:, 1:])


def test_new_run_preserves_previous_artifacts(
    trained_result: TrainingResult, tmp_path: Path
) -> None:
    first = persist_result(trained_result, tmp_path)
    snapshot = {path: path.read_bytes() for path in first.rglob("*") if path.is_file()}
    second = persist_result(trained_result, tmp_path)
    assert first != second
    assert all(path.read_bytes() == content for path, content in snapshot.items())
    assert json.loads((second / "manifest.json").read_text())["git"] is None


def test_collision_fails_instead_of_overwriting(
    trained_result: TrainingResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(artifacts, "datetime", SimpleNamespace(now=lambda tz: now))
    monkeypatch.setattr(artifacts, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    first = persist_result(trained_result, tmp_path)
    original = (first / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        persist_result(trained_result, tmp_path)
    assert (first / "manifest.json").read_bytes() == original


def test_failed_write_has_no_success_manifest(
    trained_result: TrainingResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("Disk write failure")

    monkeypatch.setattr(joblib, "dump", fail)
    with pytest.raises(OSError, match="Disk write"):
        persist_result(trained_result, tmp_path)
    assert len(list(tmp_path.iterdir())) == 1
    assert not list(tmp_path.rglob("manifest.json"))


def test_unavailable_git_does_not_prevent_persistence(
    trained_result: TrainingResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git is not installed")

    monkeypatch.setattr(artifacts.subprocess, "run", unavailable)
    run_dir = persist_result(trained_result, tmp_path, project_root=tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["git"] is None
    assert manifest["status"] == "complete"
