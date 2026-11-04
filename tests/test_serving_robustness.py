"""Fallos reales de Registry y artefactos, selección fija y ausencia de escrituras."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest
from fastapi.testclient import TestClient

from modelops.serving.app import create_app
from modelops.serving.config import ServingConfig
from modelops.tracking.adapter import compare_runs, make_client, track_result
from modelops.tracking.config import TrackingConfig
from modelops.training.artifacts import persist_result
from modelops.training.config import TrainingConfig
from modelops.training.runner import TrainingResult, train_dataset


def test_training_runs_without_importing_or_connecting_to_mlflow(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; from modelops.training.runner import train_dataset; "
            "from modelops.training.data import load_demo_dataset; "
            "from modelops.training.config import TrainingConfig; "
            "result = train_dataset(load_demo_dataset(), TrainingConfig()); "
            "assert set(result.models) == {'baseline', 'lightgbm'}; "
            "assert not any(m == 'mlflow' or m.startswith('mlflow.') "
            "for m in sys.modules)",
        ],
        cwd=tmp_path,
        env={**os.environ, "MLFLOW_TRACKING_URI": "http://127.0.0.1:1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def register(result: TrainingResult, root: Path):
    tracking = TrackingConfig(root / "mlflow.db", root / "artifacts")
    local = persist_result(result, root / "training")
    tracked = track_result(result, local, tracking)
    return tracking, tracked.candidates["baseline"]


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_real_unusable_artifact_never_becomes_ready(
    trained_result: TrainingResult, tmp_path: Path, failure: str
) -> None:
    tracking, reference = register(trained_result, tmp_path)
    directory = Path(unquote(urlsplit(reference.source_uri).path))
    model = directory / "model.pkl"
    if failure == "missing":
        model.unlink()
    else:
        model.write_bytes(b"INVALID PICKLE - private storage details")
    with TestClient(create_app(ServingConfig(tracking, reference.name, "1"))) as api:
        assert api.get("/health/live").status_code == 200
        assert api.get("/health/ready").status_code == 503
        response = api.post(
            "/predict",
            json={"instances": trained_result.split.x_test.iloc[:1].to_dict("records")},
        )
        assert response.status_code == 503
        assert "private" not in response.text
        assert str(directory) not in response.text


def test_real_corrupt_backend_is_sanitized(tmp_path: Path) -> None:
    database = tmp_path / "private.db"
    database.write_bytes(b"not a database")
    config = ServingConfig(
        TrackingConfig(database, tmp_path / "artifacts"), "modelops-baseline", "1"
    )
    with TestClient(create_app(config)) as api:
        assert api.get("/health/live").status_code == 200
        response = api.get("/health/ready")
        assert response.status_code == 503
        assert str(database) not in response.text
        assert "Traceback" not in response.text


def test_pinned_version_ignores_newer_failed_execution_and_requests_do_not_write(
    trained_result: TrainingResult, tmp_path: Path
) -> None:
    tracking, reference = register(trained_result, tmp_path)
    config = ServingConfig(tracking, reference.name, "1")
    features = trained_result.split.x_test
    body = {"instances": features.to_dict("records")}
    client = make_client(tracking)
    with TestClient(create_app(config)) as api:
        first = api.post("/predict", json=body).json()
        newer = train_dataset(trained_result.dataset, TrainingConfig(seed=123))
        local = persist_result(newer, tmp_path / "training")
        tracked = track_result(newer, local, tracking)
        assert tracked.candidates["baseline"].version == "2"
        assert not np.allclose(
            newer.models["baseline"].predict_proba(features),
            trained_result.models["baseline"].predict_proba(features),
        )
        # READY describe el artefacto, no la aprobación de la ejecución.
        client.set_terminated(tracked.parent_run_id, status="FAILED")
        assert client.get_model_version(reference.name, "2").status == "READY"
        assert {row["registered_version"] for row in compare_runs(tracking)} == {"1"}

        def snapshot() -> dict:
            experiment = client.get_experiment_by_name(tracking.experiment_name)
            return {
                "files": {
                    str(path.relative_to(tracking.artifact_dir)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in tracking.artifact_dir.rglob("*")
                    if path.is_file()
                },
                "runs": {
                    run.info.run_id: (run.info.status, run.data.to_dictionary())
                    for run in client.search_runs([experiment.experiment_id])
                },
                "versions": sorted(
                    (v.name, v.version, v.run_id, v.status)
                    for v in client.search_model_versions()
                ),
            }

        before = snapshot()
        for _ in range(3):
            assert api.post("/predict", json=body).json() == first
            assert api.get("/health/ready").json()["model"]["version"] == "1"
        assert snapshot() == before
    # Un proceso nuevo sigue resolviendo exactamente la versión configurada.
    with TestClient(create_app(config)) as api:
        assert api.post("/predict", json=body).json() == first
