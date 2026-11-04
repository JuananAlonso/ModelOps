"""Entrenar → persistir → Registry SQLite → cargar → HTTP, sin servidor externo."""

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from modelops.serving.app import create_app
from modelops.serving.config import ServingConfig
from modelops.tracking.adapter import (
    load_registered_pipeline,
    make_client,
    track_result,
)
from modelops.tracking.config import TrackingConfig
from modelops.training.artifacts import persist_result
from modelops.training.runner import TrainingResult


def test_real_registry_http_parity_pinned_version_and_restart(
    trained_result: TrainingResult, tmp_path: Path
) -> None:
    tracking = TrackingConfig(tmp_path / "mlflow.db", tmp_path / "artifacts")
    local = persist_result(trained_result, tmp_path / "training")
    tracked = track_result(trained_result, local, tracking)
    features = trained_result.split.x_test
    rows = features.to_dict("records")
    registry = make_client(tracking)
    before = registry.search_runs([tracked.experiment_id])
    for candidate, reference in tracked.candidates.items():
        config = ServingConfig(tracking, reference.name, reference.version)
        recovered = load_registered_pipeline(
            tracking, reference.name, reference.version
        )
        for _ in range(2):
            with TestClient(create_app(config)) as client:
                assert client.get("/health/ready").status_code == 200
                response = client.post("/predict", json={"instances": rows})
                assert response.status_code == 200
                value = response.json()
                assert value["model"]["version"] == reference.version
                labels = [p["label"] for p in value["predictions"]]
                probabilities = [p["probabilities"] for p in value["predictions"]]
                for pipeline in (recovered, trained_result.models[candidate]):
                    np.testing.assert_array_equal(labels, pipeline.predict(features))
                    np.testing.assert_allclose(
                        probabilities,
                        pipeline.predict_proba(features),
                        rtol=1e-9,
                        atol=1e-12,
                    )
        assert len(registry.search_model_versions(f"name='{reference.name}'")) == 1
    after = registry.search_runs([tracked.experiment_id])
    assert [(r.info.run_id, r.info.status) for r in before] == [
        (r.info.run_id, r.info.status) for r in after
    ]
    missing = ServingConfig(tracking, "modelops-baseline", "999")
    with TestClient(create_app(missing)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/predict", json={"instances": rows[:1]}).status_code == 503
