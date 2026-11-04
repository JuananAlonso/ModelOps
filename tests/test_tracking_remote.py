"""Frontera HTTP con cliente explícito; no necesita abrir conexiones."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from modelops.tracking import adapter
from modelops.tracking.config import resolve_config, ui_command


def test_remote_configuration_never_prepares_local_storage(tmp_path: Path) -> None:
    config = resolve_config(tmp_path / "unused", tracking_uri="http://mlflow:5000/")
    config.prepare()
    assert config.tracking_uri == "http://mlflow:5000"
    assert config.artifact_uri == "mlflow-artifacts:/"
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="UI remota"):
        ui_command(config)


@pytest.mark.parametrize("uri", ["http://", "https://host/?token=x", "http://u:p@host"])
def test_remote_endpoint_rejects_ambiguous_or_embedded_credentials(uri: str) -> None:
    with pytest.raises(ValueError):
        resolve_config(tracking_uri=uri)


def test_remote_rejects_local_artifact_override(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="servidor MLflow"):
        resolve_config(tracking_uri="https://host", artifact_dir=tmp_path)


def test_remote_download_uses_explicit_endpoint_and_cleans_temporary_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = resolve_config(tracking_uri="http://mlflow:5000")
    source = "mlflow-artifacts:/run/artifacts/pipeline"
    client = Mock()
    client.get_model_version_download_uri.return_value = source
    monkeypatch.setattr(adapter, "make_client", lambda _: client)
    destinations = []

    def download(**kwargs: object) -> str:
        assert kwargs["tracking_uri"] == config.tracking_uri
        assert kwargs["registry_uri"] == config.tracking_uri
        assert kwargs["artifact_uri"] == source
        path = Path(kwargs["dst_path"]) / "pipeline"
        path.mkdir()
        destinations.append(path)
        return str(path)

    monkeypatch.setattr(adapter, "download_artifacts", download)
    pipeline = object()
    monkeypatch.setattr(adapter.mlflow.sklearn, "load_model", lambda _: pipeline)
    assert (
        adapter.load_registered_pipeline(config, "modelops-baseline", "2") is pipeline
    )
    client.get_model_version_download_uri.assert_called_once_with(
        "modelops-baseline", "2"
    )
    assert not destinations[0].exists()
    client.get_model_version_download_uri.return_value = "file:///other-host/pipeline"
    with pytest.raises(ValueError, match="servir artefactos"):
        adapter.load_registered_pipeline(config, "modelops-baseline", "2")
