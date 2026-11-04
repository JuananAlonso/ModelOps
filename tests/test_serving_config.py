"""Configuración de serving explícita y CLI sin abrir sockets."""

import subprocess
import sys
from pathlib import Path

import pytest

from modelops.serving import cli
from modelops.serving.config import ServingConfig, resolve_serving_config
from modelops.tracking.config import TrackingConfig


@pytest.fixture
def tracking(tmp_path: Path) -> TrackingConfig:
    return TrackingConfig(tmp_path / "mlflow.db", tmp_path / "artifacts")


@pytest.mark.parametrize(
    "version", ["", "latest", "champion", "0", "-1", "1.0", "01", 1]
)
def test_version_must_be_explicit(tracking: TrackingConfig, version: object) -> None:
    with pytest.raises(ValueError, match="versión numérica explícita"):
        ServingConfig(tracking, "modelops-baseline", version)


@pytest.mark.parametrize("name", ["", "../baseline", "with spaces", None])
def test_invalid_model_name(tracking: TrackingConfig, name: object) -> None:
    with pytest.raises(ValueError, match="nombre del modelo"):
        ServingConfig(tracking, name, "1")


def test_environment_and_explicit_arguments(
    tracking: TrackingConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODELOPS_MODEL_NAME", "modelops-lightgbm")
    monkeypatch.setenv("MODELOPS_MODEL_VERSION", "2")
    assert resolve_serving_config(tracking=tracking) == ServingConfig(
        tracking, "modelops-lightgbm", "2"
    )
    assert resolve_serving_config(
        tracking=tracking, model_name="modelops-baseline", model_version="1"
    ) == ServingConfig(tracking, "modelops-baseline", "1")
    with pytest.raises(ValueError):
        resolve_serving_config(tracking=tracking, model_version="")


def test_missing_version_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODELOPS_MODEL_VERSION", raising=False)
    with pytest.raises(SystemExit) as error:
        cli.main([])
    assert error.value.code == 2


@pytest.mark.parametrize("port", ["0", "65536"])
def test_invalid_port(port: str) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--model-version", "1", "--port", port])
    assert error.value.code == 2


def test_cli_defaults_to_loopback_and_never_loads_before_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))
    assert cli.main(["--model-version", "3", "--storage-dir", str(tmp_path)]) == 0
    assert calls[0][1] == {"host": "127.0.0.1", "port": 8000}
    assert calls[0][0][0].state.predictor is None
    assert not list(tmp_path.iterdir())


def test_factory_import_does_not_import_training(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "from modelops.serving.app import create_app; import sys; "
            "assert not any(m.startswith('modelops.training') for m in sys.modules)",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
