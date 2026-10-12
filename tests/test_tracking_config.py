"""Configuración coherente y exclusivamente local de cliente y UI."""

from pathlib import Path

import pytest

from modelops.tracking.config import TrackingConfig, resolve_config, ui_command


def test_absolute_uris_and_ui_use_same_locations(tmp_path: Path) -> None:
    config = resolve_config(tmp_path / "storage with spaces")
    assert config.database_path.is_absolute()
    assert config.artifact_dir.is_absolute()
    assert config.tracking_uri == f"sqlite:///{config.database_path.as_posix()}"
    assert config.artifact_uri == config.artifact_dir.as_uri()
    command = ui_command(config, 5010)
    for flag in ("--backend-store-uri", "--registry-store-uri"):
        assert command[command.index(flag) + 1] == config.tracking_uri
    assert command[command.index("--default-artifact-root") + 1] == config.artifact_uri
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert "--no-serve-artifacts" in command


def test_environment_configuration_and_argument_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODELOPS_MLFLOW_DIR", str(tmp_path / "env-root"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'env.db'}")
    monkeypatch.setenv("MODELOPS_MLFLOW_ARTIFACT_DIR", str(tmp_path / "env-artifacts"))
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "from-env")
    config = resolve_config()
    assert config.database_path == tmp_path / "env.db"
    assert config.artifact_dir == tmp_path / "env-artifacts"
    assert config.experiment_name == "from-env"
    explicit = resolve_config(
        tmp_path / "explicit-root",
        tracking_uri=f"sqlite:///{tmp_path / 'explicit.db'}",
        artifact_dir=tmp_path / "explicit-artifacts",
        experiment_name="from-cli",
    )
    assert explicit.database_path == tmp_path / "explicit.db"
    assert explicit.artifact_dir == tmp_path / "explicit-artifacts"
    assert explicit.experiment_name == "from-cli"


@pytest.mark.parametrize(
    "uri",
    [
        "http://localhost:5000",
        "postgresql://localhost/db",
        "sqlite:///:memory:",
        "sqlite:///",
        "",
    ],
)
def test_reject_non_persistent_or_non_sqlite_backend(uri: str) -> None:
    with pytest.raises(ValueError):
        resolve_config(tracking_uri=uri)


@pytest.mark.parametrize("port", [0, -1, 65536, True])
def test_invalid_ui_port(port: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="puerto"):
        ui_command(TrackingConfig(tmp_path / "db", tmp_path / "artifacts"), port)


@pytest.mark.parametrize("prefix", ["", "../model", "model/name", "prod alias"])
def test_invalid_registry_prefix(prefix: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prefijo"):
        TrackingConfig(tmp_path / "db", tmp_path / "artifacts", model_prefix=prefix)
