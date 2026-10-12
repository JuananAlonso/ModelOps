"""CLI integrada y compatibilidad del entrenamiento sin importar MLflow."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelops.tracking import cli


def test_tracking_cli_and_query(tmp_path: Path) -> None:
    storage = tmp_path / "tracking"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "modelops.tracking.cli",
            "--storage-dir",
            str(storage),
            "--output-dir",
            str(tmp_path / "local"),
            "--seed",
            "123",
            "--experiment-name",
            "cli-test",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    summary = json.loads(completed.stdout)
    assert summary["experiment_name"] == "cli-test"
    assert set(summary["candidates"]) == {"baseline", "lightgbm"}
    assert summary["tracking_uri"] == f"sqlite:///{storage / 'mlflow.db'}"
    query = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "from modelops.tracking.cli import runs_main; "
            "raise SystemExit(runs_main())",
            "--storage-dir",
            str(storage),
            "--experiment-name",
            "cli-test",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rows = json.loads(query.stdout)
    assert len(rows) == 2
    assert {row["seed"] for row in rows} == {123}


def test_local_training_works_when_mlflow_imports_are_forbidden(tmp_path: Path) -> None:
    script = """
import sys
from importlib.abc import MetaPathFinder
class ForbidMLflow(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlflow' or fullname.startswith('mlflow.'):
            raise ImportError('MLflow forbidden in core training')
sys.meta_path.insert(0, ForbidMLflow())
from modelops.training.cli import main
raise SystemExit(main(['--seed', '42', '--output-dir', 'local']))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert set(json.loads(completed.stdout)["metrics"]) == {"baseline", "lightgbm"}
    assert not (tmp_path / "mlflow.db").exists()


def test_ui_builds_a_loopback_only_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def capture(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", capture)
    assert cli.ui_main(["--storage-dir", str(tmp_path), "--port", "5011"]) == 0
    command, options = calls[0]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "5011"
    assert (
        command[command.index("--backend-store-uri") + 1]
        == f"sqlite:///{tmp_path / 'mlflow.db'}"
    )
    assert options["env"]["MLFLOW_DISABLE_TELEMETRY"] == "true"
    assert options["env"]["MLFLOW_SERVER_ENABLE_JOB_EXECUTION"] == "false"


def test_cli_rejects_remote_backend_before_training(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    local = tmp_path / "local"
    with pytest.raises(SystemExit) as error:
        cli.train_main(
            ["--tracking-uri", "https://example.invalid", "--output-dir", str(local)]
        )
    assert error.value.code == 2
    assert "sqlite" in capsys.readouterr().err
    assert not local.exists()
