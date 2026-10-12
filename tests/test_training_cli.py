"""Entrada ejecutable, errores de configuración y flujo integrado sin servicios."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from modelops.training.cli import main


def test_integrated_cli_from_outside_repository(tmp_path: Path) -> None:
    output_dir = tmp_path / "runs"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "modelops.training.cli",
            "--seed",
            "19",
            "--test-size",
            "0.25",
            "--output-dir",
            str(output_dir),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    summary = json.loads(completed.stdout)
    run_dir = Path(summary["run_dir"])
    assert run_dir.parent == output_dir
    assert set(summary["metrics"]) == {"baseline", "lightgbm"}
    config = json.loads((run_dir / "config.json").read_text())
    assert config["seed"] == 19
    assert config["test_size"] == 0.25
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["git"] is None
    assert set(path.name for path in run_dir.iterdir()) == {
        "config.json",
        "metrics.json",
        "manifest.json",
        "split.json",
        "models",
    }


@pytest.mark.parametrize(
    "arguments", [["--seed", "-1"], ["--seed", "abc"], ["--test-size", "nan"]]
)
def test_cli_rejects_invalid_configuration(
    arguments: list[str], tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    output = tmp_path / "runs"
    with pytest.raises(SystemExit) as error:
        main([*arguments, "--output-dir", str(output)])
    assert error.value.code == 2
    assert capsys.readouterr().err
    assert not output.exists()
