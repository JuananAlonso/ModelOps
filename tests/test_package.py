"""Comprobaciones de instalacion e importacion del paquete."""

import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import modelops


def test_installed_package_metadata() -> None:
    """La distribucion instalada publica nombre y version coherentes."""
    installed = distribution("modelops")

    assert installed.metadata["Name"] == "modelops"
    assert installed.version == modelops.__version__
    assert modelops.__version__


def test_import_outside_repository(tmp_path: Path) -> None:
    """La importacion funciona sin depender del directorio o de PYTHONPATH."""
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import modelops; print(modelops.__version__)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == modelops.__version__
