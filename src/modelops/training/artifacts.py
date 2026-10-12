"""Persistencia local; no participa en la lógica de entrenamiento/evaluación."""

import hashlib
import json
import platform
import subprocess
import tomllib
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from uuid import uuid4

import joblib

from modelops.training.runner import TrainingResult


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_metadata(project_root: Path | None) -> dict[str, object] | None:
    if project_root is None:
        return None
    try:
        top = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        # No atribuir a ModelOps la revisión de un repositorio ajeno.
        if Path(top).resolve() != project_root.resolve():
            return None
        project = project_root / "pyproject.toml"
        if not project.is_file():
            return None
        if tomllib.loads(project.read_text())["project"]["name"] != "modelops":
            return None
        revision = subprocess.run(
            ["git", "-C", top, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", top, "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        return {"revision": revision, "dirty": bool(status)}
    except (OSError, subprocess.SubprocessError, KeyError, ValueError):
        return None


def persist_result(
    result: TrainingResult, output_dir: Path, *, project_root: Path | None = None
) -> Path:
    """Crear una ejecución exclusiva; manifest.json marca su finalización.

    Una escritura fallida conserva el directorio parcial, sin manifiesto completo.
    Joblib solo debe cargarse desde fuentes fiables y con versiones compatibles.
    """
    created_at = datetime.now(UTC)
    run_id = f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / run_id
    run_dir.mkdir(exist_ok=False)
    models_dir = run_dir / "models"
    models_dir.mkdir()

    model_config = {}
    for name, model in result.models.items():
        with (models_dir / f"{name}.joblib").open("xb") as stream:
            joblib.dump(model, stream, compress=3)
        transformation = model.named_steps["preprocess"].transformers[0][1]
        model_config[name] = {
            "estimator": type(model.named_steps["model"]).__name__,
            "parameters": model.named_steps["model"].get_params(deep=False),
            "preprocessing": {
                "numeric": (
                    {
                        "type": type(transformation).__name__,
                        "parameters": transformation.get_params(deep=False),
                    }
                    if transformation != "passthrough"
                    else {"type": "passthrough"}
                ),
                "feature_names": list(result.dataset.feature_names),
                "remainder": "drop",
            },
        }
    _write_json(
        run_dir / "config.json",
        {
            **asdict(result.config),
            "dataset": result.dataset.name,
            "positive_label": 1,
            "prediction_rule": "estimator.predict (binary probability > 0.5)",
            "native_thread_limit": 1,
            "selection": "none; fixed candidates evaluated once on held-out test",
            "models": model_config,
        },
    )
    _write_json(run_dir / "metrics.json", result.metrics)
    _write_json(
        run_dir / "split.json",
        {
            "strategy": "stratified_shuffle_split",
            "train_positions": result.split.train_positions.tolist(),
            "test_positions": result.split.test_positions.tolist(),
        },
    )
    dataset = result.dataset
    files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": run_id,
        "created_at_utc": created_at.isoformat(),
        "seed": result.config.seed,
        "dataset": {
            "id": dataset.name,
            "sha256": result.dataset_sha256,
            "fingerprint_algorithm": (
                "sha256(schema_json + pandas_row_hashes_le_uint64)"
            ),
            "rows": len(dataset.features),
            "feature_names": list(dataset.feature_names),
            "feature_dtypes": [str(dtype) for dtype in dataset.features.dtypes],
            "target_name": dataset.target_name,
            "target_dtype": str(dataset.target.dtype),
            "class_names": dict(enumerate(dataset.target_names)),
            "positive_label": 1,
        },
        "split": {
            "train_rows": len(result.split.x_train),
            "test_rows": len(result.split.x_test),
            "train_class_counts": result.split.y_train.value_counts().to_dict(),
            "test_class_counts": result.split.y_test.value_counts().to_dict(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dependencies": {
                dist.metadata["Name"]: dist.version
                for dist in distributions()
                if dist.metadata["Name"]
            },
        },
        "git": _git_metadata(project_root),
        "project_files_sha256": {
            name: _sha256(project_root / name)
            for name in ("pyproject.toml", "uv.lock", ".python-version")
            if project_root is not None and (project_root / name).is_file()
        },
        "artifacts_sha256": {
            path.relative_to(run_dir).as_posix(): _sha256(path) for path in files
        },
    }
    # Escribir al final: los consumidores deben exigir este marcador de éxito.
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir
