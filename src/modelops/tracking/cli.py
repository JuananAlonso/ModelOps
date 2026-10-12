"""Entradas de tracking, consulta y UI; reutilizan el entrenamiento de MOP-002."""

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from mlflow.exceptions import MlflowException

from modelops.tracking.adapter import compare_runs, make_client, track_result
from modelops.tracking.config import TrackingConfig, resolve_config, ui_command
from modelops.training.artifacts import persist_result
from modelops.training.config import TrainingConfig
from modelops.training.data import load_demo_dataset
from modelops.training.runner import train_dataset


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--storage-dir", type=Path, default=None)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--model-prefix", default="modelops")
    return parser


def _config(args: argparse.Namespace) -> TrackingConfig:
    return resolve_config(
        args.storage_dir,
        tracking_uri=args.tracking_uri,
        artifact_dir=args.artifact_dir,
        experiment_name=args.experiment_name,
        model_prefix=args.model_prefix,
    )


def train_main(argv: list[str] | None = None) -> int:
    parser = _parser(
        "Entrenar y registrar ambos pipelines en MLflow local, sin uso clínico."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training"))
    args = parser.parse_args(argv)
    try:
        config = _config(args)
        training = TrainingConfig(args.seed, args.test_size)
        result = train_dataset(load_demo_dataset(), training)
        local_run_dir = persist_result(result, args.output_dir, project_root=Path.cwd())
        tracked = track_result(result, local_run_dir, config)
    except (ValueError, OSError, MlflowException) as error:
        parser.exit(2, f"Error de tracking: {error}\n")
    print(
        json.dumps(
            {
                "local_run_dir": str(local_run_dir),
                "tracking_uri": config.tracking_uri,
                "artifact_uri": config.artifact_uri,
                "experiment_name": config.experiment_name,
                **asdict(tracked),
            },
            indent=2,
        )
    )
    return 0


def runs_main(argv: list[str] | None = None) -> int:
    parser = _parser("Consultar y comparar runs candidatos del experimento local.")
    args = parser.parse_args(argv)
    try:
        rows = compare_runs(_config(args))
    except (ValueError, OSError, MlflowException) as error:
        parser.exit(2, f"Error de consulta: {error}\n")
    print(json.dumps(rows, indent=2))
    return 0


def ui_main(argv: list[str] | None = None) -> int:
    parser = _parser("Abrir la UI de MLflow exclusivamente en 127.0.0.1.")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args(argv)
    try:
        config = _config(args)
        command = ui_command(config, args.port)
        # Inicializar el mismo SQLite y detectar un artifact root incompatible.
        client = make_client(config)
        experiment = client.get_experiment_by_name(config.experiment_name)
        if experiment is not None and experiment.artifact_location.rstrip(
            "/"
        ) != config.artifact_uri.rstrip("/"):
            raise ValueError(
                "El artifact location del experimento no coincide con la UI."
            )
        print(
            f"UI: http://127.0.0.1:{args.port}\nBackend: {config.tracking_uri}",
            flush=True,
        )
        environment = {
            **os.environ,
            "MLFLOW_DISABLE_TELEMETRY": "true",
            "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": "false",
        }
        return subprocess.run(command, check=False, env=environment).returncode
    except (ValueError, OSError, MlflowException) as error:
        parser.exit(2, f"Error de UI: {error}\n")


if __name__ == "__main__":
    raise SystemExit(train_main())
