"""Adaptador explícito de TrainingResult a MLflow, sin estado de runs global."""

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from modelops.tracking.config import TrackingConfig

if TYPE_CHECKING:
    from modelops.training.runner import TrainingResult


@dataclass(frozen=True)
class RegisteredCandidate:
    name: str
    version: str
    run_id: str
    run_uri: str
    source_uri: str
    model_uri: str


@dataclass(frozen=True)
class TrackedExecution:
    experiment_id: str
    parent_run_id: str
    execution_id: str
    candidates: dict[str, RegisteredCandidate]


def make_client(config: TrackingConfig) -> MlflowClient:
    config.prepare()
    return MlflowClient(
        tracking_uri=config.tracking_uri, registry_uri=config.tracking_uri
    )


def ensure_experiment(client: MlflowClient, config: TrackingConfig) -> str:
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        try:
            return client.create_experiment(
                config.experiment_name, artifact_location=config.artifact_uri
            )
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
            experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None or experiment.lifecycle_stage != "active":
        raise ValueError("El experimento no está activo; usa otro nombre o restáuralo.")
    if experiment.artifact_location.rstrip("/") != config.artifact_uri.rstrip("/"):
        raise ValueError(
            "El experimento ya existe con otro artifact location. "
            "Usa su directorio original o un nombre de experimento distinto."
        )
    return experiment.experiment_id


def _read_metadata(result: "TrainingResult", run_dir: Path) -> tuple[dict, dict]:
    """Exigir correspondencia con el resultado y verificar los archivos locales."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    if (
        manifest.get("status") != "complete"
        or manifest["run_id"] != run_dir.name
        or manifest["seed"] != result.config.seed
        or manifest["dataset"]["sha256"] != result.dataset_sha256
        or config["seed"] != result.config.seed
        or config["test_size"] != result.config.test_size
        or set(config["models"]) != set(result.models)
        or metrics != result.metrics
    ):
        raise ValueError("Los artefactos locales no corresponden a TrainingResult.")
    if set(result.models) != {"baseline", "lightgbm"}:
        raise ValueError("Se esperan los candidatos baseline y lightgbm.")
    for relative, expected in manifest["artifacts_sha256"].items():
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ValueError("El manifiesto contiene una ruta fuera de la ejecución.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"El artefacto {relative} no coincide con su checksum.")
    return manifest, config


def _parameter(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _common_tags(manifest: dict) -> dict[str, str]:
    tags = {
        "modelops.execution_id": manifest["run_id"],
        "modelops.dataset_sha256": manifest["dataset"]["sha256"],
        "modelops.positive_label": "1",
        "modelops.python": manifest["environment"]["python"],
    }
    if manifest.get("git"):
        tags["modelops.git_revision"] = manifest["git"]["revision"]
        tags["modelops.git_dirty"] = str(manifest["git"]["dirty"]).lower()
    return tags


def track_result(
    result: "TrainingResult", local_run_dir: Path, config: TrackingConfig
) -> TrackedExecution:
    """Registrar un padre, hijos diferenciados y nuevas versiones sin promoción.

    Las escrituras entre runs, archivos y Registry no son una transacción única.
    Los fallos marcan los runs FAILED; versiones ya creadas quedan para inspección.
    """
    manifest, effective = _read_metadata(result, local_run_dir)
    client = make_client(config)
    experiment_id = ensure_experiment(client, config)
    tags = _common_tags(manifest)
    parent = client.create_run(
        experiment_id,
        tags={
            **tags,
            "mlflow.runName": manifest["run_id"],
            "modelops.kind": "training",
        },
    )
    parent_id = parent.info.run_id
    shared_params = {
        "seed": result.config.seed,
        "test_size": result.config.test_size,
        "dataset_id": result.dataset.name,
        "dataset_sha256": result.dataset_sha256,
        "positive_label": 1,
        "python_version": manifest["environment"]["python"],
        **{
            f"version.{package}": version(package)
            for package in (
                "mlflow",
                "modelops",
                "numpy",
                "pandas",
                "scikit-learn",
                "lightgbm",
                "scipy",
                "joblib",
                "cloudpickle",
                "threadpoolctl",
            )
        },
    }
    candidates = {}
    try:
        client.log_batch(
            parent_id,
            params=[
                Param(key, _parameter(value)) for key, value in shared_params.items()
            ],
            synchronous=True,
        )
        # Conservar el manifiesto íntegro, incluyendo todos sus checksums originales.
        client.log_artifacts(parent_id, str(local_run_dir), artifact_path="training")
        example = result.split.x_train.iloc[:5].copy()
        requirements = [
            f"{package}=={version(package)}"
            for package in (
                "mlflow",
                "numpy",
                "pandas",
                "scikit-learn",
                "lightgbm",
                "scipy",
                "joblib",
                "cloudpickle",
                "threadpoolctl",
            )
        ]
        for candidate, pipeline in result.models.items():
            child = client.create_run(
                experiment_id,
                tags={
                    **tags,
                    "mlflow.parentRunId": parent_id,
                    "mlflow.runName": f"{candidate}-seed-{result.config.seed}",
                    "modelops.kind": "candidate",
                    "modelops.candidate": candidate,
                },
            )
            child_id = child.info.run_id
            try:
                parameters = {
                    **shared_params,
                    "candidate": candidate,
                    **{
                        f"model.{key}": value
                        for key, value in effective["models"][candidate][
                            "parameters"
                        ].items()
                    },
                }
                metric_values = result.metrics[candidate]
                undefined = [
                    key for key, value in metric_values.items() if value is None
                ]
                timestamp = int(time.time() * 1000)
                metrics = []
                for key, value in metric_values.items():
                    if value is not None:
                        if not math.isfinite(value):
                            raise ValueError(
                                f"La métrica {key} debe ser finita o None."
                            )
                        metrics.append(Metric(key, float(value), timestamp, 0))
                client.log_batch(
                    child_id,
                    params=[
                        Param(key, _parameter(value))
                        for key, value in parameters.items()
                    ],
                    metrics=metrics,
                    synchronous=True,
                )
                client.set_tag(
                    child_id, "modelops.undefined_metrics", json.dumps(undefined)
                )
                client.log_dict(child_id, metric_values, "metrics.json")
                client.log_dict(
                    child_id, effective["models"][candidate], "model-config.json"
                )
                signature = infer_signature(example, pipeline.predict(example))
                with TemporaryDirectory(prefix="modelops-mlflow-model-") as temporary:
                    model_dir = Path(temporary) / "pipeline"
                    mlflow.sklearn.save_model(
                        pipeline,
                        path=str(model_dir),
                        serialization_format="cloudpickle",
                        signature=signature,
                        input_example=example,
                        pip_requirements=requirements,
                        metadata={
                            "candidate": candidate,
                            "dataset_sha256": result.dataset_sha256,
                            "positive_label": 1,
                            "source_run_id": child_id,
                        },
                    )
                    client.log_artifacts(
                        child_id, str(model_dir), artifact_path="pipeline"
                    )
                name = f"{config.model_prefix}-{candidate}"
                try:
                    client.create_registered_model(name)
                except MlflowException as error:
                    if error.error_code != "RESOURCE_ALREADY_EXISTS":
                        raise
                source = f"{child.info.artifact_uri.rstrip('/')}/pipeline"
                registered = client.create_model_version(
                    name=name,
                    source=source,
                    run_id=child_id,
                    tags={
                        **tags,
                        "modelops.candidate": candidate,
                        "modelops.parent_run_id": parent_id,
                    },
                    description=(
                        "Pipeline completo del ejemplo técnico; "
                        "sin promoción a producción."
                    ),
                    await_creation_for=30,
                )
                if registered.status != "READY":
                    raise ValueError(
                        f"La versión {name}/{registered.version} no está READY."
                    )
                reference = RegisteredCandidate(
                    name=name,
                    version=str(registered.version),
                    run_id=child_id,
                    run_uri=f"runs:/{child_id}/pipeline",
                    source_uri=source,
                    model_uri=f"models:/{name}/{registered.version}",
                )
                candidates[candidate] = reference
                client.set_tag(child_id, "modelops.registry_name", name)
                client.set_tag(child_id, "modelops.registry_version", reference.version)
                client.log_dict(child_id, asdict(reference), "registry.json")
                client.set_terminated(child_id, status="FINISHED")
            except Exception:
                client.set_terminated(child_id, status="FAILED")
                raise
        tracked = TrackedExecution(
            experiment_id, parent_id, manifest["run_id"], candidates
        )
        client.log_dict(parent_id, asdict(tracked), "tracking.json")
        client.set_terminated(parent_id, status="FINISHED")
        return tracked
    except Exception:
        client.set_terminated(parent_id, status="FAILED")
        raise


def load_registered_pipeline(
    config: TrackingConfig, name: str, model_version: str
) -> Pipeline:
    """Resolver una versión explícita mediante el cliente y cargar el flavor nativo.

    La fuente se resuelve en el backend indicado, sin cambiar tracking URI global.
    Solo deben cargarse modelos de procedencia fiable.
    """
    if not str(model_version).isdigit() or int(model_version) < 1:
        raise ValueError("Se requiere una versión numérica explícita, sin alias.")
    source = make_client(config).get_model_version_download_uri(
        name, str(model_version)
    )
    return mlflow.sklearn.load_model(source)


def compare_runs(config: TrackingConfig) -> list[dict]:
    """Comparar candidatos de padres completos, entre hasta 1000 runs recientes."""
    client = make_client(config)
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        raise ValueError(
            "El experimento todavía no existe; ejecuta entrenamiento primero."
        )
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    rows = []
    parent_status = {}
    for run in runs:
        if run.info.status != "FINISHED" or "modelops.candidate" not in run.data.tags:
            continue
        parent = run.data.tags["mlflow.parentRunId"]
        if parent not in parent_status:
            parent_status[parent] = client.get_run(parent).info.status
        if parent_status[parent] != "FINISHED":
            continue
        rows.append(
            {
                "run_id": run.info.run_id,
                "parent_run_id": parent,
                "execution_id": run.data.tags["modelops.execution_id"],
                "candidate": run.data.tags["modelops.candidate"],
                "seed": int(run.data.params["seed"]),
                "test_size": float(run.data.params["test_size"]),
                "dataset_sha256": run.data.params["dataset_sha256"],
                "metrics": {
                    **run.data.metrics,
                    **dict.fromkeys(
                        json.loads(run.data.tags["modelops.undefined_metrics"])
                    ),
                },
                "registered_name": run.data.tags["modelops.registry_name"],
                "registered_version": run.data.tags["modelops.registry_version"],
            }
        )
    return rows
