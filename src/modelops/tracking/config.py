"""Cliente SQLite local o HTTP remoto; wrapper de UI exclusivamente local."""

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class TrackingConfig:
    database_path: Path
    artifact_dir: Path
    experiment_name: str = "modelops-tabular"
    model_prefix: str = "modelops"
    remote_tracking_uri: str | None = None

    def __post_init__(self) -> None:
        if self.remote_tracking_uri is not None:
            parsed = urlsplit(self.remote_tracking_uri)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "El endpoint MLflow requiere HTTP(S), sin credenciales."
                )
        object.__setattr__(
            self, "database_path", self.database_path.expanduser().resolve()
        )
        object.__setattr__(
            self, "artifact_dir", self.artifact_dir.expanduser().resolve()
        )
        if not self.experiment_name.strip():
            raise ValueError("El experimento debe tener un nombre no vacío.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", self.model_prefix):
            raise ValueError(
                "El prefijo del Registry debe ser alfanumérico, sin rutas."
            )
        if any(char in str(self.database_path) for char in ("?", "#")):
            raise ValueError("La ruta de SQLite no puede contener ? ni #.")
        if self.database_path == self.artifact_dir:
            raise ValueError(
                "SQLite y el directorio de artefactos deben ser distintos."
            )

    @property
    def tracking_uri(self) -> str:
        return self.remote_tracking_uri or f"sqlite:///{self.database_path.as_posix()}"

    @property
    def artifact_uri(self) -> str:
        return (
            "mlflow-artifacts:/"
            if self.remote_tracking_uri
            else self.artifact_dir.as_uri()
        )

    def prepare(self) -> None:
        if self.remote_tracking_uri:
            return
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


def resolve_config(
    storage_dir: Path | None = None,
    *,
    tracking_uri: str | None = None,
    artifact_dir: Path | None = None,
    experiment_name: str | None = None,
    model_prefix: str = "modelops",
) -> TrackingConfig:
    """Prioridad: argumento explícito, variable de entorno y valor por defecto.

    SQLite local o servidor HTTP(S) con proxy de artefactos de MLflow.
    Las rutas relativas se interpretan respecto al directorio de invocación.
    """
    root = storage_dir or Path(
        os.environ.get("MODELOPS_MLFLOW_DIR", "artifacts/mlflow")
    )
    uri = tracking_uri if tracking_uri is not None else os.getenv("MLFLOW_TRACKING_URI")
    if uri is not None and uri.startswith(("http://", "https://")):
        if artifact_dir is not None or os.getenv("MODELOPS_MLFLOW_ARTIFACT_DIR"):
            raise ValueError("En HTTP los artefactos los gestiona el servidor MLflow.")
        return TrackingConfig(
            root / "mlflow.db",
            root / "artifacts",
            experiment_name
            if experiment_name is not None
            else os.getenv("MLFLOW_EXPERIMENT_NAME", "modelops-tabular"),
            model_prefix,
            remote_tracking_uri=uri.rstrip("/"),
        )
    if uri is None:
        database_path = root / "mlflow.db"
    else:
        if not uri.startswith("sqlite:///"):
            raise ValueError("Se requiere un tracking URI SQLite local o HTTP(S).")
        database = uri.removeprefix("sqlite:///")
        if not database or database == ":memory:":
            raise ValueError("SQLite debe usar un archivo persistente, no :memory:.")
        database_path = Path(database)
    if artifact_dir is None:
        configured_artifacts = os.getenv("MODELOPS_MLFLOW_ARTIFACT_DIR")
        artifact_dir = (
            Path(configured_artifacts) if configured_artifacts else root / "artifacts"
        )
    if experiment_name is None:
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "modelops-tabular")
    return TrackingConfig(database_path, artifact_dir, experiment_name, model_prefix)


def ui_command(config: TrackingConfig, port: int = 5000) -> list[str]:
    """UI/servidor local con el mismo backend y acceso directo a artefactos."""
    if config.remote_tracking_uri:
        raise ValueError("La UI remota ya la sirve el endpoint MLflow configurado.")
    if isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("El puerto debe estar entre 1 y 65535.")
    return [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        config.tracking_uri,
        "--registry-store-uri",
        config.tracking_uri,
        "--default-artifact-root",
        config.artifact_uri,
        "--no-serve-artifacts",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
    ]
