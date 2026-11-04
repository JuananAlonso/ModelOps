"""Selección explícita de modelo; no hay latest ni promoción implícita."""

import os
import re
from dataclasses import dataclass

from modelops.tracking.config import TrackingConfig, resolve_config


@dataclass(frozen=True)
class ServingConfig:
    tracking: TrackingConfig
    model_name: str
    model_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", self.model_name
        ):
            raise ValueError("El nombre del modelo debe ser alfanumérico, sin rutas.")
        if not isinstance(self.model_version, str) or not re.fullmatch(
            r"[1-9][0-9]*", self.model_version
        ):
            raise ValueError("Se requiere una versión numérica explícita, sin alias.")


def resolve_serving_config(
    *,
    tracking: TrackingConfig | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
) -> ServingConfig:
    return ServingConfig(
        tracking=tracking if tracking is not None else resolve_config(),
        model_name=(
            model_name
            if model_name is not None
            else os.getenv("MODELOPS_MODEL_NAME", "modelops-baseline")
        ),
        model_version=(
            model_version
            if model_version is not None
            else os.getenv("MODELOPS_MODEL_VERSION", "")
        ),
    )
