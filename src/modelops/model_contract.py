"""Contrato versionado compartido por entrenamiento, CLI y HTTP."""

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from modelops.csv_data import numeric_values
from modelops.tabular import DEMO_FEATURE_NAMES, DEMO_TARGET_NAMES


class ModelContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1, 2] = 1
    feature_type: Literal["number", "mixed"] = "number"
    finite: Literal[True] = True
    allow_missing: bool = False
    allow_extra_columns: Literal[False] = False
    feature_names: tuple[str, ...]
    categorical_features: tuple[str, ...] = ()
    class_names: tuple[str, ...]
    target_name: str
    task: Literal["classification", "regression"] = "classification"

    @field_validator("feature_names", "class_names")
    @classmethod
    def unique_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name.strip() for name in value) or len(set(value)) != len(value):
            raise ValueError("El contrato requiere nombres únicos y no vacíos.")
        return value

    @model_validator(mode="after")
    def consistent_schema(self):
        if (
            not self.feature_names
            or not self.target_name.strip()
            or self.target_name in self.feature_names
        ):
            raise ValueError("El contrato requiere variables y un objetivo distinto.")
        if len(set(self.categorical_features)) != len(
            self.categorical_features
        ) or not set(self.categorical_features) <= set(self.feature_names):
            raise ValueError("Variables categóricas inválidas en el contrato.")
        if bool(self.categorical_features) != (self.feature_type == "mixed"):
            raise ValueError("El tipo de variables no coincide con el esquema.")
        if (self.task == "classification" and len(self.class_names) < 2) or (
            self.task == "regression" and self.class_names
        ):
            raise ValueError("Las clases no corresponden a la tarea del contrato.")
        if self.schema_version == 1 and (
            self.allow_missing
            or self.categorical_features
            or self.task != "classification"
            or len(self.class_names) != 2
        ):
            raise ValueError(
                "El contrato v1 requiere clasificación binaria numérica sin ausentes."
            )
        return self

    def order_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if not features.columns.is_unique:
            raise ValueError("Hay cabeceras duplicadas; usa nombres únicos.")
        missing = [name for name in self.feature_names if name not in features.columns]
        extra = [name for name in features.columns if name not in self.feature_names]
        if missing or extra:
            raise ValueError(
                f"Columnas ausentes: {missing}; columnas extra: {extra}. "
                "Usa exactamente las variables guardadas con esta versión."
            )
        return features.loc[:, list(self.feature_names)]

    def parse_csv_features(
        self, features: pd.DataFrame, decimal: str = "."
    ) -> pd.DataFrame:
        parsed = self.order_features(features).copy()
        for name in self.feature_names:
            if name in self.categorical_features:
                parsed[name] = (
                    parsed[name].astype(object).where(parsed[name].notna(), np.nan)
                )
                continue
            values = numeric_values(parsed[name], decimal)
            if (parsed[name].notna() & values.isna()).any():
                raise ValueError(
                    f"La columna {name!r} debe ser numérica; contiene texto inválido."
                )
            parsed[name] = values.astype(float)
        return self.validate_features(parsed)

    def validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        ordered = self.order_features(features).copy()
        if features.empty:
            raise ValueError("El lote está vacío; añade al menos una fila.")
        if not self.allow_missing and ordered.isna().any().any():
            raise ValueError("Hay ausentes; corrígelos antes de predecir.")
        for name in self.feature_names:
            values = ordered[name]
            if name in self.categorical_features:
                if not values.dropna().map(lambda value: isinstance(value, str)).all():
                    raise ValueError(f"La variable categórica {name!r} debe ser texto.")
                ordered[name] = values.astype(object).where(values.notna(), np.nan)
            else:
                # Columnas HTTP totalmente nulas llegan como object.
                if self.allow_missing and values.isna().all():
                    ordered[name] = np.nan
                    continue
                if values.dtype.kind not in "iuf":
                    raise ValueError(
                        f"La variable {name!r} debe ser numérica; corrige su tipo."
                    )
                if np.isinf(values.to_numpy(dtype=np.float64)).any():
                    raise ValueError(
                        "Hay valores no finitos; corrígelos antes de predecir."
                    )
        return ordered


def pipeline_contract(pipeline: object) -> ModelContract:
    value = getattr(pipeline, "modelops_contract_", None)
    if value is not None:
        contract = ModelContract.model_validate(value)
    elif tuple(pipeline.feature_names_in_) == DEMO_FEATURE_NAMES:
        contract = ModelContract(
            feature_names=DEMO_FEATURE_NAMES,
            class_names=DEMO_TARGET_NAMES,
            target_name="target",
        )
    else:
        raise ValueError(
            "El modelo no tiene contrato; vuelve a entrenarlo con modelops train."
        )
    if tuple(pipeline.feature_names_in_) != contract.feature_names:
        raise ValueError(
            "El esquema del pipeline no coincide con su contrato guardado."
        )
    return contract
