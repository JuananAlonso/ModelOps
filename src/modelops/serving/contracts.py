"""Entradas tipadas según contrato y respuestas de clasificación o regresión."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from modelops.model_contract import ModelContract
from modelops.tabular import DEMO_FEATURE_NAMES

MAX_BATCH_SIZE = 256
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]


def make_request_model(
    feature_names: tuple[str, ...], contract: ModelContract | None = None
) -> type[BaseModel]:
    fields = {}
    for index, name in enumerate(feature_names):
        kind = (
            str if contract and name in contract.categorical_features else FiniteNumber
        )
        if contract and contract.allow_missing:
            kind = kind | None
        fields[f"column_{index}"] = (kind, Field(alias=name))
    row = create_model(
        "FeatureRow",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    return create_model(
        "PredictRequest",
        __config__=ConfigDict(extra="forbid", strict=True),
        instances=(list[row], Field(min_length=1, max_length=MAX_BATCH_SIZE)),
    )


PredictRequest = make_request_model(DEMO_FEATURE_NAMES)


class ModelIdentity(BaseModel):
    name: str
    version: str
    uri: str


class Prediction(BaseModel):
    label: Annotated[int, Field(strict=True, ge=0)]
    probabilities: list[FiniteNumber]


class PredictResponse(BaseModel):
    schema_version: Literal["1"] = "1"
    model: ModelIdentity
    classes: list[int]
    class_names: list[str]
    predictions: list[Prediction]


class RegressionPrediction(BaseModel):
    value: FiniteNumber


class RegressionPredictResponse(BaseModel):
    schema_version: Literal["2"] = "2"
    task: Literal["regression"] = "regression"
    model: ModelIdentity
    predictions: list[RegressionPrediction]
