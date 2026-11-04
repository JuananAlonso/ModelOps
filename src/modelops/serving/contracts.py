"""Contrato v1: nombres originales, números finitos y batches acotados."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from modelops.tabular import DEMO_FEATURE_NAMES

MAX_BATCH_SIZE = 256
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
FeatureRow = create_model(
    "FeatureRow",
    __config__=ConfigDict(extra="forbid"),
    **{
        name.replace(" ", "_"): (FiniteNumber, Field(alias=name))
        for name in DEMO_FEATURE_NAMES
    },
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    instances: list[FeatureRow] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class ModelIdentity(BaseModel):
    name: str
    version: str
    uri: str


class Prediction(BaseModel):
    label: Literal[0, 1]
    probabilities: list[FiniteNumber]


class PredictResponse(BaseModel):
    schema_version: Literal["1"] = "1"
    model: ModelIdentity
    classes: list[int]
    class_names: list[str]
    predictions: list[Prediction]
