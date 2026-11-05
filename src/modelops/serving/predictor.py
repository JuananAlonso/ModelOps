"""Reutilizar el pipeline ajustado y verificar la salida antes de serializar."""

from threading import Lock

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from modelops.model_contract import pipeline_contract
from modelops.serving.contracts import (
    ModelIdentity,
    Prediction,
    PredictResponse,
    RegressionPrediction,
    RegressionPredictResponse,
)


class Predictor:
    def __init__(self, pipeline: Pipeline) -> None:
        if not isinstance(pipeline, Pipeline):
            raise ValueError("Se requiere un Pipeline de sklearn ajustado.")
        self.contract = pipeline_contract(pipeline)
        classes = (
            np.asarray(pipeline.classes_)
            if self.contract.task == "classification"
            else np.array([], dtype=int)
        )
        if self.contract.task == "classification" and (
            classes.shape != (len(self.contract.class_names),)
            or classes.dtype.kind not in "iu"
            or set(classes.tolist()) != set(range(len(self.contract.class_names)))
        ):
            raise ValueError(
                "Las clases del modelo deben coincidir con los códigos del contrato."
            )
        self.classes = classes.tolist()
        self.pipeline = pipeline
        self._lock = Lock()
        # Detecta modelos cargables pero inutilizables antes de anunciar readiness.
        self.predict(
            pd.DataFrame(
                [
                    [
                        "" if name in self.contract.categorical_features else 0.0
                        for name in self.contract.feature_names
                    ]
                ],
                columns=self.contract.feature_names,
            )
        )

    def predict(
        self, features: pd.DataFrame
    ) -> list[Prediction] | list[RegressionPrediction]:
        features = self.contract.validate_features(features)
        # Serializa acceso por proceso y limita pools nativos para lotes pequeños.
        with self._lock, threadpool_limits(limits=1):
            labels = np.asarray(self.pipeline.predict(features))
            if self.contract.task == "regression":
                if (
                    labels.shape != (len(features),)
                    or labels.dtype.kind not in "iuf"
                    or not np.isfinite(labels).all()
                ):
                    raise ValueError(
                        "El modelo ha producido una salida de regresión inválida."
                    )
                return [RegressionPrediction(value=float(value)) for value in labels]
            probabilities = np.asarray(self.pipeline.predict_proba(features))
        if (
            labels.shape != (len(features),)
            or labels.dtype.kind not in "iu"
            or not np.isin(labels, self.classes).all()
            or probabilities.shape != (len(features), len(self.classes))
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-8)
        ):
            raise ValueError("El modelo ha producido una salida inválida.")
        return [
            Prediction(label=int(label), probabilities=probability.tolist())
            for label, probability in zip(labels, probabilities, strict=True)
        ]

    def response(
        self, features: pd.DataFrame, identity: ModelIdentity
    ) -> PredictResponse | RegressionPredictResponse:
        predictions = self.predict(features)
        if self.contract.task == "regression":
            return RegressionPredictResponse(model=identity, predictions=predictions)
        return PredictResponse(
            model=identity,
            classes=self.classes,
            class_names=[self.contract.class_names[label] for label in self.classes],
            predictions=predictions,
        )
