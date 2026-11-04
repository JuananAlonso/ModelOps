"""Reutilizar el pipeline ajustado y verificar la salida antes de serializar."""

from threading import Lock

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from modelops.serving.contracts import Prediction
from modelops.tabular import DEMO_FEATURE_NAMES


class Predictor:
    def __init__(self, pipeline: Pipeline) -> None:
        if not isinstance(pipeline, Pipeline):
            raise ValueError("Se requiere un Pipeline de sklearn ajustado.")
        if tuple(pipeline.feature_names_in_) != DEMO_FEATURE_NAMES:
            raise ValueError("El esquema del modelo no coincide con el contrato v1.")
        classes = np.asarray(pipeline.classes_)
        if (
            classes.shape != (2,)
            or classes.dtype.kind not in "iu"
            or set(classes.tolist()) != {0, 1}
        ):
            raise ValueError("El modelo debe clasificar las etiquetas enteras 0 y 1.")
        self.classes = classes.tolist()
        self.pipeline = pipeline
        self._lock = Lock()
        # Detecta modelos cargables pero inutilizables antes de anunciar readiness.
        self.predict(
            pd.DataFrame([[0.0] * len(DEMO_FEATURE_NAMES)], columns=DEMO_FEATURE_NAMES)
        )

    def predict(self, features: pd.DataFrame) -> list[Prediction]:
        # Serializa acceso por proceso y limita pools nativos para lotes pequeños.
        with self._lock, threadpool_limits(limits=1):
            labels = np.asarray(self.pipeline.predict(features))
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
