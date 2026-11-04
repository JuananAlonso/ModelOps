"""Factoría ASGI: una carga por lifespan y ningún acceso al Registry por request."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from modelops.serving.config import ServingConfig, resolve_serving_config
from modelops.serving.contracts import ModelIdentity, PredictRequest, PredictResponse
from modelops.serving.predictor import Predictor
from modelops.tabular import DEMO_FEATURE_NAMES, DEMO_TARGET_NAMES
from modelops.tracking.adapter import load_registered_pipeline

logger = logging.getLogger(__name__)


def create_app(config: ServingConfig | None = None) -> FastAPI:
    selected = config if config is not None else resolve_serving_config()
    identity = ModelIdentity(
        name=selected.model_name,
        version=selected.model_version,
        uri=f"models:/{selected.model_name}/{selected.model_version}",
    )

    def load_predictor() -> Predictor:
        # No inicializar un Registry vacío desde el proceso de inferencia.
        if (
            not selected.tracking.remote_tracking_uri
            and not selected.tracking.database_path.is_file()
        ):
            raise FileNotFoundError("Inicializa el Registry con entrenamiento primero.")
        return Predictor(
            load_registered_pipeline(
                selected.tracking, selected.model_name, selected.model_version
            )
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = None
        try:
            app.state.predictor = await run_in_threadpool(load_predictor)
        except Exception as error:
            # No incluir URIs de backend, rutas, credenciales ni trazas del loader.
            logger.error(
                "No se pudo cargar %s versión %s (%s). Comprueba Registry, "
                "artefactos y compatibilidad; corrige la configuración y reinicia.",
                selected.model_name,
                selected.model_version,
                type(error).__name__,
            )
        try:
            yield
        finally:
            app.state.predictor = None

    app = FastAPI(
        title="ModelOps — inferencia de demostración",
        version="1",
        description=(
            "Ejemplo técnico sin uso clínico. Versión de modelo fijada al inicio."
        ),
        lifespan=lifespan,
    )
    app.state.predictor = None

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        # Los errores originales pueden contener NaN/Inf, el body o contextos Python.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
                    for item in error.errors()
                ]
            },
        )

    def require_predictor() -> Predictor:
        predictor = app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Modelo no disponible.")
        return predictor

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", responses={503: {"description": "Modelo no disponible"}})
    async def ready() -> dict:
        require_predictor()
        return {"status": "ready", "model": identity.model_dump()}

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={
            503: {"description": "Modelo no disponible"},
            500: {"description": "Fallo de inferencia"},
        },
    )
    def predict(body: PredictRequest) -> PredictResponse:
        predictor = require_predictor()
        features = pd.DataFrame(
            [row.model_dump(by_alias=True) for row in body.instances],
            columns=DEMO_FEATURE_NAMES,
        )
        try:
            predictions = predictor.predict(features)
        except Exception as error:
            logger.error("Fallo de inferencia (%s).", type(error).__name__)
            app.state.predictor = None
            raise HTTPException(
                status_code=500, detail="No se pudo completar la inferencia."
            ) from None
        return PredictResponse(
            model=identity,
            classes=predictor.classes,
            class_names=[DEMO_TARGET_NAMES[label] for label in predictor.classes],
            predictions=predictions,
        )

    return app
