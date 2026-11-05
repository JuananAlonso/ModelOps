"""Factoría ASGI: una carga por lifespan y ningún acceso al Registry por request."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from modelops.serving.config import ServingConfig, resolve_serving_config
from modelops.serving.contracts import (
    ModelIdentity,
    PredictRequest,
    PredictResponse,
    RegressionPredictResponse,
    make_request_model,
)
from modelops.serving.predictor import Predictor
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
        if app.state.predictor is not None:
            install_predict_route(
                make_request_model(
                    app.state.predictor.contract.feature_names,
                    app.state.predictor.contract,
                )
            )
        try:
            yield
        finally:
            app.state.predictor = None

    app = FastAPI(
        title="ModelOps — inferencia tabular",
        version="1",
        description=(
            "Clasificación y regresión tabular. Versión de modelo fijada al inicio."
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

    @app.get("/model/schema")
    async def model_schema() -> dict:
        return require_predictor().contract.model_dump(mode="json")

    def install_predict_route(request_model: type) -> None:
        # La versión queda fijada en el lifespan: también fija su validación y OpenAPI.
        def predict(body: request_model) -> PredictResponse | RegressionPredictResponse:
            predictor = require_predictor()
            features = pd.DataFrame(
                [row.model_dump(by_alias=True) for row in body.instances],
                columns=predictor.contract.feature_names,
            )
            try:
                response = predictor.response(features, identity)
            except Exception as error:
                logger.error("Fallo de inferencia (%s).", type(error).__name__)
                app.state.predictor = None
                raise HTTPException(
                    status_code=500, detail="No se pudo completar la inferencia."
                ) from None
            return response

        app.router.routes[:] = [
            route
            for route in app.router.routes
            if getattr(route, "path", None) != "/predict"
        ]
        app.add_api_route(
            "/predict",
            predict,
            methods=["POST"],
            response_model=PredictResponse | RegressionPredictResponse,
            dependencies=[Depends(require_predictor)],
            responses={
                503: {"description": "Modelo no disponible"},
                500: {"description": "Fallo de inferencia"},
            },
        )
        app.openapi_schema = None

    install_predict_route(PredictRequest)
    return app
