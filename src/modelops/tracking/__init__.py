"""Integración externa de Tracking y Registry; el entrenamiento no la importa."""

import os

# Entrega local: desactivar telemetría externa antes de importar MLflow.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
