# MOP-002: pipeline reproducible de entrenamiento y evaluación

Refs #2. Base: `main`. Rama: `feat/mop-002-training-pipeline`.

El paquete incorpora una CLI local que valida el dataset tabular empaquetado,
divide con estratificación y entrena LogisticRegression y LightGBM sobre el
mismo train. El preprocesamiento vive en los pipelines y test se reserva para
evaluación de candidatos prefijados, sin seleccionar un ganador.

Cada ejecución conserva ambos pipelines, métricas, configuración efectiva,
posiciones del split y un manifiesto con versiones, huella de datos, procedencia
Git y checksums. Los directorios se crean de forma exclusiva para preservar las
ejecuciones anteriores. La lógica de ML es independiente de persistencia y tracking.

Incluye dependencias gestionadas por uv, CLI documentada, pruebas sin servicios,
casos calculados a mano para métricas, tests causales de leakage y cobertura de
persistencia/colisiones. La CI existente ejecuta los tests sobre el paquete
no editable con uv en modo offline después de instalar dependencias.

## Validación

Validación local en Linux CPU, Python 3.12.15 y uv 0.13.0:

- `uv run --locked ruff check .`: sin errores.
- `uv run --locked ruff format --check .`: 23 archivos con formato correcto.
- `uv run --locked pytest`: 74 tests pasan en 1,53 s, con uv en modo offline.
- `uv build`: wheel y distribución fuente generados correctamente.
- `uv lock --check --offline`: lockfile coherente con los metadatos.
- `uv sync --locked --no-editable` y `uv run --locked --no-editable pytest`:
  74 tests pasan en 1,71 s sobre la instalación usada por CI.
- CLI ejecutada sin red en modo editable y no editable con semilla 42:
  las métricas, configuración, partición (455 train / 114 test) y huella
  coinciden. Se conservan dos directorios de ejecución diferentes.

La ejecución remota de GitHub Actions queda pendiente de publicar este cambio.
Se usó el uv 0.13.0 disponible en `/tmp/modelops-tools`; el uv 0.12.10 del
PATH del sistema no satisface el pin heredado de MOP-001. El README especifica
cómo instalar la versión requerida.

## Alcance y revisión

Ejemplo exclusivamente técnico sin uso clínico. No introduce MLflow, API,
Docker, AWS, selección de modelos ni reentrenamiento. No modifica ADR-001.

Esta propuesta requiere revisión técnica y después evaluación de comprensión
del autor. La issue #2 debe permanecer abierta hasta autorización explícita;
este PR no debe fusionarse ni cerrarla automáticamente durante esa revisión.
