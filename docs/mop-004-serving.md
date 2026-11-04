# MOP-004 — Inferencia local con FastAPI

## Auditoría breve y prioridad

Revisión del 2026-11-04 sobre `9dce979` y la rama
`feat/mop-004-fastapi-serving`:

| Área | Evidencia inicial | Siguiente acción |
| --- | --- | --- |
| Instalación y CI | Paquete `src/`, lockfile coherente, lint, tests y build definidos en CI | Incorporar serving al paquete y conservar comprobaciones |
| Entrenamiento MOP-002 | Pipelines completos, partición reproducible y manifiestos | Reutilizar el objeto ajustado con su preprocesamiento |
| Tracking MOP-003 | SQLite, artefactos locales, versiones recuperables; 100 tests iniciales pasan | Consumir una versión explícita desde otro proceso |
| Serving MOP-004, issue #7 | Sin módulo HTTP; dependencias FastAPI/Pydantic/Uvicorn/httpx ya añadidas en cambios locales | Implementar contrato, salud y paridad |
| Compose y E2E, issues #8 y #9 | Dependen de la API | Continuar después de revisar MOP-004 |

Prioridad: completar el circuito local `train → track → registry → serve`
antes de introducir servicios persistentes o infraestructura AWS. Esta entrega
implementa esa frontera HTTP. La versión la elige explícitamente el operador;
no se selecciona un ganador usando el test.

## Arranque reproducible

Desde la raíz del repositorio:

```bash
uv sync --locked
uv run --locked --offline modelops-train-mlflow --seed 42
uv run --locked --offline modelops-mlflow-runs
```

Consulta `registered_name` y `registered_version` en la salida. Usa una versión
revisada de una ejecución completa. En un Registry recién creado, la primera
versión será `1`:

```bash
uv run --locked --offline modelops-serve \
  --model-name modelops-baseline --model-version 1
```

Escucha en `127.0.0.1:8000`; `/openapi.json` describe el contrato y `/docs`
ofrece documentación interactiva. Ctrl+C detiene el proceso. Para LightGBM,
usa `--model-name modelops-lightgbm` y su versión correspondiente.

También está disponible la factoría ASGI:

```bash
MODELOPS_MODEL_VERSION=1 uv run --locked --offline uvicorn \
  modelops.serving.app:create_app --factory --host 127.0.0.1 --port 8000
```

Los tests usan `TestClient` como context manager para ejecutar el lifespan,
sin abrir puertos ni necesitar un servidor MLflow.

## Configuración

| Opción de CLI | Variable de entorno | Valor por defecto |
| --- | --- | --- |
| `--model-name` | `MODELOPS_MODEL_NAME` | `modelops-baseline` |
| `--model-version` | `MODELOPS_MODEL_VERSION` | Obligatoria |
| `--storage-dir` | `MODELOPS_MLFLOW_DIR` | `artifacts/mlflow` |
| `--tracking-uri` | `MLFLOW_TRACKING_URI` | SQLite dentro de storage-dir |
| `--host` | — | `127.0.0.1` |
| `--port` | — | `8000` |

Los argumentos explícitos prevalecen sobre las variables. La versión es un
string decimal positivo; se rechazan `latest`, alias, `0`, ceros iniciales y
valores vacíos. El backend sigue siendo SQLite local. Desde otro directorio de
trabajo, configura rutas absolutas. La ubicación efectiva de los artefactos
se obtiene del Registry.

Configuración inválida: la CLI termina con código `2`. Si la configuración es
válida pero el modelo no puede cargarse, el proceso permanece vivo y no
preparado (503), con un diagnóstico en el log. No crea un Registry si la base
de datos no existe. El cliente MLflow inspecciona el esquema SQLite; posibles
migraciones al actualizar MLflow siguen el procedimiento de MOP-003. No es
un modo de conexión SQL estrictamente de solo lectura.

## Contrato v1

`POST /predict` recibe `{"instances": [fila, ...]}`. Cada fila contiene
**exactamente las 30 claves originales** del dataset, incluidos sus espacios.
El esquema compartido está en `modelops.tabular`; OpenAPI enumera todas las
claves obligatorias. El orden de las claves JSON no importa; el orden de las
filas se conserva en la respuesta.

Se aceptan números JSON enteros y decimales finitos. Se rechazan strings
numéricos, booleanos, `null`, NaN, infinitos, columnas ausentes o extra, campos
extra en el body y batches fuera de **1 a 256 filas**. El límite de filas se
valida tras parsear el JSON; no limita los bytes HTTP. No hay imputación ni
restricciones clínicas: el dataset es exclusivamente un ejemplo técnico.

Para preparar una petición completa:

```bash
uv run --locked --offline python - <<'PY'
import json
from pathlib import Path
from modelops.training.data import load_demo_dataset

rows = load_demo_dataset().features.iloc[:3].to_dict("records")
Path("/tmp/modelops-predict.json").write_text(json.dumps({"instances": rows}))
PY
curl --fail-with-body http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @/tmp/modelops-predict.json
```

Respuesta de ejemplo, con probabilidades ilustrativas:

```json
{
  "schema_version": "1",
  "model": {
    "name": "modelops-baseline",
    "version": "1",
    "uri": "models:/modelops-baseline/1"
  },
  "classes": [0, 1],
  "class_names": ["malignant", "benign"],
  "predictions": [{"label": 0, "probabilities": [0.98, 0.02]}]
}
```

`probabilities[i]` corresponde a `classes[i]` y `class_names[i]`, en el orden
real de `pipeline.classes_`. No se presupone la posición de la clase positiva.
Se mantiene `0=malignant`, `1=benign`, **sin uso ni interpretación clínica**.

## Salud, errores y recuperación

```bash
curl --fail-with-body http://127.0.0.1:8000/health/live
curl --fail-with-body http://127.0.0.1:8000/health/ready
```

| Ruta o situación | Código | Significado |
| --- | --- | --- |
| `/health/live` | 200 | Proceso atiende peticiones |
| `/health/ready` | 200 | Modelo cargado y comprobado; incluye identidad |
| `/health/ready`, sin modelo utilizable | 503 | Falló la carga o una inferencia posterior |
| `/predict`, entrada válida | 200 | Etiquetas y probabilidades verificadas |
| `/predict`, entrada inválida | 422 | Ubicación, tipo y mensaje, sin repetir valores |
| `/predict`, sin modelo | 503 | No cambia a otra versión |
| `/predict`, excepción o salida inválida | 500 | Error genérico; invalida readiness hasta reiniciar |

La carga ocurre una vez durante el lifespan, antes de aceptar tráfico, usando
un thread pool. Se comprueba que el objeto es un Pipeline ajustado, que tiene
las 30 features y las clases enteras 0/1 y que produce una salida válida sobre
una fila sintética de ceros. Esto comprueba funcionamiento, no calidad.

Tras un fallo, revisa nombre, versión, permisos, artefactos y compatibilidad,
corrige y reinicia. No se reintenta por petición ni se recarga al crear nuevas
versiones. Borrar del Registry una versión ya cargada no modifica el objeto
en memoria. Las respuestas y mensajes propios de carga no incluyen trazas,
rutas privadas ni texto original de excepciones; MLflow puede emitir sus
propios mensajes de inicialización.

## Decisiones y límites

- **Una carga por proceso:** fija la versión y evita consultas/deserialización
  por petición. Permite seguir sirviendo si el Registry deja de estar accesible
  después de arrancar. Cambiar versión exige reinicio.
- **Pipeline nativo completo:** `predict` y `predict_proba` reutilizan selección
  y preprocesamiento ajustado. Pydantic valida datos; la firma MLflow no garantiza
  por sí sola la interfaz nativa de probabilidades ni una salida finita.
- **Esquema compartido:** `tabular.py` no importa entrenamiento ni carga datos.
  Serving depende de ese contrato y del adaptador de recuperación.
- **Concurrencia:** handlers síncronos en thread pool, lock de inferencia por
  proceso y pools nativos limitados a un hilo. No se afirma un SLO productivo.
- **Selección explícita:** Registry READY indica disponibilidad, no aprobación.
  Las versiones de padres FAILED requieren inspección y nunca se seleccionan
  automáticamente. No hay promoción, reentrenamiento ni selección por métricas.
- **Procedencia fiable:** cloudpickle solo se usa con modelos controlados por
  el operador. No hay autenticación ni TLS; el host local es el valor por defecto.
  `--host` permite cambiarlo expresamente para desarrollo.

## Verificación y siguiente entrega

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked --offline pytest
uv build
```

Los tests comprueban batches de 1/N/256 filas, validación, OpenAPI, orden de
columnas y clases, salud, apagado, carga única, ausencia de `fit` y errores
sanitizados. La integración real usa SQLite y artefactos temporales, ambos
candidatos y dos arranques por candidato. Exige igualdad exacta de etiquetas
y probabilidades con `rtol=1e-9`, `atol=1e-12` entre pipeline original,
recuperado y JSON HTTP. Comprueba también versiones inexistentes y ausencia
de nuevos runs/versiones durante serving. Los tests bloquean conexiones de
red externa y no requieren Docker ni AWS.

Resultado local del 2026-11-04: **158 tests pasan** en instalación no editable
(37,38 s); Ruff lint, formato y construcción de wheel/sdist correctos. Una
prueba adicional con los comandos instalados entrenó en almacenamiento
temporal, arrancó Uvicorn en localhost, obtuvo readiness 200 y verificó paridad
para tres filas por HTTP real; el servidor se detuvo y los datos temporales
se limpiaron al terminar. Quedan avisos de deprecación de Starlette/httpx y
SQLAlchemy/MLflow, sin fallos. El aislamiento de esta sesión bloquea sockets
internos de asyncio; la suite completa se ejecutó fuera de ese aislamiento,
manteniendo la fixture que prohíbe conexiones externas durante los tests.

Tras revisar MOP-004, sigue **MOP-005 (#8)**: Compose con API, MLflow y
PostgreSQL y artefactos accesibles entre contenedores. Las rutas locales de
MOP-003 no son portables automáticamente. MOP-006 (#9) completará E2E de
infraestructura y mediciones operativas.

Referencias: [lifespan de FastAPI](https://fastapi.tiangolo.com/advanced/events/)
y [tipos numéricos de Pydantic](https://docs.pydantic.dev/latest/api/standard_library_types/).
