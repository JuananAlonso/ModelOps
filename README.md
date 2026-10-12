# ModelOps

Base de una plataforma MLOps en Python 3.12, organizada como un monolito modular.
El diseño prevé procesos separados para entrenamiento e inferencia. MOP-001
estableció paquete, entorno y CI; MOP-002 añade entrenamiento y evaluación local
de clasificación binaria con un baseline y LightGBM. MOP-003 añade tracking
local y versiones iniciales de modelos con MLflow, desde una capa independiente.

## Requisitos

- [uv](https://docs.astral.sh/uv/getting-started/installation/) **0.13.0**, única
  herramienta para gestionar Python, dependencias, entornos y empaquetado.
- Acceso a Internet durante la primera instalación para descargar Python y
  las dependencias desde sus distribuciones públicas.
- Git para obtener el repositorio.

No es necesario instalar Python por separado: uv proporciona Python 3.12.
Solo se soporta esta versión menor durante el bootstrap.

Para instalar la versión fijada de uv en Linux o macOS:

```bash
curl -LsSf https://astral.sh/uv/0.13.0/install.sh | sh
```

En Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.13.0/install.ps1 | iex"
```

Abre una terminal nueva si el instalador lo indica y comprueba `uv --version`.
La versión requerida también se declara en `pyproject.toml` y la utiliza CI.

## Instalación reproducible

Desde la raíz del repositorio:

```bash
uv python install
uv sync --locked
uv run --locked python -c "import modelops; print(modelops.__version__)"
```

`.python-version` fija Python **3.12.15**, incluida la versión de parche.
`uv.lock` registra las versiones y hashes
de las dependencias directas y transitivas. `--locked` rechaza un lockfile
desactualizado en vez de modificarlo. `uv sync` crea `.venv/`, instala el paquete
en modo editable y añade el grupo `dev` (pytest y Ruff) por defecto.
No hace falta activar el entorno: `uv run` ejecuta cada comando dentro de él.
Las dependencias de entrenamiento son NumPy, pandas, scikit-learn, LightGBM,
joblib y threadpoolctl. Todas se utilizan en el pipeline y quedan fijadas por
el lockfile. MLflow **3.16.1** se instala para Tracking, Registry y UI. Su
distribución completa amplía las dependencias transitivas; algunas incluyen
componentes web y el SDK de Docker, pero esta entrega no implementa una API de
inferencia ni utiliza Docker o cloud. El lockfile pasa de 20 a 98 paquetes
resueltos contando las variantes de plataforma. La primera instalación puede
tardar más; las siguientes reutilizan la caché de uv.

## Entrenamiento local — MOP-002

```bash
uv run --locked modelops-train --seed 42 --output-dir artifacts/training
```

También puedes configurar el tamaño de test:

```bash
uv run --locked modelops-train --seed 17 --test-size 0.25 --output-dir artifacts/training
```

Después de `uv sync --locked`, el entrenamiento puede ejecutarse sin red:

```bash
uv run --locked --offline modelops-train --seed 42 --output-dir artifacts/training
```

El dataset `sklearn.datasets.load_breast_cancer` está incluido en scikit-learn.
Se usa **exclusivamente como ejemplo técnico, sin uso ni afirmaciones clínicas**.
Se conserva su codificación: `0 = malignant`, `1 = benign`; las métricas de
precision, recall, F1 y ROC-AUC consideran **1 la clase positiva**.

El comando valida el contrato, crea una partición estratificada compartida,
entrena LogisticRegression y LightGBM y presenta sus métricas en JSON. Solo usa
train para ajustar los pipelines. Test se utiliza una vez por candidato para
evaluación final; no se ajustan hiperparámetros ni se elige un ganador con él.

Valores por defecto: semilla `42`, fracción de test `0.2` y directorio
`artifacts/training`. Las semillas válidas son enteros entre `0` y `2**32 - 1`.
La fracción debe dejar ambas clases en ambas particiones. Un error de datos,
configuración o escritura produce un mensaje en stderr y código de salida `2`.

Cada ejecución crea un directorio nuevo con fecha UTC y sufijo aleatorio:

```text
artifacts/training/20261012T180000000000Z-a1b2c3d4/
├── config.json         # Configuración efectiva y parámetros de ambos modelos
├── metrics.json        # Accuracy, precision, recall, F1 y ROC-AUC por modelo
├── split.json          # Posiciones originales de train y test, en su orden
├── manifest.json       # Identidad, huellas, esquema, versiones, Git y checksums
└── models/
    ├── baseline.joblib # Selector de columnas, escalado y LogisticRegression
    └── lightgbm.joblib # Selector de columnas y LightGBM
```

El ID de ejecución cambia aunque el experimento sea idéntico. Los directorios
anteriores se conservan y una colisión falla en lugar de sobrescribir. Solo un
`manifest.json` válido con `status = "complete"` identifica una escritura
completada; un fallo puede dejar un directorio parcial que debe inspeccionarse.
Los artefactos quedan ignorados por Git.

Para usar un pipeline guardado, conserva todo el entorno versionado y carga
**únicamente archivos joblib de procedencia fiable**:

```python
import joblib
from modelops.training.data import load_demo_dataset

pipeline = joblib.load("artifacts/training/<run-id>/models/baseline.joblib")
features = load_demo_dataset().features.iloc[:3]
predictions = pipeline.predict(features)
positive_probabilities = pipeline.predict_proba(features)[
    :, list(pipeline.classes_).index(1)
]
```

El objeto incluye selección de columnas y transformaciones. Las columnas deben
tener los nombres documentados en el manifiesto: el selector restablece su orden,
falla si falta alguna y descarta columnas adicionales durante predicción. La
validación de entrenamiento exige el esquema exacto; la validación explícita de
entradas de una futura API pertenece a otra entrega.

Para datos, entorno, configuración y semilla iguales, se espera la misma
partición y métricas/probabilidades iguales con `rtol=1e-9`, `atol=1e-12` en el
mismo entorno. No se garantiza igualdad de bytes de los archivos ni resultados
idénticos entre versiones, compilaciones o hardware distintos. Se registran
versiones, plataforma, huella de datos y revisión Git cuando se conoce; una
revisión con `dirty=true` no reproduce por sí sola cambios sin commit.

Consulta la [guía técnica de MOP-002](docs/mop-002-training.md) para los contratos,
las fórmulas, las alternativas y cómo extender el diseño. No hay imputación,
variables categóricas, búsqueda de hiperparámetros, validación cruzada, selección
automática ni serving. El comando local anterior no activa MLflow.

## Tracking y Registry local — MOP-003

Desde la raíz del repositorio, después de `uv sync --locked`:

```bash
uv run --locked --offline modelops-train-mlflow --seed 42
uv run --locked --offline modelops-train-mlflow --seed 123
uv run --locked --offline modelops-mlflow-runs
uv run --locked --offline modelops-mlflow-ui --port 5000
```

Abre `http://127.0.0.1:5000`. El cliente escribe directamente en SQLite y
archivos locales; la UI no necesita estar activa para entrenar ni consultar.
La UI utiliza el mismo backend y los mismos archivos, con acceso directo al
artifact store. El wrapper fija `127.0.0.1` y no ofrece un argumento para
exponerlo públicamente. Detén la UI con Ctrl+C.

Cada ejecución conserva los artefactos originales de MOP-002 y crea un run padre
con dos hijos, `baseline` y `lightgbm`. En la UI, expande los padres y compara
los hijos filtrando por `candidate` y `seed`. El comando de consulta presenta
hasta 1000 runs recientes y devuelve candidatos de ejecuciones completas; no
selecciona un ganador. Una semilla diferente cambia el test, por lo que esta
comparación no equivale a validar un modelo sobre un único holdout fijo.

Parámetros y configuración opcionales:

```bash
uv run --locked modelops-train-mlflow --seed 42 --test-size 0.25 \
  --output-dir artifacts/training --storage-dir artifacts/mlflow \
  --experiment-name modelops-tabular --model-prefix modelops
```

| Opción | Variable de entorno | Valor por defecto |
| --- | --- | --- |
| `--storage-dir` | `MODELOPS_MLFLOW_DIR` | `artifacts/mlflow` |
| `--tracking-uri` | `MLFLOW_TRACKING_URI` | SQLite `mlflow.db` dentro de storage-dir |
| `--artifact-dir` | `MODELOPS_MLFLOW_ARTIFACT_DIR` | `artifacts/` dentro de storage-dir |
| `--experiment-name` | `MLFLOW_EXPERIMENT_NAME` | `modelops-tabular` |
| `--model-prefix` | — | `modelops` |

Un argumento explícito prevalece sobre su variable de entorno. `--storage-dir`
define las rutas por defecto, sin sustituir un URI o artifact-dir configurado
explícitamente. Las tres CLI comparten estas opciones. Las rutas se convierten
en absolutas; para ejecutar desde distintos directorios, utiliza rutas absolutas
o las mismas variables de entorno. MOP-003 admite solo SQLite persistente y
artefactos locales. Un experimento existente conserva su artifact location:
si cambias la carpeta, usa otro nombre de experimento o restaura la original.

La salida del entrenamiento contiene el ID del padre, los IDs de los hijos,
nombres y versiones del Registry y URIs para recuperación. Cada ejecución crea
una versión nueva de `modelops-baseline` y `modelops-lightgbm`.
**Registrar una versión no significa promocionarla a producción**: no se crean
alias de producción ni reglas de promoción o rollback.

```text
artifacts/
├── training/<execution-id>/             # Artefactos originales de MOP-002
└── mlflow/
    ├── mlflow.db                        # Metadatos de Tracking y Registry
    └── artifacts/
        ├── <parent-run-id>/artifacts/
        │   ├── training/                # Copia verificable del resultado local
        │   └── tracking.json            # Referencias a hijos y versiones
        └── <child-run-id>/artifacts/
            ├── metrics.json             # Conserva null para métricas indefinidas
            ├── model-config.json
            ├── registry.json
            └── pipeline/
                ├── MLmodel              # Flavors, firma y metadatos
                ├── model.pkl            # Pipeline completo serializado
                ├── input_example.json
                └── requirements.txt     # También se generan archivos de entorno
```

Las métricas indefinidas se omiten de los valores numéricos de Tracking; se
conservan como `null` en JSON y en la etiqueta `modelops.undefined_metrics`.
Los valores cero que sí estén definidos se registran como cero.

Para recuperar una versión explícita con el mismo backend:

```python
from modelops.tracking.adapter import load_registered_pipeline
from modelops.tracking.config import resolve_config
from modelops.training.data import load_demo_dataset

pipeline = load_registered_pipeline(resolve_config(), "modelops-baseline", "1")
predictions = pipeline.predict(load_demo_dataset().features.iloc[:3])
```

El helper resuelve la versión con un cliente explícito y no cambia el URI global.
Solo carga modelos fiables: esta integración usa cloudpickle para preservar
los pipelines completos de sklearn y LightGBM. La firma describe nombres y tipos
de entrada y salida; el flavor `pyfunc` la valida, mientras que la carga nativa
de sklearn conserva su API `predict/predict_proba`.

La telemetría externa de MLflow se desactiva por defecto al importar la
integración; la UI y CI la desactivan explícitamente. El proceso de UI también
desactiva la ejecución de jobs de MLflow, que no se utiliza en esta entrega.
SQLite puede migrar su esquema al inicializarse. Antes de actualizar MLflow,
detén escritores y UI y conserva juntos la base de datos y el artifact store.
La primera inicialización debe hacerse con un único proceso: espera a que
termine antes de iniciar otra CLI o la UI. MLflow puede rechazar un cliente
que observe el esquema mientras otro proceso lo está migrando.
Los tests utilizan bases temporales independientes, sin servidor ni Internet.

Consulta la [guía técnica de MOP-003](docs/mop-003-mlflow.md) para contratos,
alternativas, seguridad, recuperación y las preguntas de revisión técnica.

## Comprobaciones

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

Los tests comprueban instalación, contrato y rechazo de datos inválidos,
estratificación, reproducibilidad, prevención de leakage, ambos modelos,
métricas con ejemplos calculados a mano, persistencia y CLI integrada. Usan
datos empaquetados, directorios temporales y bloquean conexiones de red en los
tests del proceso principal. CI ejecuta pytest con `UV_OFFLINE=1` una vez
instaladas las dependencias. Pytest utiliza `importlib`; no se añade `src/` a
`PYTHONPATH` para evitar que una instalación incorrecta quede oculta.

Para reproducir las comprobaciones de CI sobre una instalación no editable:

```bash
uv sync --locked --no-editable
uv run --locked --no-editable ruff check .
uv run --locked --no-editable ruff format --check .
uv run --locked --no-editable pytest
uv build
```

`uv build` genera el wheel y la distribución de código fuente en `dist/`.
Para volver al desarrollo editable, ejecuta `uv sync --locked`.
El workflow `.github/workflows/ci.yml` ejecuta estas comprobaciones en cada
`push` y `pull_request`, usando Python gestionado por uv y el mismo lockfile.

## Actualización del entorno

Los cambios de dependencias se hacen con uv, por ejemplo:

```bash
uv add --dev "pytest>=8.4,<9"
uv lock --upgrade-package pytest
uv sync --locked
```

Revisa y versiona juntos `pyproject.toml` y `uv.lock`, y repite las comprobaciones.
La versión del intérprete se actualiza explícitamente mediante `uv python pin`
y debe permanecer en la serie 3.12. El backend de construcción `uv_build` tiene
una versión exacta en `pyproject.toml`; cambia esa versión y la de uv de forma
coordinada, verificando instalación y construcción.

## Estructura del proyecto

```text
.
├── .github/workflows/ci.yml     # Lint, formato, tests y empaquetado
├── .gitignore                  # Entornos, secretos y artefactos locales
├── .python-version             # Python gestionado por uv
├── docs/
│   ├── adr/001-architecture.md # Decisiones y propuestas arquitectónicas
│   ├── mop-002-training.md     # Guía técnica y decisiones de entrenamiento
│   ├── pr-mop-002.md           # Descripción de la entrega anterior
│   ├── mop-003-mlflow.md       # Tracking, Registry y decisiones técnicas
│   └── pr-mop-003.md           # Propuesta para revisión del PR de MOP-003
├── pyproject.toml              # Paquete, grupo dev y configuración
├── README.md
├── src/modelops/
│   ├── __init__.py              # Paquete y versión instalada
│   ├── tracking/
│   │   ├── config.py          # SQLite, artifact location y UI localhost
│   │   ├── adapter.py         # Tracking, Registry, consulta y recuperación
│   │   └── cli.py             # Entrenamiento con tracking, consulta y UI
│   └── training/
│       ├── data.py             # Contrato y carga
│       ├── validation.py       # Validación y huella del dataset
│       ├── config.py           # Configuración y restricciones
│       ├── splitting.py        # Partición estratificada
│       ├── models.py           # Factoría de pipelines sin ajustar
│       ├── evaluation.py       # Métricas comunes
│       ├── runner.py           # Orquestación de ML en memoria
│       ├── artifacts.py        # Persistencia local y procedencia
│       └── cli.py              # Entrada ejecutable
├── tests/                      # Pruebas de paquete y entrenamiento
└── uv.lock                     # Dependencias resueltas y hashes
```

Las decisiones del stack y sus riesgos están en
[ADR-001](docs/adr/001-architecture.md). Serving, servicios locales,
infraestructura AWS, observabilidad y reentrenamiento quedan para entregas
posteriores. MLflow se integra desde una capa externa al entrenamiento;
PostgreSQL y S3 son destinos futuros y no se utilizan en el desarrollo local.

`.gitignore` excluye secretos y artefactos habituales, pero los archivos que ya
estén versionados no quedan protegidos por estas reglas. Las plantillas
`.env.example` pueden versionarse y deben contener únicamente valores ficticios.
