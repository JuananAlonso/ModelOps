# ModelOps

Base de una plataforma MLOps en Python 3.12, organizada como un monolito modular.
El diseño prevé procesos separados para entrenamiento e inferencia. MOP-001
estableció paquete, entorno y CI; MOP-002 añade entrenamiento y evaluación local
de clasificación binaria con un baseline y LightGBM. MOP-003 añade tracking
local y versiones iniciales de modelos con MLflow, desde una capa independiente.
MOP-004 sirve una versión explícita mediante FastAPI, con validación de entradas
y comprobaciones de salud, en un proceso separado del entrenamiento.
MOP-005 permite ejecutar API, MLflow y PostgreSQL con Docker Compose y
transferir artefactos por HTTP entre procesos. La CLI CSV permite entrenar
clasificación binaria, multiclase y regresión con tablas propias, variables
numéricas y categóricas, y valores ausentes.

Para usarlo sin instalar Python ni uv en el host, sigue el recorrido
[Docker Compose desde cero](#docker-compose-csv-propio-desde-cero).

## Quickstart con un CSV propio

Desde la raíz del proyecto, con un CSV UTF-8 separado por comas:

```bash
uv sync --locked
uv run --locked modelops inspect clientes.csv
uv run --locked modelops train clientes.csv --target churn \
  --algorithm lightgbm --name churn --exclude id_cliente
uv run --locked modelops predict nuevos_clientes.csv --model churn --version 1
```

`1` representa la versión devuelta por el entrenamiento. Cada entrenamiento
imprime el comando para predecir con la **versión realmente registrada** y el
backend efectivo. Sustituye `nuevos_clientes.csv` por tu archivo de variables
de entrada. Registrar un modelo no cambia la versión que sirve la API.

`inspect` presenta una tabla alineada con el tipo inferido de cada columna,
valores ausentes, cardinalidad y posibles objetivos. Estos candidatos son
orientativos; `inspect` no selecciona el objetivo ni entrena. `train` exige
objetivo, algoritmo y nombre. Para clasificación
binaria o multiclase admite `logistic-regression` y `lightgbm`; para regresión,
usa `--task regression` con `linear-regression` o `lightgbm`. La semilla
predeterminada es **42** y test usa **0.2** (estratificado en clasificación).
Repite `--exclude` para ignorar columnas y `--categorical` para forzar categorías
como códigos postales; `--numeric` exige números válidos. Se imputan ausentes
y se codifican categorías con estadísticas aprendidas solo de train. `predict`
exige exactamente las columnas del contrato guardado; acepta otro orden y conserva
el orden de las filas. Elimina del CSV de predicción el objetivo y las columnas
excluidas, incluidos identificadores.

Si omites `--experiment-name`, `train` usa un experimento derivado del
algoritmo y modelo, por ejemplo `modelops-lightgbm-iris_v1`. También respeta
`MLFLOW_EXPERIMENT_NAME` cuando configuras esa variable.

Para CSV de otros formatos, usa `--delimiter ';' --decimal ','` o
`--delimiter tab`, y `--encoding latin-1` si corresponde. Estas opciones
funcionan en inspect, train y predict. Por defecto se usa UTF-8 y coma.

Para automatizar, añade `--json`: stdout contiene solo el resultado JSON;
progreso y errores van a stderr. Los errores esperables devuelven código `2`.
Cada subcomando incluye un ejemplo en `--help`.

### Demo incluida: dos esquemas distintos, sin editar Python

Los archivos de `examples/csv/` son datos sintéticos. Desde un clon limpio:

```bash
uv sync --locked
uv run --locked modelops inspect examples/csv/clientes.csv
uv run --locked modelops train examples/csv/clientes.csv --target churn \
  --algorithm lightgbm --name churn --exclude id_cliente \
  --storage-dir artifacts/csv-demo --experiment-name csv-demo
uv run --locked modelops predict examples/csv/nuevos_clientes.csv \
  --model churn --version 1 --storage-dir artifacts/csv-demo

uv run --locked modelops inspect examples/csv/pedidos.csv
uv run --locked modelops train examples/csv/pedidos.csv --target devuelto \
  --algorithm logistic-regression --name devoluciones \
  --storage-dir artifacts/csv-demo --experiment-name csv-demo
uv run --locked modelops predict examples/csv/nuevos_pedidos.csv \
  --model devoluciones --version 1 --storage-dir artifacts/csv-demo
```

En la primera ejecución ambos modelos tendrán versión 1. Si repites el
entrenamiento, utiliza la versión y el comando que imprima. Los CSV nuevos
incluidos ya tienen las variables en otro orden para mostrar su reordenación.
`--storage-dir` aísla la demo local; si tienes `MLFLOW_TRACKING_URI` configurado,
este prevalece. `--tracking-uri` permite elegir explícitamente otro backend.

Para servir la versión de clientes en otro proceso:

```bash
uv run --locked modelops-serve --model-name churn --model-version 1 \
  --storage-dir artifacts/csv-demo --port 8002
```

En `http://127.0.0.1:8002/docs` aparece el esquema de esa versión; también se
puede consultar con `GET /model/schema`. Para probarla desde otra terminal:

```bash
curl --fail-with-body http://127.0.0.1:8002/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances":[{"antiguedad":2,"edad":25},{"edad":67,"antiguedad":8}]}'
```

La CLI y HTTP leen el mismo contrato embebido en el pipeline. Los modelos
antiguos del dataset integrado siguen cargando con su esquema original.
Consulta el [contrato CSV y las decisiones técnicas](docs/csv-cli.md) para
la codificación de etiquetas, configuración, artefactos y límites.

### Nuevos ejemplos: categorías, ausentes y regresión

```bash
uv run --locked modelops train examples/csv/segmentos.csv --target segmento \
  --algorithm logistic-regression --name segmentos --exclude id \
  --categorical codigo_postal --storage-dir artifacts/csv-general
uv run --locked modelops predict examples/csv/nuevos_segmentos.csv \
  --model segmentos --version 1 --storage-dir artifacts/csv-general

uv run --locked modelops train examples/csv/viviendas.csv --target precio \
  --task regression --algorithm linear-regression --name viviendas \
  --exclude id --delimiter ';' --decimal ',' --storage-dir artifacts/csv-general
uv run --locked modelops predict examples/csv/nuevas_viviendas.csv \
  --model viviendas --version 1 --delimiter ';' --decimal ',' \
  --storage-dir artifacts/csv-general
```

La regresión devuelve valores predichos y métricas MAE/RMSE/R². La clasificación
multiclase conserva etiquetas originales y devuelve una probabilidad por clase.
La API adapta su validación al modelo cargado y admite `null` para imputar.
Consulta el [plan, formatos admitidos y límites](docs/csv-cli.md): se requiere
una tabla supervisada con objetivo explícito; no se interpreta texto libre ni
se infiere el problema de negocio a partir de un archivo.

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
componentes web y el SDK de Docker. FastAPI, Pydantic y Uvicorn implementan la
API local; httpx permite probarla con TestClient. psycopg2-binary permite que
MLflow utilice PostgreSQL en Compose. No se utiliza cloud.
El lockfile contiene 101 paquetes resueltos contando variantes de plataforma.
La primera instalación puede tardar más; las siguientes reutilizan la caché de uv.

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
entradas HTTP se implementa en MOP-004 con esquema exacto y números finitos.

Para datos, entorno, configuración y semilla iguales, se espera la misma
partición y métricas/probabilidades iguales con `rtol=1e-9`, `atol=1e-12` en el
mismo entorno. No se garantiza igualdad de bytes de los archivos ni resultados
idénticos entre versiones, compilaciones o hardware distintos. Se registran
versiones, plataforma, huella de datos y revisión Git cuando se conoce; una
revisión con `dirty=true` no reproduce por sí sola cambios sin commit.

Consulta la [guía técnica de MOP-002](docs/mop-002-training.md) para los contratos,
las fórmulas, las alternativas y cómo extender el diseño. No hay imputación,
variables categóricas, búsqueda de hiperparámetros, validación cruzada, selección
automática en entrenamiento. El comando local anterior no activa MLflow.

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

## Inferencia HTTP local — MOP-004

Después de registrar modelos con MOP-003, consulta una versión concreta y
arranca la API en otro proceso:

```bash
uv run --locked --offline modelops-mlflow-runs
# Sustituye 1 por la versión revisada que muestre el comando anterior.
uv run --locked --offline modelops-serve \
  --model-name modelops-baseline --model-version 1
```

Escucha en `http://127.0.0.1:8000`. `/docs` documenta el contrato completo;
`GET /health/live` devuelve 200 si el proceso responde y `GET /health/ready`
solo devuelve 200 con el modelo cargado y comprobado. Si la carga falla, la
API permanece viva con readiness 503; corrige el problema y reinicia.

`POST /predict` recibe `{"instances": [fila, ...]}` con entre 1 y 256 filas,
cada una con las 30 claves originales y valores numéricos finitos. Rechaza
columnas ausentes o extra, strings numéricos, booleanos, null, NaN e infinitos
con 422. La respuesta incluye `schema_version`, nombre/versión/URI del modelo,
`classes`, `class_names` y `predictions` con etiqueta y probabilidades en el
orden de `classes_`. Sigue siendo **un ejemplo técnico sin uso clínico**.

La versión numérica es obligatoria: también puede fijarse mediante
`MODELOPS_MODEL_VERSION`, con `MODELOPS_MODEL_NAME` opcional. Se carga una vez
al arrancar, sin seleccionar `latest`, entrenar ni consultar Registry por
petición. Comparte la configuración SQLite de MOP-003; los modelos deben
tener procedencia fiable. No hay promoción automática ni autenticación pública.

Consulta la [guía de MOP-004](docs/mop-004-serving.md) para el ejemplo `curl`
completo, configuración, errores, pruebas de paridad y auditoría del siguiente
paso. Docker Compose y PostgreSQL corresponden a MOP-005.

## Docker Compose: CSV propio desde cero

Necesitas Docker Engine en ejecución, Docker Compose y acceso al motor.
Estos pasos ejecutan entrenamiento e inferencia dentro de contenedores; no
requieren instalar Python ni uv en el host. Ejecútalos desde la raíz del clon.

### 1. Construir y arrancar MLflow

```bash
cp -n .env.example .env
docker compose build api
docker compose up -d --wait mlflow
```

La imagen construida por `api` es compartida por MLflow y `train`. Compose
arranca PostgreSQL como dependencia. Espera a entrenar antes de arrancar API:
su readiness exige que el modelo ya exista. Los puertos del host se configuran
en `.env` (API 8001 y MLflow 5001 por defecto).

### 2. Inspeccionar y entrenar

Este ejemplo funciona con los CSV sintéticos incluidos en cualquier clon:

```bash
docker compose run --rm -v "$PWD/examples/csv:/csv:ro" \
  train modelops inspect /csv/segmentos.csv
docker compose run --rm -v "$PWD/examples/csv:/csv:ro" \
  train modelops train /csv/segmentos.csv --target segmento \
  --algorithm lightgbm --name segmentos --exclude id --categorical codigo_postal
```

`train` es el nombre del servicio; `modelops train ...` reemplaza su comando
predeterminado. Ejecutar solo `docker compose run --rm train` entrena el antiguo
dataset de demostración. El montaje `:ro` permite leer los CSV del host sin
modificarlos; dentro del contenedor la carpeta se llama `/csv`.

Compose configura el experimento compartido `modelops-compose`. Puedes elegir
otro con `--experiment-name`. El nombre `--name segmentos` identifica el modelo
en el Registry. Apunta la **versión devuelta** por el entrenamiento: en un
Registry nuevo será 1; al repetir se crean 2, 3, etc.

### 3. Predecir con la CLI de la imagen

```bash
docker compose run --rm -v "$PWD/examples/csv:/csv:ro" \
  train modelops predict /csv/nuevos_segmentos.csv --model segmentos --version 1
```

Sustituye `1` por la versión que quieras usar. El archivo de predicción lleva
exactamente las variables de entrada, sin objetivo ni columnas excluidas.
No es necesario que las columnas estén en el mismo orden.

La URI `http://mlflow:5000` que muestra el entrenamiento funciona dentro de la
red de Compose. Para ejecutar desde el host el comando de predicción, usa el
recorrido `docker compose run` anterior con el montaje del CSV.

### 4. Desplegar la API del modelo

Escribe en `.env` el nombre y versión devueltos, para conservar la selección
en futuros arranques:

```text
MODELOPS_MODEL_NAME=segmentos
MODELOPS_MODEL_VERSION=1
```

```bash
docker compose up -d --wait api
curl --fail-with-body http://127.0.0.1:8001/health/ready
curl --fail-with-body http://127.0.0.1:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances":[{"edad":24,"gasto":130,"canal":"web","codigo_postal":"001"}]}'
```

La API documenta las variables y sus tipos en <http://127.0.0.1:8001/docs> y
`GET /model/schema`. MLflow y sus modelos están en <http://127.0.0.1:5001>.
Para cambiar de modelo o versión, edita `.env` y repite `up -d --wait api`;
registrar un modelo nuevo no cambia automáticamente el que sirve la API.

### Usar tus propios datos, Iris o regresión

Pon tus CSV en una carpeta local, por ejemplo `data/iris/`. Esa carpeta está
ignorada por Git y no aparece en un clon nuevo. Reemplaza el montaje del ejemplo
por `-v "$PWD/data/iris:/csv:ro"` y los nombres de archivo/columnas por los tuyos.
Para Iris usa `/csv/iris.csv --target species --algorithm lightgbm --name iris_v1`.
Un archivo nuevo de Iris podría ser:

```csv
sepal_length,sepal_width,petal_length,petal_width
5.1,3.5,1.4,0.2
6.7,3.0,5.2,2.3
```

Guárdalo en `data/iris/nuevas_iris.csv` y predice con `/csv/nuevas_iris.csv`,
`--model iris_v1` y la versión registrada. Para una carpeta situada en otra
ruta usa su ruta absoluta en la parte izquierda del montaje. El usuario del
contenedor debe tener permiso de lectura sobre la carpeta y los archivos.

La clasificación binaria y multiclase usa `logistic-regression` o `lightgbm`.
Para regresión numérica se usa `--task regression` con `linear-regression` o
`lightgbm`. Ejemplo incluido con punto y coma, coma decimal, categorías y ausentes:

```bash
docker compose run --rm -v "$PWD/examples/csv:/csv:ro" \
  train modelops train /csv/viviendas.csv --target precio --task regression \
  --algorithm linear-regression --name viviendas --exclude id \
  --delimiter ';' --decimal ','
docker compose run --rm -v "$PWD/examples/csv:/csv:ro" \
  train modelops predict /csv/nuevas_viviendas.csv --model viviendas --version 1 \
  --delimiter ';' --decimal ','
```

Los CSV admitidos son tablas con cabecera única, objetivo explícito y suficientes
filas para train/test. Se aceptan números, categorías y ausentes en variables;
el objetivo debe estar completo. Fechas y texto libre se tratan como categorías,
sin interpretación semántica. No se garantiza entrenar datos arbitrarios o un
modelo útil para cualquier problema. Consulta los [formatos y límites CSV](docs/csv-cli.md).

El Registry Docker es independiente del SQLite local del host. Sus modelos,
metadatos y artefactos se conservan en los volúmenes de PostgreSQL y MLflow.
Los archivos locales del entrenamiento dentro del contenedor puntual se borran
con `--rm`, después de que el registro haya copiado los artefactos a MLflow.
`docker compose down` conserva los volúmenes; `down --volumes` elimina sus datos.

La [guía de Compose](docs/mop-005-compose.md) amplía configuración, copias y
resolución de errores. La auditoría reproducible de CSV se ejecuta con:

```bash
python3 scripts/compose_csv_smoke.py
```

Crea un proyecto aislado, comprueba CLI y HTTP con ambos algoritmos para cada
tarea, formatos alternativos y persistencia, y limpia solo sus propios recursos.
El informe queda en `artifacts/e2e/csv-compose.json`.
Esta auditoría opcional usa Python 3 del host para orquestar Docker; el
entrenamiento, el registro y la inferencia se ejecutan en contenedores.

## Pruebas E2E y robustez — MOP-006

La suite rápida comprueba contratos HTTP, paridad entre entrenamiento,
Registry y API, artefactos y backend corruptos, versión fijada aunque aparezca
una nueva, y ausencia de escrituras por petición. Usa SQLite y archivos
temporales sin Docker ni red externa.

La prueba de infraestructura se ejecuta por separado:

```bash
uv run --locked python scripts/compose_smoke.py \
  --report artifacts/e2e/mop-006.json
```

Comprueba PostgreSQL y proxy de artefactos reales, caída del backend y
persistencia tras recrear la pila. Crea y limpia su propio proyecto temporal;
el informe incluye tiempos informativos de arranque, latencia y bytes de
respuesta. CI ejecuta esta prueba después de la suite rápida y conserva el
informe como artifact. El [manual MOP-006](docs/mop-006-e2e.md) detalla la matriz
de casos, comandos, diagrama, diagnóstico y límites de las garantías.

## Comprobaciones

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

Los tests comprueban instalación, contrato y rechazo de datos inválidos,
estratificación, reproducibilidad, prevención de leakage, ambos modelos,
métricas con ejemplos calculados a mano, persistencia y CLI integrada. También
comprueban el contrato HTTP, salud, carga única y paridad entre entrenamiento,
Registry temporal y respuestas de la API. Usan
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
│   ├── pr-mop-003.md           # Propuesta para revisión del PR de MOP-003
│   └── mop-004-serving.md      # Auditoría, API, contratos y recuperación
├── pyproject.toml              # Paquete, grupo dev y configuración
├── README.md
├── src/modelops/
│   ├── __init__.py              # Paquete y versión instalada
│   ├── tabular.py               # Esquema del ejemplo compartido
│   ├── serving/                 # Factoría FastAPI, predictor, contratos y CLI
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
[ADR-001](docs/adr/001-architecture.md). Infraestructura AWS, observabilidad
y reentrenamiento quedan para entregas
posteriores. MLflow se integra desde una capa externa al entrenamiento;
PostgreSQL se usa en Compose; S3 queda para una entrega posterior.

`.gitignore` excluye secretos y artefactos habituales, pero los archivos que ya
estén versionados no quedan protegidos por estas reglas. Las plantillas
`.env.example` pueden versionarse y deben contener únicamente valores ficticios.
