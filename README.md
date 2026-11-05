# ModelOps

[![CI](https://github.com/JuananAlonso/ModelOps/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JuananAlonso/ModelOps/actions/workflows/ci.yml)

Entrena modelos con tus CSV, registra sus versiones en MLflow y realiza
predicciones por terminal o mediante una API HTTP. Puedes ejecutar todo el
recorrido con Docker Compose o trabajar localmente con Python y uv.

- **Clasificación binaria y multiclase** con LogisticRegression y LightGBM.
- **Regresión** con LinearRegression y LightGBM.
- Variables numéricas y categóricas, con imputación de valores ausentes.
- Preprocesamiento y esquema guardados con cada modelo para reutilizarlos al predecir.
- Métricas, parámetros y versiones consultables en MLflow.

[Inicio rápido](#inicio-rápido-con-docker) · [Tus CSV](#usar-tus-propios-csv) ·
[Comandos](#referencia-de-la-cli) · [Configuración](#configuración) ·
[Problemas frecuentes](#problemas-frecuentes) · [Desarrollo](#desarrollo)

## Inicio rápido con Docker

Necesitas **Git, Docker y Docker Compose**, acceso al motor de Docker e Internet
para construir la imagen por primera vez. No necesitas instalar Python ni uv
en el host. Los comandos siguientes usan Bash y se ejecutan desde la raíz del
proyecto.

El ejemplo usa datos sintéticos incluidos en el repositorio: predice el segmento
de un cliente a partir de su edad, gasto, canal y código postal.

### 1. Construir la imagen y arrancar MLflow

```bash
git clone https://github.com/JuananAlonso/ModelOps.git
cd ModelOps
# Conserva .env si ya existe.
test -f .env || cp .env.example .env
docker compose build api
docker compose up -d --wait mlflow
```

La imagen de `api` también la usan los servicios `train` y `mlflow`. Compose
arranca PostgreSQL como dependencia de MLflow. Espera a registrar un modelo
antes de iniciar la API: esta necesita cargar una versión existente.

### 2. Inspeccionar los datos y entrenar

```bash
docker compose run --rm -v "$PWD/examples/csv:/csv:ro" \
  train modelops inspect /csv/segmentos.csv

docker compose run --rm -v "$PWD/examples/csv:/csv:ro" \
  train modelops train /csv/segmentos.csv --target segmento \
  --algorithm lightgbm --name segmentos --exclude id --categorical codigo_postal
```

`inspect` muestra tipos, ausentes y valores distintos por columna. Sus candidatos
a objetivo son orientativos; tú eliges qué predecir con `--target`.

`train` ajusta el preprocesamiento solo con los datos de entrenamiento, evalúa
sobre test y registra el modelo. **Apunta la versión devuelta**: en un Registry
nuevo será `1`; si repites el entrenamiento se creará otra versión.

En estos comandos, `train` es el servicio de Compose y `modelops ...` es el
comando ejecutado dentro del contenedor. El montaje `:ro` deja leer la carpeta
del host como `/csv` sin modificar sus archivos. Ejecutar únicamente
`docker compose run --rm train` usa el dataset de demostración integrado.

### 3. Predecir con el modelo registrado

```bash
docker compose run --rm -v "$PWD/examples/csv:/csv:ro" \
  train modelops predict /csv/nuevos_segmentos.csv --model segmentos --version 1
```

Sustituye `1` por la versión que quieras utilizar. La salida muestra la categoría
estimada y sus probabilidades para cada fila. En regresión muestra un valor
numérico por fila.

### 4. Servir el modelo mediante HTTP

Edita estas variables en `.env` con el nombre y la versión registrados:

```dotenv
MODELOPS_MODEL_NAME=segmentos
MODELOPS_MODEL_VERSION=1
```

Arranca la API y realiza una predicción:

```bash
docker compose up -d --wait api
curl --fail-with-body http://127.0.0.1:8001/health/ready
curl --fail-with-body http://127.0.0.1:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances":[{"edad":24,"gasto":130,"canal":"web","codigo_postal":"001"}]}'
```

| Servicio | Dirección | Para qué sirve |
| --- | --- | --- |
| API interactiva | <http://127.0.0.1:8001/docs> | Consultar el esquema y probar peticiones |
| MLflow | <http://127.0.0.1:5001> | Revisar experimentos, métricas y versiones |
| Esquema del modelo | <http://127.0.0.1:8001/model/schema> | Obtener columnas, tipos y política de ausentes |

La API devuelve códigos de clase y probabilidades; `class_names` permite
recuperar las etiquetas originales. Su esquema se adapta al modelo cargado.

## Usar tus propios CSV

Guarda los archivos en una carpeta local, por ejemplo `data/mi_proyecto/`, y
cambia el montaje y las opciones del entrenamiento:

```bash
docker compose run --rm -v "$PWD/data/mi_proyecto:/csv:ro" \
  train modelops train /csv/entrenamiento.csv --target resultado \
  --algorithm lightgbm --name mi-modelo

docker compose run --rm -v "$PWD/data/mi_proyecto:/csv:ro" \
  train modelops predict /csv/nuevos.csv --model mi-modelo --version 1
```

La carpeta debe existir y ser legible por el usuario del contenedor. Puedes
usar una ruta absoluta externa en la parte izquierda del montaje. `data/`,
`artifacts/` y `.env` están ignorados por Git; Iris no se distribuye con el clon.

### Preparar los datos

- **Entrenamiento:** cabeceras únicas, una columna objetivo y al menos una variable.
  El objetivo no admite ausentes. Usa `--exclude` para ignorar identificadores
  u otras columnas; la columna objetivo se separa automáticamente.
- **Predicción:** exactamente las variables guardadas con el modelo, sin objetivo
  ni columnas excluidas. Se admite otro orden de columnas. Un valor ausente se
  puede imputar; una columna completamente omitida es un error.
- **Tipos:** los números se detectan automáticamente y el texto se trata como
  categorías. Usa `--categorical codigo_postal` para conservar códigos como
  categorías y `--numeric importe` para exigir números válidos.
- **Formato:** por defecto, UTF-8, coma y punto decimal. Todos los subcomandos
  admiten `--delimiter ';'`, `--decimal ','` y `--encoding latin-1`;
  `--delimiter tab` selecciona un tabulador.

Por ejemplo, para Iris el objetivo es `species`. El CSV de predicción contiene
solo estas cuatro columnas:

```csv
sepal_length,sepal_width,petal_length,petal_width
5.1,3.5,1.4,0.2
6.7,3.0,5.2,2.3
```

### Elegir la tarea y el algoritmo

| Problema | Tarea | Algoritmos | Métricas |
| --- | --- | --- | --- |
| Clasificación binaria | `--task classification` (por defecto) | `logistic-regression`, `lightgbm` | Accuracy, precision, recall, F1, ROC-AUC |
| Clasificación multiclase | `--task classification` | `logistic-regression`, `lightgbm` | Accuracy, precision/recall/F1 macro, ROC-AUC OVR macro |
| Regresión numérica | `--task regression` | `linear-regression`, `lightgbm` | MAE, RMSE, R² |

Este ejemplo de regresión también está incluido en el repositorio:

```bash
docker compose run --rm -v "$PWD/examples/csv:/csv:ro" \
  train modelops train /csv/viviendas.csv --target precio --task regression \
  --algorithm linear-regression --name viviendas --exclude id \
  --delimiter ';' --decimal ','

docker compose run --rm -v "$PWD/examples/csv:/csv:ro" \
  train modelops predict /csv/nuevas_viviendas.csv --model viviendas --version 1 \
  --delimiter ';' --decimal ','
```

Se necesitan suficientes filas para separar train y test. En clasificación,
cada clase debe tener al menos dos ejemplos y aparecer en ambas particiones.
La semilla predeterminada es `42` y la fracción de test `0.2`; puedes ajustarlas
con `--seed` y `--test-size`.

El alcance es aprendizaje supervisado sobre tablas en memoria. El texto libre
y las fechas se tratan como categorías; series temporales, objetivos múltiples
y aprendizaje no supervisado requieren otros flujos. El sistema no selecciona
el objetivo, el algoritmo ni los hiperparámetros automáticamente. Los detalles
de validación y preprocesamiento están en la [guía CSV](docs/csv-cli.md).

## Referencia de la CLI

| Comando | Función | Opciones principales |
| --- | --- | --- |
| `modelops inspect archivo.csv` | Examinar estructura y calidad de datos | `--delimiter`, `--decimal`, `--encoding` |
| `modelops train archivo.csv` | Entrenar y registrar una versión | `--target`, `--algorithm`, `--name`, `--task`, `--exclude` |
| `modelops predict archivo.csv` | Predecir con una versión registrada | `--model`, `--version` |

Añade `--json` para automatizar: el resultado va a stdout y los mensajes a
stderr. Los errores esperables devuelven código `2`. Consulta todas las opciones
sin instalar nada en el host:

```bash
docker compose run --rm train modelops --help
docker compose run --rm train modelops train --help
```

## Configuración

El **modelo** (`--name`) identifica lo que cargas para predecir. El
**experimento** (`--experiment-name`) agrupa ejecuciones en MLflow.

| Configuración | Docker Compose | Ejecución local |
| --- | --- | --- |
| Backend de MLflow | Servidor con PostgreSQL y volumen de artefactos | SQLite y archivos en `artifacts/mlflow` |
| Experimento predeterminado | `modelops-compose` | `modelops-<algoritmo>-<modelo>` para `modelops train` |
| Modelo servido por la API | `MODELOPS_MODEL_NAME` en `.env` | `modelops-serve --model-name` |
| Versión servida | `MODELOPS_MODEL_VERSION` en `.env` | `modelops-serve --model-version` |

Puedes elegir un experimento con `--experiment-name` o `MLFLOW_EXPERIMENT_NAME`.
Un argumento explícito prevalece sobre la variable de entorno. `--tracking-uri`
selecciona un servidor o un SQLite; `--storage-dir` establece las rutas locales
predeterminadas. Un URI configurado explícitamente tiene prioridad sobre esas
rutas.

Los clientes de Compose usan `http://mlflow:5000`, accesible desde su red interna.
El Registry de Docker y el local son independientes. Usa los comandos
`docker compose run` para predecir con un modelo registrado dentro de Compose.

En `.env` también puedes cambiar `MODELOPS_API_PORT` (8001),
`MODELOPS_MLFLOW_PORT` (5001) y `POSTGRES_PASSWORD`. El despliegue incluido está
orientado a desarrollo local: publica puertos en `127.0.0.1` y no proporciona
autenticación ni TLS. Configuración, copias y recuperación se detallan en la
[guía de operación con Docker](docs/mop-005-compose.md).

## Gestionar modelos y servicios

Cada entrenamiento crea una nueva versión. La API carga una versión al arrancar:
para cambiarla, edita nombre y versión en `.env` y ejecuta
`docker compose up -d --wait api`. Esto también permite volver a una versión
anterior.

```bash
docker compose ps                    # Estado de los servicios
docker compose logs --tail 100 api   # Diagnóstico de la API
docker compose logs --tail 100 mlflow
docker compose down                 # Detener y conservar los volúmenes
```

PostgreSQL conserva los metadatos y MLflow los artefactos en volúmenes Docker.
**`docker compose down --volumes` elimina esos datos.** Los archivos temporales
del contenedor de entrenamiento se borran con `--rm`; un registro completado
ya ha copiado sus artefactos a MLflow.

La API admite lotes de 1 a 256 filas. `GET /health/live` indica si responde el
proceso; `GET /health/ready` confirma que el modelo está cargado y utilizable.

## Problemas frecuentes

| Síntoma | Qué revisar |
| --- | --- |
| API devuelve 503 o aparece `unhealthy` | Entrena primero, comprueba nombre y versión en `.env`, recrea API y consulta sus logs. |
| `El experimento ya existe con otro artifact location` | Reutiliza su almacenamiento original o elige otro `--experiment-name`. |
| Error al excluir el objetivo | Quita `--exclude` de esa columna; `--target` ya la separa de las variables. |
| Columnas ausentes o extra al predecir | Conserva exactamente las variables del contrato, sin objetivo ni identificadores excluidos. Consulta `/model/schema`. |
| No se puede leer el CSV en Docker | Comprueba la ruta del host, el montaje y los permisos de lectura de carpeta y archivo. |
| Puerto ocupado | Cambia los puertos en `.env` y vuelve a arrancar los servicios. |
| La partición no conserva todas las clases | Añade ejemplos por clase o ajusta `--test-size`; revisa el objetivo elegido. |

## Ejecución local sin Docker

Necesitas [uv 0.13.0](https://docs.astral.sh/uv/getting-started/installation/).
uv instala el Python 3.12 fijado por el proyecto y sus dependencias:

```bash
uv python install
uv sync --locked
uv run --locked modelops inspect examples/csv/segmentos.csv
uv run --locked modelops train examples/csv/segmentos.csv --target segmento \
  --algorithm lightgbm --name segmentos --exclude id --categorical codigo_postal
uv run --locked modelops predict examples/csv/nuevos_segmentos.csv \
  --model segmentos --version 1
```

`--locked` exige que `uv.lock` esté actualizado y evita modificarlo al ejecutar.
Para abrir MLflow o servir el modelo local en otra terminal:

```bash
uv run --locked modelops-mlflow-ui --port 5000
# En otra terminal, usa la versión devuelta al entrenar:
uv run --locked modelops-serve --model-name segmentos --model-version 1 --port 8000
```

La API local estará en <http://127.0.0.1:8000/docs>. Si utilizaste
`--storage-dir` o `--tracking-uri`, usa la misma configuración al predecir o
servir. `MLFLOW_EXPERIMENT_NAME` permite fijar un experimento local explícito.

## Estructura del proyecto

Archivos y directorios principales:

```text
ModelOps/
├── .github/workflows/ci.yml     # Pruebas, empaquetado y auditorías Docker
├── .env.example                # Plantilla de configuración local
├── .gitignore                  # Datos, secretos y artefactos excluidos de Git
├── .python-version             # Versión fijada de Python
├── Dockerfile                  # Imagen compartida por la CLI, MLflow y la API
├── compose.yaml                # Servicios y volúmenes de Docker Compose
├── pyproject.toml              # Dependencias, ejecutables y herramientas
├── uv.lock                     # Versiones y hashes de dependencias
├── README.md                   # Guía de uso
├── docs/                       # Contratos, arquitectura y operación
├── examples/csv/               # CSV sintéticos de entrenamiento y predicción
├── scripts/
│   ├── compose_csv_smoke.py     # Auditoría de CSV propios en Docker
│   └── compose_smoke.py         # Auditoría de servicios y recuperación
├── src/modelops/
│   ├── cli.py                  # Comandos inspect, train y predict
│   ├── csv_data.py             # Lectura y validación de CSV
│   ├── model_contract.py       # Esquema compartido por CLI y API
│   ├── tabular.py              # Esquema del dataset de demostración
│   ├── training/               # Partición, preprocesamiento, modelos y métricas
│   ├── tracking/               # Configuración y registro en MLflow
│   └── serving/                # API HTTP, validación e inferencia
└── tests/                      # Pruebas unitarias y de integración
```

Las carpetas locales `data/` y `artifacts/` se crean al trabajar con datos y
entrenamientos y permanecen ignoradas por Git.

## Desarrollo

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv build
```

Para comprobar la integración real con Docker, con Python disponible en el host:

```bash
python3 scripts/compose_csv_smoke.py
uv run --locked python scripts/compose_smoke.py
```

Las auditorías crean proyectos Compose aislados y limpian sus propios recursos.
Los informes quedan en `artifacts/e2e/`. CI ejecuta las pruebas, el empaquetado y
las auditorías de Docker en cada push y pull request.

Para contribuir, abre una issue o un pull request con el problema, el cambio y
su validación. Actualiza los ejemplos y la documentación cuando cambie el uso.
Si modificas dependencias, versiona juntos `pyproject.toml` y `uv.lock`.

## Documentación técnica

- [Formato CSV, etiquetas, preprocesamiento y contratos](docs/csv-cli.md).
- [Entrenamiento y evaluación](docs/mop-002-training.md).
- [Tracking y registro de modelos](docs/mop-003-mlflow.md).
- [API HTTP y recuperación ante fallos](docs/mop-004-serving.md).
- [Docker Compose, persistencia y operación](docs/mop-005-compose.md).
- [Pruebas de integración y robustez](docs/mop-006-e2e.md).
- [Decisiones de arquitectura](docs/adr/001-architecture.md).

## Licencia

El repositorio todavía no declara una licencia en un archivo `LICENSE`.
