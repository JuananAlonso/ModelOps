# ModelOps

Base de una plataforma MLOps en Python 3.12, organizada como un monolito modular.
El diseño prevé procesos separados para entrenamiento e inferencia. MOP-001
estableció paquete, entorno y CI; MOP-002 añade entrenamiento y evaluación local
de clasificación binaria con un baseline y LightGBM.

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
el lockfile. No se instalan MLflow, FastAPI ni herramientas de cloud.

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
automática, MLflow, serving ni servicios externos en esta entrega.

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
│   └── pr-mop-002.md           # Descripción preparada para revisión del PR
├── pyproject.toml              # Paquete, grupo dev y configuración
├── README.md
├── src/modelops/
│   ├── __init__.py              # Paquete y versión instalada
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
posteriores. MLflow se integrará en MOP-003 desde una capa externa al entrenamiento.

`.gitignore` excluye secretos y artefactos habituales, pero los archivos que ya
estén versionados no quedan protegidos por estas reglas. Las plantillas
`.env.example` pueden versionarse y deben contener únicamente valores ficticios.
