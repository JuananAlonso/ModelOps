# ModelOps

Base de una plataforma MLOps en Python 3.12, organizada como un monolito modular.
El diseño prevé procesos separados para entrenamiento e inferencia; MOP-001
incluye únicamente el paquete instalable, las herramientas de desarrollo, CI
y la documentación de arquitectura.

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
El paquete no tiene dependencias de ejecución en MOP-001.

## Comprobaciones

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

Los tests comprueban los metadatos de la distribución instalada y la importación
desde un directorio ajeno al repositorio, con Python en modo aislado. Pytest
utiliza `importlib`; no se añade `src/` a `PYTHONPATH` para evitar que una
instalación incorrecta quede oculta.

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
├── docs/adr/001-architecture.md # Decisiones y propuestas arquitectónicas
├── pyproject.toml              # Paquete, grupo dev y configuración
├── README.md
├── src/modelops/__init__.py     # Paquete y versión instalada
├── tests/test_package.py        # Instalación e importación
└── uv.lock                     # Dependencias resueltas y hashes
```

Las decisiones del stack y sus riesgos están en
[ADR-001](docs/adr/001-architecture.md). Entrenamiento, serving, servicios locales,
infraestructura AWS, observabilidad y reentrenamiento quedan para entregas
posteriores; sus bibliotecas no se instalan en este bootstrap.

`.gitignore` excluye secretos y artefactos habituales, pero los archivos que ya
estén versionados no quedan protegidos por estas reglas. Las plantillas
`.env.example` pueden versionarse y deben contener únicamente valores ficticios.
