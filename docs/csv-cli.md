# CSV propios: clasificación y regresión

La CLI `modelops inspect/train/predict` permite trabajar con tablas propias sin
editar Python. El objetivo se elige explícitamente con `--target`. Se mantiene
el ejemplo integrado y los ejecutables anteriores de entrenamiento y tracking.

## Plan ejecutado

1. Validar y cargar CSV con nombres de columnas y formatos configurables.
2. Construir un pipeline para números, categorías y valores ausentes, ajustado
   exclusivamente sobre train.
3. Soportar clasificación binaria, multiclase y regresión con métricas propias.
4. Persistir el contrato con el pipeline y reutilizarlo en CLI, MLflow y HTTP.
5. Verificar compatibilidad, ausencia de fuga de datos y recorridos completos;
   proporcionar ejemplos ejecutables y documentar los límites.

## Lectura y selección de columnas

Por defecto se usa UTF-8 con BOM opcional, coma como separador y punto decimal.
Todos los subcomandos aceptan `--delimiter`, `--encoding` y `--decimal`. Por ejemplo:

```bash
uv run --locked modelops inspect mi_tabla.csv \
  --delimiter ';' --decimal ',' --encoding latin-1
```

`--delimiter tab` selecciona un tabulador. No se adivina el dialecto. Se admiten
comillas CSV estándar, separadores y saltos de línea dentro de campos citados.
Las cabeceras deben ser únicas y no vacías; cada fila debe tener el número de
campos de la cabecera. Se ignoran las líneas vacías. No se corrigen archivos
malformados silenciosamente.

`inspect` lee y valida la estructura del CSV y muestra una tabla alineada con
una fila por columna. `Filas` cuenta los registros leídos y `Columnas` el número
de variables. En cada fila:

- `Tipo` indica `int64` o `float64` cuando los valores presentes se interpretan
  como números, o `string` cuando se interpretan como texto/categorías.
- `Ausentes` cuenta los valores que pandas reconoce como vacíos.
- `Distintos` cuenta los valores únicos presentes.
- `Clasificación` marca columnas completas con varias etiquetas y al menos dos ejemplos
  por etiqueta. `Regresión` marca columnas numéricas finitas con valores.

Estos dos últimos campos son **candidatos**, no una elección ni un resultado de
entrenamiento. La clasificación también requiere que la partición contenga cada
clase; `inspect` no puede garantizarlo para cualquier tamaño de test. Para
entrenar se elige explícitamente la columna con `--target`; para predecir un
valor continuo hay que indicar `--task regression`. `Revisar` avisa sobre
columnas vacías, posibles identificadores y clases con pocos ejemplos. Comprueba
estas señales y corrige el CSV si es necesario antes de entrenar.

La tabla de terminal calcula el ancho de las columnas y parte nombres largos en
líneas continuadas, para que sigan completos y legibles. `--json` conserva la
salida estructurada para automatización.

`train` exige objetivo, algoritmo y nombre de registro. Repite `--exclude` para
ignorar identificadores u otras variables. El objetivo nunca puede excluirse.
Las demás columnas forman la entrada. Los nombres, incluso con espacios o
caracteres especiales, se conservan.

Si `train` se ejecuta sin `--experiment-name` ni la variable
`MLFLOW_EXPERIMENT_NAME`, crea o reutiliza un experimento con nombre
`modelops-<algoritmo>-<modelo>` (por ejemplo, `modelops-lightgbm-iris_v1`).
Así, cada modelo tiene un experimento predeterminado coherente con el nombre de
registro. Puedes proporcionar `--experiment-name` para compartir un experimento;
si defines `MLFLOW_EXPERIMENT_NAME`, se respeta ese valor como configuración
explícita.

Se infieren como numéricas las columnas cuyos valores presentes se pueden leer
como números. El resto se considera categórico. `--categorical codigo_postal`
conserva códigos numéricos como categorías, incluyendo ceros iniciales.
`--numeric importe` exige conversión numérica y rechaza texto inválido. Ambas
opciones se pueden repetir y no pueden solaparse ni nombrar columnas excluidas.
Revisa estas decisiones: un texto erróneo en una columna numérica podría hacer
que se infiera como categórica si no se especifica `--numeric`.

## Preprocesamiento y tareas

El pipeline persistido contiene selección de columnas, imputación, codificación
y estimador. La partición se realiza antes de ajustar cualquier transformación:

- Números: mediana de train para ausentes; una columna enteramente vacía en
  train se conserva y se rellena con cero. Los infinitos se rechazan.
- Categorías: ausentes sustituidos por `__modelops_missing__`; one-hot aprendido
  en train con un máximo de 100 salidas por columna (las categorías menos
  frecuentes se agrupan). Una categoría desconocida se codifica como ceros.
  El marcador de ausentes se trata igual que esa cadena literal si ya existe.
- Baseline: escalado numérico con estadísticas de train. LightGBM recibe la
  matriz transformada, sin restricciones sobre nombres externos de columnas.

El objetivo no admite ausentes. Los marcadores NA de pandas se interpretan como
ausentes, también en categorías y objetivos. Las etiquetas de clasificación se
leen como texto: `01` y `1` son distintas. Se ordenan lexicográficamente y se
codifican como `0..K-1`; `class_names[codigo]` conserva su valor original.

| Tarea | Opción | Algoritmos | Métricas |
| --- | --- | --- | --- |
| Binaria | `--task classification` (por defecto) | `logistic-regression`, `lightgbm` | accuracy, precision, recall, F1, ROC-AUC |
| Multiclase | `--task classification` | `logistic-regression`, `lightgbm` | accuracy, precision/recall/F1 macro, ROC-AUC OVR macro |
| Regresión | `--task regression` | `linear-regression`, `lightgbm` | MAE, RMSE, R² |

En binaria, la clase positiva matemática es el código 1 y su etiqueta se imprime
al entrenar. No se infiere la clase de interés del negocio. Las métricas
indefinidas se guardan como `null`. La regresión exige un objetivo numérico
finito; no se elige automáticamente a partir de la cardinalidad del objetivo.

La semilla predeterminada es 42 y `test_size=0.2`; se ajustan con `--seed` y
`--test-size`. Se exigen al menos cuatro filas y dos filas en cada partición.
La clasificación usa partición estratificada, con al menos dos ejemplos por
clase en origen y todas las clases presentes en train y test. Aumenta los datos
o ajusta la fracción si no es viable. La regresión usa una partición aleatoria
sin estratificar. Test mide el algoritmo solicitado una vez; no selecciona un
ganador ni ajusta hiperparámetros.

## Ejemplos ejecutables

Los CSV incluidos son sintéticos. El ejemplo binario anterior sigue disponible:

```bash
uv run --locked modelops train examples/csv/clientes.csv --target churn \
  --algorithm lightgbm --name churn --exclude id_cliente \
  --storage-dir artifacts/csv-demo --experiment-name csv-demo
uv run --locked modelops predict examples/csv/nuevos_clientes.csv \
  --model churn --version 1 --storage-dir artifacts/csv-demo
```

Clasificación multiclase con categorías y valores ausentes:

```bash
uv run --locked modelops inspect examples/csv/segmentos.csv
uv run --locked modelops train examples/csv/segmentos.csv --target segmento \
  --task classification --algorithm logistic-regression --name segmentos \
  --exclude id --categorical codigo_postal \
  --storage-dir artifacts/csv-general --experiment-name csv-general
uv run --locked modelops predict examples/csv/nuevos_segmentos.csv \
  --model segmentos --version 1 --storage-dir artifacts/csv-general
```

Regresión con punto y coma y coma decimal:

```bash
uv run --locked modelops train examples/csv/viviendas.csv --target precio \
  --task regression --algorithm linear-regression --name viviendas \
  --exclude id --delimiter ';' --decimal ',' \
  --storage-dir artifacts/csv-general --experiment-name csv-general
uv run --locked modelops predict examples/csv/nuevas_viviendas.csv \
  --model viviendas --version 1 --delimiter ';' --decimal ',' \
  --storage-dir artifacts/csv-general
```

Usa la versión realmente devuelta por train si ya existe un modelo con ese
nombre. El comando impreso incluye la versión, backend efectivo y formato del
CSV de entrenamiento; adapta el formato si el CSV de inferencia usa otro.

## Predicción y contrato

Los CSV de inferencia deben incluir exactamente las variables guardadas, sin
objetivo ni columnas excluidas. Se admite otro orden de columnas y se conserva
el orden de filas. Los valores ausentes se imputan usando train; una columna
completamente omitida es un error. Los tipos se leen del contrato del modelo,
sin volver a inferirlos con los datos nuevos.

`--json` reserva stdout para JSON; progreso y errores van a stderr. Los fallos
esperables devuelven código 2 y una indicación para corregirlos. No hay preguntas
interactivas. La salida de clasificación conserva su formato:

```json
{"schema_version":"1","model":{"name":"segmentos","version":"1","uri":"models:/segmentos/1"},"classes":[0,1,2],"class_names":["A","B","C"],"predictions":[{"label":0,"probabilities":[0.8,0.1,0.1]}]}
```

Las probabilidades siguen el orden de `classes` y `class_names`, que respeta
`pipeline.classes_`. En regresión la salida es:

```json
{"schema_version":"2","task":"regression","model":{"name":"viviendas","version":"1","uri":"models:/viviendas/1"},"predictions":[{"value":180000.0}]}
```

El contrato de datos v2 viaja dentro de `modelops_contract_` con tarea, nombres,
tipos y política de ausentes. Se copia a `contracts.json` local y a
`pipeline/contract.json` en MLflow. Se conservan pipeline completo, parámetros,
opciones de lectura, métricas, posiciones de partición, huella del dataset y
checksums de artefactos. Cada entrenamiento crea una ejecución nueva.

La CLI y la API usan el mismo contrato. Para servir regresión, por ejemplo:

```bash
uv run --locked modelops-serve --model-name viviendas --model-version 1 \
  --storage-dir artifacts/csv-general --port 8002
curl --fail-with-body http://127.0.0.1:8002/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances":[{"metros":85,"habitaciones":3,"zona":"centro"},{"metros":null,"habitaciones":2,"zona":"nueva"}]}'
```

`GET /model/schema` expone el contrato activo; `/docs` y OpenAPI muestran sus
columnas numéricas/categóricas y la admisión de `null`. Las columnas son siempre
obligatorias. Una entrada inválida devuelve 422 y conserva readiness. Un modelo
no disponible devuelve 503; un fallo interno de inferencia devuelve 500 y
retira readiness. El modelo se carga una vez al arrancar.

## Compatibilidad y límites

Los contratos v1 siguen siendo numéricos, binarios y sin ausentes. Los modelos
antiguos del ejemplo sin contrato embebido conservan el esquema conocido.
No se reconstruye un contrato arbitrario para modelos externos sin metadatos.
No se modifican versiones previas al registrar nuevas.

Se mantienen las opciones MLflow `--storage-dir`, `--tracking-uri` y
`--experiment-name`, SQLite local y servidor HTTP(S). No hay promoción automática
ni selección latest; registrar un modelo tampoco cambia la versión de la API.
Si el registro falla después de persistir, se informa del directorio conservado.
Un fallo parcial puede haber creado una versión: revisa el Registry antes de
repetir el entrenamiento.

«CSV propio» significa una tabla válida para aprendizaje supervisado. Los CSV
se leen completos en memoria y la predicción CLI procesa el lote completo;
HTTP limita cada petición a 256 filas. Texto libre, fechas, imágenes, series
temporales, multilabel y aprendizaje no supervisado requieren ingeniería o
flujos específicos; las cadenas se tratan como categorías, sin interpretación
semántica. No se eliminan duplicados ni identificadores automáticamente. La
partición aleatoria no garantiza separación temporal o por entidad. El soporte
de un formato no garantiza que sus datos permitan entrenar un modelo útil.

## Verificación

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked --offline pytest
uv build --offline
# Auditoría real de CSV en la imagen y los servicios de Compose:
python3 scripts/compose_csv_smoke.py
```

`tests/test_csv_generalization.py` cubre los cuatro recorridos multiclase y
regresión (ambos algoritmos), CSV con dialecto y codificación distintos,
persistencia y carga real de MLflow, paridad CLI/pipeline/HTTP, ausentes,
categorías desconocidas, entradas dispersas y ausencia de fuga hacia el
preprocesamiento. Las pruebas anteriores cubren los esquemas binarios, versiones,
fallos de registro y compatibilidad con modelos del ejemplo.

### Resultados comprobados — 2026-11-05

- **213 pruebas pasan**, con el paquete reconstruido e instalado sin
  modo editable: 55,83 s. Incluyen los 44 casos de las dos suites CSV, el
  nombre automático del experimento y la alineación de `inspect`.
- Ruff, comprobación de formato y `git diff --check` correctos. Wheel y sdist
  construidos con `uv build --offline`.
- Los ejemplos `segmentos.csv` y `viviendas.csv` se han inspeccionado, entrenado,
  registrado y usado para predecir mediante CLI en procesos separados. Informe
  local: `artifacts/csv-general-20261105T165708984146/demo-report.json`.
- Smoke Docker Compose con imagen reconstruida: **passed**; persistencia y
  limpieza: **passed**. Comprueba el flujo integrado, paridad de predicciones,
  caída del backend y recreación de la pila. Informe local:
  `artifacts/e2e/audit-legacy-compose.json`.
- Auditoría CSV sobre la imagen de Compose: siete recorridos completos de
  inspección, entrenamiento, registro, predicción CLI y API con paridad de
  resultados. Incluyen ambos algoritmos por tarea y un CSV externo con
  Latin-1, tabulador, coma decimal, categorías nuevas y valores ausentes.
  Se comprueban versiones explícitas y rechazo de entradas inválidas. Script
  reproducible: `scripts/compose_csv_smoke.py`; informe local:
  `artifacts/e2e/csv-compose.json`. También se ha ejecutado el comando del
  README con el experimento compartido de Compose.
  Persistencia tras recrear la pila, versión fijada y limpieza: **passed**.
  Esta auditoría queda incorporada al workflow de CI para futuras ejecuciones.

Las pruebas HTTP se ejecutaron fuera del aislamiento de comandos porque este
bloqueaba la comunicación interna de asyncio/TestClient. Se mantienen los dos
avisos de dependencias sobre Starlette/httpx y SQLAlchemy/MLflow.
