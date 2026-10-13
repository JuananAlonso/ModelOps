# MOP-003: tracking y Registry local con MLflow

Refs #3. Base: `main`. Rama: `feat/mop-003-mlflow-tracking`.

La entrega incorpora un adaptador que consume `TrainingResult` y los artefactos
locales de MOP-002. Crea un run padre y dos hijos, registra parámetros, métricas
y procedencia y conserva pipelines completos con firma y ejemplo de entrada.
Cada candidato crea una versión inicial o sucesiva bajo un nombre estable,
con referencia inequívoca al run y a la fuente de sus artefactos.

La lógica de entrenamiento y evaluación permanece independiente de MLflow y
el comando local original sigue funcionando. Tracking, consulta y UI comparten
SQLite y un artifact store local con rutas absolutas. La UI solo escucha en
loopback. No hay promoción, alias de producción, AWS, PostgreSQL ni serving.

Se fija MLflow 3.16.1 mediante uv, se conserva el lockfile generado y se añaden
tests con bases temporales independientes, sin servidor ni Internet. README y
guía técnica explican IDs, firma, límites de SQLite, seguridad y recuperación.
ADR-001 no se modifica.

## Validación

Validación local en Linux CPU, Python 3.12.15, uv 0.13.0 y MLflow 3.16.1:

- Ruff lint y formato: pasan.
- Pytest completo en modo offline: 100 tests pasan en 29,74 s; una advertencia
  interna de MLflow/SQLAlchemy 2.1 sobre `noload`, sin fallos.
- Instalación no editable como CI: 100 tests pasan en 24,22 s con la misma
  advertencia. Se conservan los 74 tests originales y se añaden 26 casos.
- Wheel y distribución fuente: generados correctamente.
- CLI real con semillas 42 y 123: mismo experimento `modelops-tabular`,
  dos runs padres y cuatro hijos, con métricas consultables mediante la CLI.
- Registry: versiones 1 y 2 de `modelops-baseline` y `modelops-lightgbm`.
  Los cuatro pipelines recuperados producen etiquetas idénticas a los
  originales y diferencia máxima de probabilidades 0,0.
- UI temporal en `127.0.0.1`: verificados mediante HTTP el experimento,
  los runs, las versiones y la lista de artefactos del mismo SQLite.
  El proceso se detuvo al finalizar la comprobación.

La primera preparación/descarga de dependencias tardó 25 min 04 s en la red
de esta sesión; la instalación posterior desde caché tardó 1,80 s. El lockfile
incluye 98 paquetes resueltos frente a los 20 de MOP-002. Estos tiempos dependen
de red y caché y no son un benchmark de CI.

Al lanzar la segunda CLI antes de terminar la primera inicialización de SQLite,
MLflow observó un esquema en migración y la rechazó. Después de finalizar
la inicialización, repetir la ejecución de forma secuencial pasó. Se documenta
el requisito de inicializar con un único proceso; no se borró ni forzó la base.

La exportación avisa de cloudpickle y de pip ausente en el `conda.yaml` generado.
Se documentan ambos: artefactos fiables y reproducción mediante uv.lock.
Se utilizó el uv 0.13.0 de `/tmp/modelops-tools`; el uv 0.12.10 del PATH del
sistema sigue sin cumplir el pin heredado. El README describe la instalación.

El informe E2E, las salidas de ambas CLI y la comparación se conservan localmente
en `artifacts/mlflow/mop-003-*.json`, ignorados por Git. La ejecución remota de
GitHub Actions queda pendiente de publicar el cambio.

## Revisión

La entrega requiere revisión técnica y evaluación de comprensión antes de ser
considerada completa. No autoriza merge ni cierre automático de la issue #3.
