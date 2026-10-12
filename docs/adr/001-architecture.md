# ADR-001 — Arquitectura inicial de ModelOps

- **Fecha:** 2026-10-12
- **Estado:** decisiones de arquitectura aprobadas para la dirección del proyecto;
  propuestas de detalle revisables y ejecución por fases.
- **Entrega:** MOP-001 — Bootstrap.

## Contexto y alcance

ModelOps necesita una base reproducible y mantenible para desarrollar una
plataforma MLOps. Todavía no se han definido el dataset, los objetivos de calidad
del modelo, la carga de inferencia, los SLO, el presupuesto ni los requisitos
de privacidad. Se busca equilibrar separación de responsabilidades y coste
operativo mientras se valida el primer caso de uso.

MOP-001 materializa exclusivamente el paquete Python instalable, la gestión del
entorno, lint, tests y CI. Este ADR registra también la dirección tecnológica
de las próximas fases. Aprobar una tecnología en este documento no equivale a
instalarla, desplegarla ni validar sus prestaciones para un caso de uso real.

## Decisiones aprobadas

Las elecciones siguientes forman la dirección arquitectónica solicitada. Su
sustitución requiere actualizar este ADR o crear uno que lo sustituya con
evidencias. Los parámetros, versiones y topologías pendientes se separan en
la sección de propuestas revisables.

| Decisión | Justificación | Alternativas consideradas | Consecuencias y riesgos |
| --- | --- | --- | --- |
| Monolito modular con procesos separados para entrenamiento e inferencia | Mantener un único repositorio y contratos compartidos reduce coordinación; separar procesos permite gestionar sus recursos y ciclos de vida. | Un único proceso para todo; microservicios desde el inicio. | Facilita refactorizar límites, pero exige evitar acoplamiento entre módulos. Las cargas de entrenamiento deben aislarse del servicio de inferencia. |
| Python 3.12 como lenguaje principal | Permite compartir contratos y lógica en un ecosistema adecuado para ML y servicios. | Otras versiones de Python; varios lenguajes. | El bootstrap solo soporta 3.12. Las futuras bibliotecas y sus componentes nativos deberán verificar compatibilidad antes de incorporarse. |
| uv como herramienta única de gestión de Python | Gestiona intérprete, entorno, dependencias, lockfile y construcción mediante una interfaz común en local y CI. | pip con venv y herramientas de lock separadas; Poetry; Conda. | Se versionan `.python-version` y `uv.lock`, y se fija uv y el backend. La primera instalación necesita red; actualizar estas versiones requiere repetir las comprobaciones. |
| LightGBM para el primer modelo | Ofrece un candidato práctico para un primer modelo de datos tabulares con coste de entrenamiento controlable. | Regresión y árboles de scikit-learn; XGBoost; redes neuronales. | La adecuación a datos tabulares es una hipótesis pendiente de confirmar. La selección necesita una comparación con una baseline y métricas de negocio; no garantiza calidad ni equidad. |
| FastAPI y Pydantic para serving | Permiten declarar contratos, validación de entradas y una API HTTP documentable en Python. | Flask con validación manual; gRPC; serving gestionado. | Deben versionarse contratos y comprobarse serialización, límites y validación. El trabajo CPU intensivo requiere medir concurrencia y dimensionar los procesos. |
| MLflow para experimentos y registro de modelos | Centraliza trazabilidad de ejecuciones, artefactos y versiones del modelo. | Registro propio; herramientas gestionadas de experimentación. | Hay que definir promoción y acceso al registro, fijar versiones y comprobar compatibilidad de los formatos. El registro no decide por sí solo qué modelo pasa a producción. |
| PostgreSQL para los metadatos de MLflow | Separa metadatos del almacenamiento de artefactos y permite operación compartida. | SQLite local; otros backends SQL compatibles. | Requiere migraciones, copias de seguridad, recuperación y gestión de conexiones. Las actualizaciones de MLflow pueden requerir cambios del esquema. |
| S3 para artefactos en cloud | Separa los objetos grandes del backend de metadatos y se integra con el destino AWS elegido. | Disco local; otros almacenes de objetos. | Deben resolverse permisos, cifrado, retención y costes de transferencia. Metadatos y objetos necesitan una estrategia coherente de recuperación. |
| Docker Compose para desarrollo local | Describe servicios y redes locales de manera repetible sin exigir un orquestador cloud. | Servicios instalados manualmente; Kubernetes local. | Hay diferencias con Fargate y volúmenes locales que persistirán entre ejecuciones. Versiones, health checks y limpieza de datos deberán documentarse en su entrega. |
| GitHub Actions para CI/CD | Mantiene comprobaciones y automatización junto al código y los pull requests. | GitLab CI; Jenkins; ejecución manual. | MOP-001 ejecuta CI con permisos de lectura. Publicación, credenciales y promoción de despliegues requieren una entrega posterior; los tiempos y límites del runner afectan al pipeline. |
| Terraform y AWS ECS Fargate para infraestructura y despliegue | Terraform permite revisar infraestructura declarativa; Fargate permite ejecutar contenedores sin gestionar hosts. | CloudFormation/CDK; máquinas virtuales; ECS sobre EC2; Kubernetes/EKS. | Introduce dependencia de AWS, gestión del estado de Terraform, IAM y coste continuo. Recursos y límites para entrenamiento deben comprobarse antes de asignarlo a Fargate. |
| Prometheus y Grafana para observabilidad | Separan recolección de métricas y visualización y permiten definir alertas operativas. | CloudWatch exclusivamente; otra plataforma gestionada. | Hay que limitar cardinalidad, definir retención y mantener dashboards y alertas. No cubren automáticamente logs, trazas ni calidad del modelo. |
| Evidently para detección de drift | Proporciona herramientas para comparar distribuciones y evaluar cambios en datos y predicciones. | Tests estadísticos propios; alternativas de monitorización ML. | Requiere referencia, ventanas y umbrales adecuados. El drift puede producir falsos positivos y no demuestra por sí solo degradación ni justifica reentrenar. |
| Locust para pruebas de carga | Permite describir escenarios HTTP y comportamiento de usuarios en Python. | k6; JMeter; scripts ad hoc. | Las pruebas necesitan una carga representativa y un generador capaz de producirla. Los resultados no sustituyen SLO ni predicen por sí solos el tráfico real. |

## Propuestas revisables

Estas propuestas **no están aprobadas como requisitos ni implementadas** en
MOP-001. Se validarán con datos, requisitos operativos y nuevos ADR cuando proceda:

- Organizar futuros módulos de dominio y contratos compartidos, entrenamiento,
  serving e integraciones dentro de `modelops`. Los nombres y límites se decidirán
  cuando existan responsabilidades reales; hoy solo se crea el paquete raíz.
- Compartir contratos y transformaciones deterministas entre entrenamiento e
  inferencia para reducir diferencias de comportamiento. Revisar estos límites
  con el primer pipeline y sus pruebas, evitando imports del proceso de
  entrenamiento desde el servicio de inferencia.
- Ejecutar inferencia como servicio Fargate y valorar entrenamiento como tarea
  independiente. La duración, CPU, memoria y posibles necesidades de GPU pueden
  exigir otro servicio para entrenamiento; no se define aún su plataforma final.
- Usar Compose para MLflow y PostgreSQL, y evaluar un almacén compatible con S3
  en local. La topología y las imágenes concretas quedan abiertas.
- Definir un flujo de promoción explícito del registro a serving, con revisión,
  métricas de aceptación, trazabilidad y rollback. No se fija todavía el método
  de promoción, la estrategia de rollout ni el orquestador.
- Preferir credenciales temporales y roles de ejecución en AWS. Región, redes,
  IAM, backend del estado de Terraform y mecanismo de autenticación de CI/CD
  requieren diseño posterior; aquí no se generan credenciales ni recursos.
- Establecer SLO de latencia, disponibilidad y calidad, y umbrales de drift con
  datos representativos. Frecuencia de evaluación, ventanas, alertas, retención,
  dashboards y eventual política de reentrenamiento permanecen pendientes.
- Fijar versiones y añadir dependencias del stack futuro solamente cuando una
  entrega las utilice y verifique su compatibilidad con Python 3.12.

## Consecuencias del bootstrap

- El layout `src/modelops/` exige instalar la distribución para importarla. Los
  tests evitan inyectar rutas y comprueban la importación fuera del repositorio.
- `pyproject.toml` define metadatos, backend de construcción y configuración de
  Ruff y pytest. `uv.lock` fija versiones y hashes de las dependencias, y CI usa
  `--locked` para rechazar divergencias con los metadatos del proyecto.
- Solo Ruff y pytest se declaran en el grupo de desarrollo. `uv_build` se usa
  como backend de empaquetado aislado; no hay dependencias de ejecución.
- CI instala el paquete sin modo editable y verifica lint, formato, tests y
  generación de distribuciones. Las comprobaciones locales usan las mismas
  herramientas. Este pipeline no despliega ni publica artefactos.
- Separar las decisiones futuras de su implementación mantiene la entrega
  pequeña y evita arrastrar dependencias o servicios que todavía no se utilizan.

## Riesgos pendientes y criterios de revisión

1. **Dominio y modelo sin validar:** confirmar tipo de datos, baseline, métricas,
   licencias y requisitos de privacidad antes de implementar el primer modelo.
2. **Acoplamiento y diferencias entre entrenamiento y serving:** revisar contratos
   y límites de módulos al introducir el primer proceso real. Extraer servicios
   solo si existen necesidades de escala, autonomía o aislamiento demostradas.
3. **Coste y capacidad cloud:** contrastar Fargate con duración y recursos reales,
   y estimar costes de ejecución, metadatos, artefactos y transferencia antes del
   primer despliegue. Revisar el destino si no cumple los requisitos.
4. **Cadena de suministro y compatibilidad:** revisar cambios del lockfile y
   actualizar deliberadamente Python, uv y bibliotecas. Los hashes ayudan a
   identificar artefactos; no sustituyen evaluar dependencias y sus licencias.
5. **Disponibilidad y recuperación:** definir recuperación de PostgreSQL y S3,
   autenticación de MLflow, acceso a la API y rollback antes de producción.
6. **Señales de drift ambiguas:** validar referencias, tamaños de muestra y
   umbrales antes de activar alertas. Cualquier reentrenamiento automático
   necesitará una decisión y validación propias.

La arquitectura se revisará cuando se conozcan el caso de uso y sus SLO, al
integrar entrenamiento y serving, y antes del primer despliegue cloud. Las
revisiones documentarán evidencia, impacto y qué decisión queda sustituida.

## Referencias del bootstrap

- [Proyectos con uv](https://docs.astral.sh/uv/guides/projects/).
- [Lockfile y sincronización](https://docs.astral.sh/uv/concepts/projects/sync/).
- [uv en GitHub Actions](https://docs.astral.sh/uv/guides/integration/github/).
- [Configuración de Ruff](https://docs.astral.sh/ruff/configuration/).
- [Integración de pytest y layout src](https://pytest.org/en/stable/explanation/goodpractices.html).
