# NYC Taxi Trip Duration — Prueba técnica

Predicción de la duración de viajes en taxi en Nueva York a partir del dataset
de la competencia de Kaggle (1.458.644 viajes, enero a junio de 2016).

Implementado sobre Databricks Free Edition con Unity Catalog, arquitectura
medallion sobre Delta Lake y seguimiento de experimentos en MLflow.

## Mapeo con los entregables solicitados

| Entregable | Ubicación |
|---|---|
| Notebook con el pipeline de datos (Parte 1) | `notebooks/pipeline/` — `01_bronze`, `02_silver`, `03_gold` |
| Notebook con EDA y modelo (Partes 2 y 3) | `notebooks/analytics/` — `04_eda`, `05_modelo` |
| Resumen ejecutivo (máx. 1 página) | `docs/resumen_ejecutivo.pdf` |
| Nota de decisiones técnicas y supuestos | `docs/decisiones_tecnicas.md` |
| Bonus · monitoreo de deriva | `notebooks/analytics/06_monitoreo` |
| Bonus · job parametrizado | `resources/nyc_taxi_pipeline_job.yml` y `docs/orquestacion.md` |

El pipeline se separó en tres notebooks en vez de uno porque cada capa es una
tarea independiente del job de orquestación: permite reintentar o reprocesar
Silver sin volver a leer el CSV, y hace explícitas las dependencias entre capas.

## Estructura

```
notebooks/     Notebooks de Databricks (orquestación y narrativa)
               00_setup crea la estructura, 00_tests corre la suite
src/nyc_taxi/  Lógica pura, importable y testeable
tests/unit/    85 pruebas con pytest, sin dependencia de Spark
resources/     Definición del job de orquestación
docs/          Entregables de comunicación y notas de diseño
evidencias/    Figuras, capturas y artefactos de la ejecución real
```

El criterio de separación: la lógica pura sube a `src/`, la orquestación de
Spark se queda en el notebook. Así el notebook se lee como un documento y el
paquete guarda lo que se puede probar automáticamente.

## Ejecución

1. Ejecutar `notebooks/00_setup` para crear catálogo, esquemas y volúmenes
2. Subir `train.csv` y `test.csv` a `/Volumes/nyc_taxi/bronze/landing/`
3. Ejecutar `notebooks/00_tests` para verificar la instalación
4. Ejecutar `01_bronze` con el parámetro `p_origen=train`, luego `p_origen=scoring`
5. Ejecutar `02_silver` y `03_gold`
6. Ejecutar `04_eda`, `05_modelo` y `06_monitoreo`

El catálogo destino se controla con la variable de entorno `NYC_TAXI_CATALOG`
(por defecto `nyc_taxi`). Ver `docs/migracion_azure.md` para ejecutar en un
espacio de trabajo distinto.

## Tests

En local, desde la raíz del repositorio:

```bash
pip install pytest
pytest
```

Dentro de Databricks: ejecutar `notebooks/00_tests`. Localiza la raíz del repo de
forma dinámica, así que funciona sin importar dónde cuelgue el proyecto en el
workspace.

## Resultados

| Indicador | Valor |
|---|---|
| Error absoluto medio | 3,27 min sobre un trayecto mediano de 11,3 |
| Mejora sobre la mejor heurística manual | 24,4 % en RMSLE |
| Brecha entre entrenamiento y evaluación | 6,0 % en RMSLE |
| Retención tras validación de calidad | 99,79 % |

La carpeta `evidencias/` contiene las figuras, las capturas de la ejecución y los
artefactos generados.

## Decisiones de diseño destacadas

- **Sin fugas de datos.** `dropoff_datetime` y la velocidad promedio se excluyen
  del modelo: la primera porque la fuente la retira de `test.csv`, la segunda
  porque se deriva del target. Gold se bifurca en una tabla de análisis y una de
  features para que la arquitectura garantice la separación.
- **Split temporal.** Las dos últimas semanas se reservan para evaluación, ya que
  el uso real del modelo es estimar viajes futuros.
- **Sin UDFs de Python** donde existe función nativa: todo el cálculo geográfico
  se expresa con funciones de Spark.
