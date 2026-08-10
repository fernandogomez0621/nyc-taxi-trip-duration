# Evidencias de ejecución

Esta carpeta contiene la salida real de la ejecución del proyecto. No es código:
es la prueba de que lo que está en `notebooks/` y `src/` se ejecutó de extremo a
extremo sobre los 1.458.644 viajes del conjunto de datos.

## Figuras

Generadas por los notebooks 04, 05 y 06, y guardadas automáticamente en el
volumen `/Volumes/nyc_taxi/gold/artifacts/figuras/`.

| Archivo | Contenido | Notebook |
|---|---|---|
| `01_distribucion_target.png` | Distribución de la duración en escala original y logarítmica | 04 |
| `02_dispersion_por_distancia.png` | Dispersión condicional por tramo de distancia | 04 |
| `03_efectos_por_factor.png` | Efectos marginales frente a parciales | 04 |
| `04_patron_horario.png` | Demanda y velocidad por hora del día | 04 |
| `05_regimen_aeropuerto.png` | Contraste entre trayectos de aeropuerto y del resto | 04 |
| `06_serie_diaria.png` | Volumen diario de viajes y la tormenta del 23 de enero | 04 |
| `07_geografia_demanda.png` | Concentración por zona y pares origen-destino | 04 |
| `08_residuos_modelo.png` | Análisis de residuos en cuatro paneles | 05 |
| `09_importancia_variables.png` | Importancia por permutación | 05 |
| `10_sensibilidad_drift.png` | Sensibilidad del detector de deriva por escenario | 06 |

## Capturas

| Archivo | Qué documenta |
|---|---|
| `01_catalogo_creado.png` | Estructura de Unity Catalog tras ejecutar `00_setup` |
| `02_catalogo_tablas_finales.png` | Las nueve tablas de las tres capas al finalizar el pipeline |
| `03_job_grafo_tareas.png` | Grafo de dependencias del job de orquestación |
| `04_job_ejecucion_exitosa.png` | Ejecución completa del job en 3 min 31 s, seis tareas correctas |
| `05_mlflow_experimentos.png` | Corridas registradas: dos baselines, modelo lineal y dos configuraciones de GBT |
| `06_mlflow_modelo_metricas.png` | Métricas y parámetros del modelo seleccionado |
| `07_modelo_registrado_uc.png` | Modelo versionado en Unity Catalog como `nyc_taxi.gold.trip_duration_gbt` |
| `08_tests_unitarios.png` | Suite de pruebas ejecutada dentro del entorno |

## Artefactos

Salidas serializadas que los notebooks producen y consumen entre sí.

| Archivo | Contenido |
|---|---|
| `eda_efectos.json` | Tamaños de efecto por factor y orden previsto de importancia. Lo genera el notebook 04 y lo consume el 05 para contrastar su predicción con la importancia real del modelo |
| `modelo_resumen.json` | Hiperparámetros, métricas en ambos conjuntos, comparación de modelos, importancias y resultados de la búsqueda |
| `monitoring_drift.csv` | Historial de evaluaciones de deriva sobre el lote posterior |
| `silver_combinaciones_reglas.csv` | Combinaciones de reglas de calidad que fallan simultáneamente, base del análisis de tipos de error |

## Notebooks ejecutados

Exportados con sus salidas visibles, para consultarlos sin necesidad de
ejecutar el proyecto. El código fuente sin salidas vive en `notebooks/`.

## Resultados principales

| Indicador | Valor |
|---|---|
| Registros procesados | 1.458.644 |
| Retención tras validación | 99,79 % |
| Error absoluto medio del modelo | 3,27 min |
| Trayecto mediano | 11,3 min |
| Mejora sobre la mejor heurística manual | 24,4 % en RMSLE |
| Brecha entre entrenamiento y evaluación | 6,0 % en RMSLE |
| Concordancia entre el orden previsto y el obtenido | 0,730 |
| Duración de la ejecución completa del pipeline | 3 min 31 s |
