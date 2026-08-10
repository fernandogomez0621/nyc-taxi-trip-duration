"""
Configuración central del proyecto.

Todos los supuestos del pipeline viven aquí y en ningún otro sitio: rutas,
umbrales de limpieza, límites geográficos y la lista blanca de features. Así
cada decisión es citable en la nota técnica y cambiar un umbral no obliga a
buscar en cinco notebooks.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Unity Catalog
#
# El catalogo se lee de la variable de entorno NYC_TAXI_CATALOG y solo si no
# esta definida se usa el valor por defecto. Eso permite mover el proyecto entre
# espacios de trabajo sin tocar el codigo.
#
# El valor por defecto es `nyc_taxi`, un catalogo dedicado creado por
# `notebooks/00_setup`. En espacios de trabajo donde no se pueden crear
# catalogos —Databricks Free Edition, por ejemplo— basta exportar
# NYC_TAXI_CATALOG=workspace antes de ejecutar el pipeline.
# ---------------------------------------------------------------------------
CATALOGO = os.environ.get("NYC_TAXI_CATALOG", "nyc_taxi")
SCHEMA_BRONZE = "bronze"
SCHEMA_SILVER = "silver"
SCHEMA_GOLD = "gold"

VOLUMEN_LANDING = f"/Volumes/{CATALOGO}/{SCHEMA_BRONZE}/landing"

TBL_RAW_TRIPS = f"{CATALOGO}.{SCHEMA_BRONZE}.raw_trips"
TBL_RAW_SCORING = f"{CATALOGO}.{SCHEMA_BRONZE}.raw_scoring"
TBL_CLEAN = f"{CATALOGO}.{SCHEMA_SILVER}.clean_trips"
TBL_QUARANTINE = f"{CATALOGO}.{SCHEMA_SILVER}.quarantine_trips"
TBL_DQ_METRICS = f"{CATALOGO}.{SCHEMA_SILVER}.dq_metrics"
TBL_ANALYTICS = f"{CATALOGO}.{SCHEMA_GOLD}.trips_analytics"
TBL_AGG_DEMANDA = f"{CATALOGO}.{SCHEMA_GOLD}.agg_demanda_zona_hora"
TBL_FEATURES = f"{CATALOGO}.{SCHEMA_GOLD}.trips_features"

EXPERIMENTO_MLFLOW = "/Shared/nyc_taxi_trip_duration"

# ---------------------------------------------------------------------------
# Conteos publicados por la fuente (Kaggle). Sirven de verificación de carga.
# ---------------------------------------------------------------------------
FILAS_TRAIN = 1_458_644
FILAS_SCORING = 625_134

# ---------------------------------------------------------------------------
# Umbrales de limpieza (capa Silver)
#
# Cada uno es un supuesto explícito que hay que poder defender:
#   - Duración: un viaje de 0 segundos no ocurrió; uno de más de 6 horas en el
#     área de NYC es un medidor que quedó abierto, no un trayecto real.
#   - Pasajeros: 0 indica un registro sin ocupante; por encima de 6 excede la
#     capacidad legal de un taxi amarillo (incluidas las minivans).
#   - Coordenadas: caja que cubre los cinco condados más JFK, LGA y Newark.
# ---------------------------------------------------------------------------
DURACION_MIN_SEG = 1
DURACION_MAX_SEG = 6 * 3600  # 21600

PASAJEROS_MIN = 1
PASAJEROS_MAX = 6

NYC_LAT_MIN, NYC_LAT_MAX = 40.50, 41.00
NYC_LON_MIN, NYC_LON_MAX = -74.30, -73.70

# ---------------------------------------------------------------------------
# Split temporal
#
# Se separa por fecha y no de forma aleatoria porque el uso real del modelo es
# estimar viajes futuros. Un split aleatorio deja viajes del mismo día y la
# misma hora en ambos lados, lo que infla las métricas.
# El archivo cubre del 2016-01-01 al 2016-06-30; se reservan las dos últimas
# semanas para evaluación.
# ---------------------------------------------------------------------------
FECHA_CORTE_SPLIT = "2016-06-17"

# ---------------------------------------------------------------------------
# Lista blanca de features
#
# La regla que la gobierna: una variable solo puede entrar al modelo si es
# calculable con lo que trae test.csv, es decir, con información disponible en
# el instante de la recogida.
#
# Por eso quedan FUERA:
#   - dropoff_datetime : la fuente la retira de test.csv; permite derivar el
#                        target por simple resta.
#   - trip_duration    : es el target.
#   - velocidad_kmh    : se calcula como distancia / duración, o sea que deriva
#                        del target. No aparece en ningún archivo, así que no la
#                        atrapa una comparación de columnas: hay que excluirla a
#                        mano. Es legítima para el EDA, nunca como input.
# ---------------------------------------------------------------------------
FEATURES_MODELO = [
    "vendor_id",
    "passenger_count",
    "store_and_fwd_flag",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
    "distancia_haversine_km",
    "distancia_manhattan_km",
    "bearing_deg",
    "pickup_cluster",
    "dropoff_cluster",
    "es_aeropuerto",
]

COLUMNAS_PROHIBIDAS = [
    "dropoff_datetime",
    "trip_duration",
    "log_trip_duration",
    "velocidad_kmh",
]

TARGET = "trip_duration"
TARGET_LOG = "log_trip_duration"

# Número de zonas para el KMeans sobre coordenadas. Se ajusta SOLO sobre la
# partición de entrenamiento; ajustarlo sobre el dataset completo metería
# información del conjunto de evaluación dentro de las features.
# Aeropuertos. El analisis exploratorio (INSIGHT-05) mostro que los trayectos
# que tocan una terminal constituyen un regimen distinto: duracion mediana de
# 32 minutos frente a 10,4 del resto, y circulacion por autopista en lugar de
# retícula urbana. El radio de 2 km cubre terminales y areas de espera.
AEROPUERTOS = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
}
RADIO_AEROPUERTO_KM = 2.0

# Escala local de grados en la latitud de Nueva York. Evita recalcular la
# distancia de circulo maximo para una simple comprobacion de proximidad.
KM_POR_GRADO_LAT = 111.0
KM_POR_GRADO_LON_NYC = 84.0

N_CLUSTERS_ZONA = 20
SEMILLA = 42
