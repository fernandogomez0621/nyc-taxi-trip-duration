"""
Feature engineering: expresiones nativas de Spark.

Deliberadamente NO se usan UDFs de Python. Todo el cálculo geográfico se expresa
con funciones nativas (`F.radians`, `F.sin`, `F.atan2`), que se ejecutan dentro
de la JVM sin el costo de serializar cada fila hacia un intérprete de Python. Un
UDF aquí sería más legible pero varias veces más lento sobre 1,4 millones de
filas, y es el antipatrón más común en este tipo de pipeline.
"""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F

from nyc_taxi.features.definitions import RADIO_TIERRA_KM


def _haversine(lat1: Column, lon1: Column, lat2: Column, lon2: Column) -> Column:
    phi1, phi2 = F.radians(lat1), F.radians(lat2)
    dphi = F.radians(lat2 - lat1)
    dlambda = F.radians(lon2 - lon1)

    a = F.pow(F.sin(dphi / 2), 2) + F.cos(phi1) * F.cos(phi2) * F.pow(F.sin(dlambda / 2), 2)
    return F.lit(2 * RADIO_TIERRA_KM) * F.asin(F.sqrt(a))


def col_haversine_km(lat1: str, lon1: str, lat2: str, lon2: str) -> Column:
    """Distancia en línea recta entre recogida y destino."""
    return _haversine(F.col(lat1), F.col(lon1), F.col(lat2), F.col(lon2))


def col_manhattan_km(lat1: str, lon1: str, lat2: str, lon2: str) -> Column:
    """Distancia en cuadrícula: componente norte-sur más componente este-oeste."""
    d_lat = _haversine(F.col(lat1), F.col(lon1), F.col(lat2), F.col(lon1))
    d_lon = _haversine(F.col(lat1), F.col(lon1), F.col(lat1), F.col(lon2))
    return d_lat + d_lon


def col_bearing_deg(lat1: str, lon1: str, lat2: str, lon2: str) -> Column:
    """Rumbo inicial del trayecto, en grados desde el norte."""
    phi1, phi2 = F.radians(F.col(lat1)), F.radians(F.col(lat2))
    dlambda = F.radians(F.col(lon2) - F.col(lon1))

    y = F.sin(dlambda) * F.cos(phi2)
    x = F.cos(phi1) * F.sin(phi2) - F.sin(phi1) * F.cos(phi2) * F.cos(dlambda)
    return F.pmod(F.degrees(F.atan2(y, x)) + F.lit(360), F.lit(360.0))


def agregar_features_temporales(df, col_pickup: str = "pickup_datetime"):
    """
    Deriva hora, día de la semana, mes y bandera de fin de semana.

    Todas se conocen en el instante de la recogida, así que son legítimas como
    variables de entrada del modelo.
    """
    return (
        df.withColumn("pickup_hour", F.hour(col_pickup))
        .withColumn("pickup_dayofweek", F.dayofweek(col_pickup))
        .withColumn("pickup_month", F.month(col_pickup))
        .withColumn("is_weekend", F.when(F.dayofweek(col_pickup).isin(1, 7), 1).otherwise(0))
    )


def agregar_features_geograficas(df):
    """Deriva las tres medidas de trayecto a partir de las cuatro coordenadas."""
    args = ("pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude")
    return (
        df.withColumn("distancia_haversine_km", col_haversine_km(*args))
        .withColumn("distancia_manhattan_km", col_manhattan_km(*args))
        .withColumn("bearing_deg", col_bearing_deg(*args))
    )


def col_es_aeropuerto(aeropuertos: dict, radio_km: float,
                      km_lat: float = 111.0, km_lon: float = 84.0) -> Column:
    """
    Bandera de trayecto de aeropuerto como expresion de Spark.

    Construye la disyuncion de las comprobaciones de proximidad para cada
    terminal y cada extremo del trayecto, sin recurrir a un UDF.
    """
    condicion = None
    for ref_lat, ref_lon in aeropuertos.values():
        for lat, lon in [("pickup_latitude", "pickup_longitude"),
                         ("dropoff_latitude", "dropoff_longitude")]:
            dx = (F.col(lat) - F.lit(ref_lat)) * F.lit(km_lat)
            dy = (F.col(lon) - F.lit(ref_lon)) * F.lit(km_lon)
            cerca = (dx * dx + dy * dy) <= F.lit(radio_km * radio_km)
            condicion = cerca if condicion is None else (condicion | cerca)
    return F.when(condicion, F.lit(1)).otherwise(F.lit(0))
