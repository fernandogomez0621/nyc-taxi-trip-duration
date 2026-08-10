"""
Reglas de calidad de la capa Silver.

Cada regla existe en dos formas equivalentes: una función pura (testeable, y que
documenta el criterio) y una expresión de Spark que se aplica sobre el
DataFrame. La lista `REGLAS` es la fuente única de verdad sobre qué se valida.

Principio de la capa: no se borra el dato inválido, se aísla. Borrar oculta el
problema; derivar a cuarentena lo hace contable y trazable.
"""

from __future__ import annotations

from nyc_taxi import config


# ---------------------------------------------------------------------------
# Reglas como funciones puras
# ---------------------------------------------------------------------------

def es_duracion_valida(duracion_seg: float) -> bool:
    """Un viaje debe durar al menos un segundo y menos de seis horas."""
    return config.DURACION_MIN_SEG <= duracion_seg <= config.DURACION_MAX_SEG


def es_pasajeros_valido(n: int) -> bool:
    """Entre 1 y 6 pasajeros: 0 es un registro sin ocupante, más de 6 excede la capacidad."""
    return config.PASAJEROS_MIN <= n <= config.PASAJEROS_MAX


def esta_en_nyc(lat: float, lon: float) -> bool:
    """Coordenada dentro de la caja que cubre los cinco condados y los aeropuertos."""
    return (
        config.NYC_LAT_MIN <= lat <= config.NYC_LAT_MAX
        and config.NYC_LON_MIN <= lon <= config.NYC_LON_MAX
    )


def es_trayecto_en_nyc(lat_o: float, lon_o: float, lat_d: float, lon_d: float) -> bool:
    """Ambos extremos del trayecto deben caer dentro del área."""
    return esta_en_nyc(lat_o, lon_o) and esta_en_nyc(lat_d, lon_d)


# ---------------------------------------------------------------------------
# Las mismas reglas como expresiones de Spark
#
# Se declaran como funciones que devuelven Column para no importar pyspark al
# nivel del módulo: así los tests unitarios corren sin sesión de Spark.
# ---------------------------------------------------------------------------

def construir_reglas_spark() -> dict:
    """
    Devuelve {nombre_bandera: expresión booleana de Spark}.

    El nombre de cada bandera se convierte en una columna del DataFrame, lo que
    permite contar cuántas filas falla cada regla por separado en vez de saber
    solo que algo falló.
    """
    from pyspark.sql import functions as F

    en_caja = lambda lat, lon: (  # noqa: E731
        (F.col(lat) >= config.NYC_LAT_MIN)
        & (F.col(lat) <= config.NYC_LAT_MAX)
        & (F.col(lon) >= config.NYC_LON_MIN)
        & (F.col(lon) <= config.NYC_LON_MAX)
    )

    return {
        "v_duracion": (
            (F.col("trip_duration") >= config.DURACION_MIN_SEG)
            & (F.col("trip_duration") <= config.DURACION_MAX_SEG)
        ),
        "v_pasajeros": (
            (F.col("passenger_count") >= config.PASAJEROS_MIN)
            & (F.col("passenger_count") <= config.PASAJEROS_MAX)
        ),
        "v_coords_pickup": en_caja("pickup_latitude", "pickup_longitude"),
        "v_coords_dropoff": en_caja("dropoff_latitude", "dropoff_longitude"),
        "v_completitud": (
            F.col("id").isNotNull()
            & F.col("pickup_datetime").isNotNull()
            & F.col("trip_duration").isNotNull()
        ),
    }


NOMBRES_REGLAS = [
    "v_duracion",
    "v_pasajeros",
    "v_coords_pickup",
    "v_coords_dropoff",
    "v_completitud",
]
