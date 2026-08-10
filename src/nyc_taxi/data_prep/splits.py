"""
Asignación del split de entrenamiento y evaluación.

El split se materializa como una columna en la tabla de features en vez de
calcularse al vuelo en cada corrida. Dos razones: cualquier notebook que lea la
tabla obtiene exactamente la misma partición, y el reparto queda auditable en el
propio dato en lugar de esconderse en el código de entrenamiento.
"""

from __future__ import annotations

from datetime import date, datetime

from nyc_taxi import config

SPLIT_TRAIN = "train"
SPLIT_TEST = "test"


def _a_fecha(valor) -> date:
    if isinstance(valor, datetime):
        return valor.date()
    if isinstance(valor, date):
        return valor
    return datetime.strptime(str(valor)[:10], "%Y-%m-%d").date()


def asignar_split(fecha_pickup, fecha_corte: str = config.FECHA_CORTE_SPLIT) -> str:
    """
    Devuelve 'train' para lo anterior al corte y 'test' del corte en adelante.

    El corte es temporal y no aleatorio porque el modelo se usaría para estimar
    viajes futuros. Un reparto aleatorio dejaría trayectos del mismo día, la
    misma hora y la misma congestión a ambos lados, lo que hace que la métrica
    de evaluación mida memorización de condiciones concretas y no capacidad de
    generalizar.
    """
    return SPLIT_TEST if _a_fecha(fecha_pickup) >= _a_fecha(fecha_corte) else SPLIT_TRAIN


def col_split(col_pickup: str = "pickup_datetime", fecha_corte: str = config.FECHA_CORTE_SPLIT):
    """Versión en columna de Spark de `asignar_split`."""
    from pyspark.sql import functions as F

    return F.when(F.to_date(F.col(col_pickup)) >= F.lit(fecha_corte), F.lit(SPLIT_TEST)).otherwise(
        F.lit(SPLIT_TRAIN)
    )
