"""
Tipado de la capa Silver.

Bronze guarda todo como texto para no alterar el archivo original. Aquí se
convierte a los tipos reales, y cada elección está justificada porque varias son
contraintuitivas.

El mismo módulo tipa el set de entrenamiento y el lote de scoring: la función
solo convierte las columnas presentes, así que las dos columnas ausentes en
scoring (`dropoff_datetime` y `trip_duration`) simplemente se omiten. Reusar la
misma conversión en ambos es lo que evita el training/serving skew, que es la
causa más común de modelos que funcionan en el notebook y fallan en producción.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Mapa de tipos
#
# Decisiones que vale la pena poder defender:
#
#   - Coordenadas a DOUBLE y no FLOAT. Un float de 32 bits da ~7 dígitos
#     significativos, y -73.982154846191406 necesita más. El redondeo se
#     traduce en decenas de metros de error, que sobre viajes de kilómetro y
#     medio contamina directamente la feature de distancia.
#
#   - id y vendor_id se quedan en STRING. Son identificadores, no cantidades:
#     nadie va a promediar un vendor_id, y tratarlos como número invita a
#     operaciones que no tienen sentido.
#
#   - trip_duration a INT. Son segundos exactos; un decimal sugeriría una
#     precisión que la fuente no tiene.
# ---------------------------------------------------------------------------

TIPOS = {
    "id": "string",
    "vendor_id": "string",
    "pickup_datetime": "timestamp",
    "dropoff_datetime": "timestamp",
    "passenger_count": "int",
    "pickup_longitude": "double",
    "pickup_latitude": "double",
    "dropoff_longitude": "double",
    "dropoff_latitude": "double",
    "trip_duration": "int",
}

# store_and_fwd_flag se trata aparte: no es un cast sino un mapeo de dominio.
FLAG_ORIGEN = "store_and_fwd_flag"
FLAG_MAPA = {"Y": 1, "N": 0}


def columnas_a_tipar(columnas_presentes: list[str]) -> dict[str, str]:
    """
    Filtra el mapa de tipos a las columnas que realmente existen en el DataFrame.

    Es lo que permite usar la misma función para el set de entrenamiento (11
    columnas de negocio) y el de scoring (9).
    """
    return {c: t for c, t in TIPOS.items() if c in columnas_presentes}


def mapear_flag(valor: str | None) -> int | None:
    """
    Convierte la bandera Y/N a 1/0.

    Devuelve None ante un valor fuera del dominio en vez de asumir 0: un valor
    inesperado es un problema de calidad que debe hacerse visible, no un cero
    silencioso.
    """
    if valor is None:
        return None
    return FLAG_MAPA.get(valor.strip().upper())


def tipar_trips(df):
    """
    Aplica el tipado a un DataFrame de Bronze.

    Las columnas técnicas de auditoría (prefijo `_`) se conservan intactas.
    """
    from pyspark.sql import functions as F

    for columna, tipo in columnas_a_tipar(df.columns).items():
        df = df.withColumn(columna, F.col(columna).cast(tipo))

    if FLAG_ORIGEN in df.columns:
        df = df.withColumn(
            FLAG_ORIGEN,
            F.when(F.upper(F.trim(F.col(FLAG_ORIGEN))) == "Y", 1)
            .when(F.upper(F.trim(F.col(FLAG_ORIGEN))) == "N", 0)
            .otherwise(None)
            .cast("int"),
        )

    return df


def verificar_tipado(df) -> dict[str, str]:
    """
    Devuelve {columna: tipo} tras el casteo, para dejar constancia en el notebook.

    Un cast fallido en Spark no lanza excepción: produce nulos. Comparar el
    conteo de nulos antes y después es la única forma de detectarlo.
    """
    return dict(df.dtypes)
