# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 01 · Bronze — Ingesta cruda
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Objetivo** | Materializar el archivo de origen como tabla Delta sin alterar ningun valor |
# MAGIC | **Entradas** | `train.csv` o `test.csv` en el volumen de aterrizaje |
# MAGIC | **Salidas** | `bronze.raw_trips` (entrenamiento) o `bronze.raw_scoring` (lote de scoring) |
# MAGIC | **Parametro** | `p_origen` = `train` \| `scoring` |
# MAGIC | **Depende de** | `00_setup` ejecutado y los archivos cargados en el volumen |
# MAGIC
# MAGIC **Proceso**
# MAGIC 1. Leer el CSV con esquema explicito de tipo texto
# MAGIC 2. Anexar columnas tecnicas de auditoria y hash de fila
# MAGIC 3. Escribir la tabla Delta
# MAGIC 4. Verificar el conteo contra el publicado por la fuente
# MAGIC
# MAGIC **Principio de la capa:** Bronze es un espejo fiel del archivo. Todo se lee
# MAGIC como texto para que ningun tipo se interprete prematuramente; lo unico que
# MAGIC se agrega son columnas tecnicas. El tipado y la limpieza corresponden a
# MAGIC Silver.
# MAGIC
# MAGIC El mismo notebook ingesta ambos conjuntos, que tienen esquemas distintos:
# MAGIC el de scoring no incluye `dropoff_datetime` ni `trip_duration`.

# COMMAND ----------

import os
import sys


# Localiza la raiz del repo subiendo hasta encontrar src/, en vez de fijar un
# numero de saltos. Asi el notebook funciona a cualquier profundidad.
def _preparar_path(marcador="src", max_niveles=10):
    ruta = os.getcwd()
    for _ in range(max_niveles):
        if os.path.isdir(os.path.join(ruta, marcador)):
            destino = os.path.join(ruta, marcador)
            if destino not in sys.path:
                sys.path.insert(0, destino)
            return destino
        padre = os.path.dirname(ruta)
        if padre == ruta:
            break
        ruta = padre
    raise RuntimeError(f"No se encontro la raiz del repo (carpeta con {marcador}/)")


_preparar_path()

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from nyc_taxi import config

# COMMAND ----------

dbutils.widgets.dropdown("p_origen", "train", ["train", "scoring"], "Origen del archivo")

# COMMAND ----------

# El set de scoring es el de entrenamiento menos las dos columnas que la fuente
# retira deliberadamente: dropoff_datetime, porque permite derivar el target por
# simple resta, y trip_duration, que es el target. Esa ausencia es la evidencia
# documental de qué información NO está disponible al momento de predecir.
COLS_TRAIN = [
    "id", "vendor_id", "pickup_datetime", "dropoff_datetime", "passenger_count",
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "store_and_fwd_flag", "trip_duration",
]
COLS_SCORING = [c for c in COLS_TRAIN if c not in ("dropoff_datetime", "trip_duration")]

ORIGENES = {
    "train": {
        "archivo": "train.csv",
        "tabla": config.TBL_RAW_TRIPS,
        "columnas": COLS_TRAIN,
        "filas_esperadas": config.FILAS_TRAIN,
    },
    "scoring": {
        "archivo": "test.csv",
        "tabla": config.TBL_RAW_SCORING,
        "columnas": COLS_SCORING,
        "filas_esperadas": config.FILAS_SCORING,
    },
}

origen = dbutils.widgets.get("p_origen")
cfg = ORIGENES[origen]
ruta = f"{config.VOLUMEN_LANDING}/{cfg['archivo']}"

print(f"Origen : {origen}\nArchivo: {ruta}\nDestino: {cfg['tabla']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lectura
# MAGIC
# MAGIC Schema explícito de tipo texto en vez de `inferSchema`. Evita una pasada
# MAGIC completa extra sobre los 191 MB solo para adivinar tipos, y garantiza que
# MAGIC Bronze no altere ningún valor.
# MAGIC
# MAGIC `FAILFAST` hace que un archivo que no calce con el schema reviente aquí y
# MAGIC no tres pasos después: por defecto Spark pondría nulos y seguiría.

# COMMAND ----------

schema = StructType([StructField(c, StringType(), True) for c in cfg["columnas"]])

df_raw = (
    spark.read.format("csv")
    .option("header", "true")
    .option("mode", "FAILFAST")
    .schema(schema)
    .load(ruta)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Columnas de auditoría
# MAGIC
# MAGIC El hash SHA-256 de la fila detecta duplicados exactos entre cargas y da
# MAGIC idempotencia a los reprocesos. Se calcula solo sobre columnas de negocio
# MAGIC —nunca sobre las de auditoría— para que reingestar el mismo archivo
# MAGIC produzca el mismo hash.
# MAGIC
# MAGIC Los nulos se sustituyen por un centinela porque `concat_ws` los omite, y
# MAGIC sin eso dos filas distintas podrían colisionar en el mismo hash.

# COMMAND ----------

cols_hash = [F.coalesce(F.col(c), F.lit("<null>")) for c in cfg["columnas"]]

df_bronze = (
    df_raw
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.lit(cfg["archivo"]))
    .withColumn("_origen", F.lit(origen))
    .withColumn("_row_hash", F.sha2(F.concat_ws("||", *cols_hash), 256))
)

# COMMAND ----------

(
    df_bronze.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(cfg["tabla"])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verificación de carga
# MAGIC
# MAGIC Comparar contra el conteo publicado por la fuente detecta de inmediato una
# MAGIC subida truncada o un archivo mal delimitado.
# MAGIC
# MAGIC Los dos conteos de duplicados detectan cosas distintas: un `id` repetido
# MAGIC con hash distinto es un reenvío corregido; un hash repetido es duplicación
# MAGIC real del mismo registro.

# COMMAND ----------

df_check = spark.table(cfg["tabla"])
n_filas = df_check.count()
n_hash = df_check.select("_row_hash").distinct().count()
n_ids = df_check.select("id").distinct().count()

print(f"Filas cargadas   : {n_filas:,}")
print(f"Filas esperadas  : {cfg['filas_esperadas']:,}")
print(f"Hashes distintos : {n_hash:,}   (duplicados exactos: {n_filas - n_hash:,})")
print(f"IDs distintos    : {n_ids:,}   (ids repetidos: {n_filas - n_ids:,})")

assert n_filas == cfg["filas_esperadas"], (
    f"Conteo inesperado: {n_filas:,} vs {cfg['filas_esperadas']:,}. "
    "Revisar si la subida del archivo quedó incompleta."
)

# COMMAND ----------

display(df_check.limit(10))