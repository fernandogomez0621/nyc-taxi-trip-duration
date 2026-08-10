# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 02 · Silver — Tipado, validacion y cuarentena
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Objetivo** | Convertir el dato crudo en dato confiable, aislando lo que no cumple las reglas |
# MAGIC | **Entradas** | `bronze.raw_trips` |
# MAGIC | **Salidas** | `silver.clean_trips`, `silver.quarantine_trips`, `silver.dq_metrics` |
# MAGIC | **Depende de** | Notebook 01 ejecutado con `p_origen=train` |
# MAGIC
# MAGIC **Proceso**
# MAGIC 1. Tipar las columnas y verificar que ningun cast falle en silencio
# MAGIC 2. Deduplicar por hash de fila
# MAGIC 3. Evaluar cinco reglas de calidad, cada una como bandera independiente
# MAGIC 4. Registrar las metricas de calidad por regla
# MAGIC 5. Separar entre tabla limpia y cuarentena
# MAGIC
# MAGIC **Principio de la capa:** no se borra el dato invalido, se aisla. Borrar
# MAGIC oculta el problema; derivar a cuarentena lo hace contable y trazable.
# MAGIC
# MAGIC La logica reside en `src/nyc_taxi/data_prep/`; este notebook orquesta y
# MAGIC documenta, de modo que las conversiones y las reglas queden cubiertas por
# MAGIC pruebas automaticas.

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

from nyc_taxi import config
from nyc_taxi.data_prep import typing as tp
from nyc_taxi.data_prep import validation

# COMMAND ----------

df_bronze = spark.table(config.TBL_RAW_TRIPS)
n_inicial = df_bronze.count()
print(f"Filas en Bronze: {n_inicial:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Tipado
# MAGIC
# MAGIC Ver `data_prep/typing.py` para el detalle de cada conversión. Las dos
# MAGIC decisiones que no son obvias: las coordenadas van a `double` y no a
# MAGIC `float`, porque 32 bits no alcanzan para la precisión del dato original y
# MAGIC el redondeo se traduce en decenas de metros; y los identificadores se
# MAGIC quedan en texto porque no son cantidades.
# MAGIC
# MAGIC **Un cast fallido en Spark no lanza excepción: produce nulos en silencio.**
# MAGIC Por eso se compara el conteo de nulos antes y después de convertir.

# COMMAND ----------

cols_negocio = [c for c in df_bronze.columns if not c.startswith("_")]

nulos_antes = df_bronze.select(
    [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols_negocio]
).first().asDict()

df = tp.tipar_trips(df_bronze)

nulos_despues = df.select(
    [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols_negocio]
).first().asDict()

print(f"{'columna':<22} {'tipo':<12} {'nulos antes':>12} {'nulos despues':>14}")
for c, t in df.dtypes:
    if c in cols_negocio:
        print(f"{c:<22} {t:<12} {nulos_antes[c]:>12,} {nulos_despues[c]:>14,}")

nuevos_nulos = {c: nulos_despues[c] - nulos_antes[c] for c in cols_negocio
                if nulos_despues[c] > nulos_antes[c]}
assert not nuevos_nulos, f"El casteo introdujo nulos: {nuevos_nulos}"
print("\nNingun cast fallo silenciosamente.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Deduplicación
# MAGIC
# MAGIC Se usa el `_row_hash` calculado en Bronze, que cubre todas las columnas de
# MAGIC negocio. Bronze ya reportó cero duplicados exactos y cero ids repetidos,
# MAGIC así que esta regla no va a eliminar nada: queda operando de forma
# MAGIC preventiva, para el caso de una reingesta o de un archivo reenviado.

# COMMAND ----------

df = df.dropDuplicates(["_row_hash"])
n_tras_dedup = df.count()
print(f"Filas tras deduplicar: {n_tras_dedup:,}  (eliminadas: {n_inicial - n_tras_dedup:,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validaciones
# MAGIC
# MAGIC Cada regla se materializa como una columna booleana propia en vez de
# MAGIC encadenar filtros. Eso permite contar cuántas filas falla **cada regla por
# MAGIC separado**, en lugar de saber solo que algo falló, y es lo que hace
# MAGIC posible detectar un umbral mal calibrado: si una sola regla rechaza un
# MAGIC porcentaje alto, el sospechoso es el umbral y no el dato.
# MAGIC
# MAGIC Los umbrales y su justificación están en `config.py`.

# COMMAND ----------

reglas = validation.construir_reglas_spark()

for nombre, expr in reglas.items():
    df = df.withColumn(nombre, expr)

df = df.withColumn("es_valido", F.expr(" AND ".join(validation.NOMBRES_REGLAS)))

# El cómputo serverless no soporta cache(). Sin persistir de alguna forma, el
# DataFrame se recomputaría en cada acción: una vez para el resumen de calidad y
# otra por cada tabla escrita. Materializar en una tabla Delta intermedia logra
# el mismo objetivo: se escribe una vez y las lecturas posteriores no vuelven a
# evaluar las expresiones de validación. El guion bajo marca que es intermedia.
TBL_VALIDADO = f"{config.CATALOGO}.{config.SCHEMA_SILVER}._trips_validado"

(df.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true").saveAsTable(TBL_VALIDADO))

df = spark.table(TBL_VALIDADO)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Métricas de calidad
# MAGIC
# MAGIC Una sola pasada agregada en vez de un `count()` por regla: cinco conteos
# MAGIC separados serían cinco escaneos completos de la tabla.

# COMMAND ----------

agregados = [
    F.sum((~F.col(r)).cast("int")).alias(r) for r in validation.NOMBRES_REGLAS
] + [
    F.sum((~F.col("es_valido")).cast("int")).alias("total_invalidas"),
    F.count(F.lit(1)).alias("total_filas"),
]

resumen = df.select(agregados).first().asDict()

total = resumen["total_filas"]
print(f"{'regla':<20} {'fallos':>10} {'% del total':>12}")
for r in validation.NOMBRES_REGLAS:
    print(f"{r:<20} {resumen[r]:>10,} {resumen[r] / total * 100:>11.3f}%")
print("-" * 44)
print(f"{'total invalidas':<20} {resumen['total_invalidas']:>10,} "
      f"{resumen['total_invalidas'] / total * 100:>11.3f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC Los conteos se persisten en `dq_metrics` con marca de tiempo. Esa tabla es
# MAGIC la evidencia citable del resumen ejecutivo y, acumulada entre corridas, el
# MAGIC insumo natural del monitoreo de calidad en producción.

# COMMAND ----------

filas_dq = [
    (regla, int(resumen[regla]), int(total), float(resumen[regla] / total * 100))
    for regla in validation.NOMBRES_REGLAS
]

df_dq = (
    spark.createDataFrame(filas_dq, ["regla", "filas_fallidas", "filas_evaluadas", "pct_fallo"])
    .withColumn("tabla_origen", F.lit(config.TBL_RAW_TRIPS))
    .withColumn("evaluado_en", F.current_timestamp())
)

df_dq.write.format("delta").mode("append").saveAsTable(config.TBL_DQ_METRICS)
display(df_dq)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Separación
# MAGIC
# MAGIC Las banderas por regla se conservan en cuarentena —son el motivo del
# MAGIC rechazo— y se descartan de la tabla limpia, donde ya no aportan nada
# MAGIC porque todas valen verdadero por construcción.

# COMMAND ----------

cols_finales = [c for c in df.columns if c not in validation.NOMBRES_REGLAS + ["es_valido"]]

df_clean = df.filter("es_valido").select(*cols_finales)
df_quarantine = df.filter("NOT es_valido")

(df_clean.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true").saveAsTable(config.TBL_CLEAN))

(df_quarantine.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true").saveAsTable(config.TBL_QUARANTINE))

n_clean = spark.table(config.TBL_CLEAN).count()
n_quar = spark.table(config.TBL_QUARANTINE).count()

print(f"clean_trips      : {n_clean:,}")
print(f"quarantine_trips : {n_quar:,}")
print(f"retencion        : {n_clean / n_inicial * 100:.2f}%")

assert n_clean + n_quar == n_tras_dedup, "Se perdieron filas en la separacion"

# La tabla intermedia ya cumplió su función.
spark.sql(f"DROP TABLE IF EXISTS {TBL_VALIDADO}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspección de lo rechazado
# MAGIC
# MAGIC Mirar la cuarentena no es opcional: es lo que distingue un dato realmente
# MAGIC inválido de un umbral mal puesto. Si un grupo grande de filas cae por una
# MAGIC sola regla y a simple vista parecen viajes legítimos, hay que revisar el
# MAGIC criterio antes que el dato.
# MAGIC
# MAGIC La primera tabla muestra las combinaciones de reglas que fallan juntas,
# MAGIC que suele revelar el patrón: coordenadas en (0,0) con duración absurda
# MAGIC apunta a un GPS sin señal, no a dos problemas independientes.

# COMMAND ----------

display(
    spark.table(config.TBL_QUARANTINE)
    .groupBy(*validation.NOMBRES_REGLAS)
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

display(
    spark.table(config.TBL_QUARANTINE)
    .select("id", "pickup_datetime", "trip_duration", "passenger_count",
            "pickup_latitude", "pickup_longitude", *validation.NOMBRES_REGLAS)
    .limit(20)
)

# COMMAND ----------

