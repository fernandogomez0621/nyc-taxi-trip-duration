# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 03 · Gold — Analitica, agregados y features
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Objetivo** | Preparar una tabla por consumidor, con el contrato que cada uno necesita |
# MAGIC | **Entradas** | `silver.clean_trips` |
# MAGIC | **Salidas** | `gold.trips_analytics`, `gold.agg_demanda_zona_hora`, `gold.trips_features` |
# MAGIC | **Depende de** | Notebook 02 |
# MAGIC
# MAGIC **Proceso**
# MAGIC 1. Derivar features temporales, geograficas y la bandera de aeropuerto
# MAGIC 2. Asignar el split temporal como columna materializada
# MAGIC 3. Ajustar el agrupamiento de zonas solo sobre la particion de entrenamiento
# MAGIC 4. Construir `trips_analytics` con el detalle completo, particionada por mes
# MAGIC 5. Construir `agg_demanda_zona_hora` con el grano zona x hora x dia
# MAGIC 6. Construir `trips_features` por lista blanca, particionada por split
# MAGIC 7. Verificar que ninguna columna prohibida sobrevivio
# MAGIC
# MAGIC **Decision de diseno.** Gold se divide segun quien consume cada tabla, y
# MAGIC esa separacion es la barrera contra fugas de datos:
# MAGIC
# MAGIC - `trips_analytics` conserva todo, incluida la velocidad y la hora de
# MAGIC   llegada. Alimenta el analisis exploratorio.
# MAGIC - `agg_demanda_zona_hora` precalcula los agregados que consumen las
# MAGIC   visualizaciones, evitando recorrer el detalle en cada consulta.
# MAGIC - `trips_features` contiene unicamente lo conocido en el instante de la
# MAGIC   recogida. Alimenta el modelo.
# MAGIC
# MAGIC La velocidad promedio se calcula como distancia sobre duracion, de modo que
# MAGIC deriva del target: es legitima para describir el fenomeno e invalida como
# MAGIC variable de entrada. Separar las tablas hace que la arquitectura garantice
# MAGIC esa distincion en lugar de depender de recordarla.

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

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F

from nyc_taxi import config
from nyc_taxi.data_prep import splits
from nyc_taxi.features import transformations as tf

# COMMAND ----------

df = spark.table(config.TBL_CLEAN)
n_silver = df.count()
print(f"Filas en Silver: {n_silver:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Features derivadas
# MAGIC
# MAGIC Todo el calculo geografico usa funciones nativas de Spark, sin UDFs de
# MAGIC Python. Un UDF resultaria mas legible pero obliga a serializar cada fila
# MAGIC hacia un interprete de Python y de vuelta, lo que sobre 1,45 millones de
# MAGIC registros cuesta varias veces mas. Las definiciones equivalentes en Python
# MAGIC puro viven en `features/definitions.py` y estan cubiertas por pruebas.
# MAGIC
# MAGIC Se calculan tres medidas de trayecto porque capturan cosas distintas: la
# MAGIC haversine mide la linea recta, la de cuadricula aproxima el recorrido real
# MAGIC sobre la retícula de calles de la ciudad, y el rumbo captura la direccion,
# MAGIC que no es simetrica: entrar a Manhattan y salir de ella tienen perfiles de
# MAGIC congestion distintos aunque la distancia sea identica.

# COMMAND ----------

df = tf.agregar_features_temporales(df)
df = tf.agregar_features_geograficas(df)
df = df.withColumn(
    "es_aeropuerto",
    tf.col_es_aeropuerto(
        config.AEROPUERTOS, config.RADIO_AEROPUERTO_KM,
        config.KM_POR_GRADO_LAT, config.KM_POR_GRADO_LON_NYC,
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Split temporal materializado
# MAGIC
# MAGIC El reparto se guarda como columna en lugar de recalcularse en cada corrida
# MAGIC de entrenamiento. Asi cualquier notebook que lea la tabla obtiene la misma
# MAGIC particion, y el criterio queda auditable en el propio dato.
# MAGIC
# MAGIC El corte es temporal y no aleatorio porque el uso real del modelo es
# MAGIC estimar viajes futuros. Un reparto aleatorio dejaria trayectos del mismo
# MAGIC dia, la misma hora y la misma congestion a ambos lados, de modo que la
# MAGIC metrica mediria memorizacion de condiciones concretas y no generalizacion.

# COMMAND ----------

df = df.withColumn("split_flag", splits.col_split())

reparto = df.groupBy("split_flag").count().orderBy("split_flag").collect()
for fila in reparto:
    print(f"{fila['split_flag']:<6} {fila['count']:>10,}  ({fila['count'] / n_silver * 100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Materializacion intermedia
# MAGIC
# MAGIC El computo serverless no admite `cache()`. Sin persistir de alguna forma,
# MAGIC el DataFrame se recomputaria completo en cada escritura posterior. Escribir
# MAGIC una tabla Delta intermedia cumple el mismo proposito: se evalua una vez y
# MAGIC las lecturas siguientes no vuelven a calcular las expresiones geograficas.

# COMMAND ----------

TBL_ENRIQUECIDO = f"{config.CATALOGO}.{config.SCHEMA_GOLD}._trips_enriquecido"

(df.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true").saveAsTable(TBL_ENRIQUECIDO))

df = spark.table(TBL_ENRIQUECIDO)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Agrupamiento de zonas
# MAGIC
# MAGIC Las cuatro coordenadas crudas son poco informativas para un modelo de
# MAGIC arboles, que tendria que aprender los limites de cada barrio a base de
# MAGIC cortes sucesivos. Agruparlas en zonas convierte esa geometria en dos
# MAGIC variables categoricas que el modelo aprovecha directamente.
# MAGIC
# MAGIC **El modelo se ajusta unicamente sobre la particion de entrenamiento.**
# MAGIC Ajustarlo sobre el conjunto completo definiria los centroides usando
# MAGIC tambien los viajes de evaluacion, lo que introduce informacion del futuro
# MAGIC en las variables de entrada. Es una fuga sutil, porque no involucra al
# MAGIC target, y por eso mismo es facil de pasar por alto.
# MAGIC
# MAGIC Se ajusta un solo modelo sobre las coordenadas de recogida y se aplica a
# MAGIC ambos extremos, de modo que una zona signifique lo mismo como origen que
# MAGIC como destino.

# COMMAND ----------

ensamblador_pickup = VectorAssembler(
    inputCols=["pickup_latitude", "pickup_longitude"], outputCol="coords"
)
ensamblador_dropoff = VectorAssembler(
    inputCols=["dropoff_latitude", "dropoff_longitude"], outputCol="coords"
)

coords_entrenamiento = ensamblador_pickup.transform(
    df.filter(F.col("split_flag") == splits.SPLIT_TRAIN)
).select("coords")

modelo_zonas = KMeans(
    featuresCol="coords",
    predictionCol="zona",
    k=config.N_CLUSTERS_ZONA,
    seed=config.SEMILLA,
).fit(coords_entrenamiento)

print(f"Zonas ajustadas sobre {coords_entrenamiento.count():,} viajes de entrenamiento")

# COMMAND ----------

df_zonas = (
    modelo_zonas.transform(ensamblador_pickup.transform(df))
    .withColumnRenamed("zona", "pickup_cluster")
    .drop("coords")
)
df_zonas = (
    modelo_zonas.transform(ensamblador_dropoff.transform(df_zonas))
    .withColumnRenamed("zona", "dropoff_cluster")
    .drop("coords")
)

TBL_ZONAS = f"{config.CATALOGO}.{config.SCHEMA_GOLD}._trips_zonas"

(df_zonas.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true").saveAsTable(TBL_ZONAS))

df_zonas = spark.table(TBL_ZONAS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. `trips_analytics`
# MAGIC
# MAGIC Grano de viaje, con todo el detalle. Es la unica tabla donde aparece la
# MAGIC velocidad promedio, precisamente porque no puede llegar al modelo.
# MAGIC Incluye tambien las zonas, que el analisis exploratorio necesita para
# MAGIC estudiar los pares origen-destino.
# MAGIC
# MAGIC Particionada por mes: seis particiones de unas 240 mil filas. A este
# MAGIC volumen el particionamiento es demostrativo mas que necesario, y conviene
# MAGIC decirlo: el criterio real en produccion seria un tamano de particion
# MAGIC objetivo de entre 128 MB y 1 GB. Particionar por hora del dia generaria 24
# MAGIC carpetas de archivos diminutos y degradaria el rendimiento en lugar de
# MAGIC mejorarlo, que es el error mas comun al aplicar este patron.

# COMMAND ----------

df_analytics = df_zonas.withColumn(
    "velocidad_kmh",
    F.when(F.col("trip_duration") > 0,
           F.col("distancia_haversine_km") / (F.col("trip_duration") / 3600.0))
    .otherwise(F.lit(0.0)),
).withColumn("duracion_min", F.col("trip_duration") / 60.0)

(df_analytics.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .partitionBy("pickup_month")
 .saveAsTable(config.TBL_ANALYTICS))

print(f"{config.TBL_ANALYTICS}: {spark.table(config.TBL_ANALYTICS).count():,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. `agg_demanda_zona_hora`
# MAGIC
# MAGIC Tabla de hechos agregada con grano de zona de origen por hora por dia de
# MAGIC la semana. Existe para que las visualizaciones del analisis exploratorio no
# MAGIC recorran 1,45 millones de filas en cada consulta: son unos pocos miles de
# MAGIC registros que responden lo mismo.
# MAGIC
# MAGIC Se usa la mediana y no el promedio porque la distribucion de duraciones
# MAGIC esta fuertemente sesgada a la derecha, y unos pocos trayectos largos
# MAGIC desplazarian la media hacia arriba sin representar al viaje tipico.

# COMMAND ----------

df_agg = (
    df_zonas.groupBy("pickup_cluster", "pickup_hour", "pickup_dayofweek")
    .agg(
        F.count(F.lit(1)).alias("n_viajes"),
        F.expr("percentile_approx(trip_duration, 0.5)").alias("duracion_mediana_seg"),
        F.expr("percentile_approx(distancia_haversine_km, 0.5)").alias("distancia_mediana_km"),
        F.round(F.avg("passenger_count"), 2).alias("pasajeros_promedio"),
    )
    .withColumn(
        "velocidad_mediana_kmh",
        F.round(F.col("distancia_mediana_km") / (F.col("duracion_mediana_seg") / 3600.0), 2),
    )
    .withColumn("duracion_mediana_min", F.round(F.col("duracion_mediana_seg") / 60.0, 2))
)

(df_agg.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true").saveAsTable(config.TBL_AGG_DEMANDA))

print(f"{config.TBL_AGG_DEMANDA}: {spark.table(config.TBL_AGG_DEMANDA).count():,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. `trips_features`
# MAGIC
# MAGIC Seleccion por lista blanca, no por descarte. Con `drop()` se olvida una
# MAGIC columna y nadie se entera; con lista blanca, todo lo que no este declarado
# MAGIC en `config.FEATURES_MODELO` queda fuera por defecto.
# MAGIC
# MAGIC El target se guarda en sus dos formas. La transformacion logaritmica
# MAGIC responde al sesgo de la distribucion: sin ella, el error cuadratico queda
# MAGIC dominado por los trayectos mas largos y el modelo optimiza el caso raro a
# MAGIC costa del habitual. Ademas alinea el entrenamiento con la metrica oficial
# MAGIC de la competencia, que es logaritmica.
# MAGIC
# MAGIC Particionada por `split_flag`: cada corrida de entrenamiento lee su
# MAGIC particion sin recorrer la tabla completa. Particionar por mes no aportaria
# MAGIC nada aqui, porque el modelado nunca consulta por mes.

# COMMAND ----------

df_features = (
    df_zonas
    .withColumn(config.TARGET_LOG, F.log1p(F.col(config.TARGET)))
    .select(
        "id",
        *config.FEATURES_MODELO,
        config.TARGET,
        config.TARGET_LOG,
        "split_flag",
    )
)

(df_features.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true")
 .partitionBy("split_flag")
 .saveAsTable(config.TBL_FEATURES))

print(f"{config.TBL_FEATURES}: {spark.table(config.TBL_FEATURES).count():,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Verificacion anti fuga
# MAGIC
# MAGIC Comprobacion explicita de que ninguna columna prohibida sobrevivio hasta la
# MAGIC tabla que alimenta el modelo. Convierte una intencion de diseno en una
# MAGIC garantia que el pipeline verifica en cada ejecucion.

# COMMAND ----------

columnas_features = set(spark.table(config.TBL_FEATURES).columns)
fugas = columnas_features & set(config.COLUMNAS_PROHIBIDAS) - {config.TARGET, config.TARGET_LOG}

assert not fugas, f"Columnas con fuga de datos en la tabla de features: {fugas}"

for prohibida in ["dropoff_datetime", "velocidad_kmh", "duracion_min"]:
    assert prohibida not in columnas_features, f"'{prohibida}' llego a la tabla de features"

print("Sin fugas. Columnas de la tabla de features:")
for c in sorted(columnas_features):
    print(f"  {c}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verificacion de integridad y limpieza

# COMMAND ----------

n_features = spark.table(config.TBL_FEATURES).count()
n_analytics = spark.table(config.TBL_ANALYTICS).count()

assert n_features == n_silver, f"trips_features perdio filas: {n_features:,} vs {n_silver:,}"
assert n_analytics == n_silver, f"trips_analytics perdio filas: {n_analytics:,} vs {n_silver:,}"

nulos = spark.table(config.TBL_FEATURES).select(
    [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in config.FEATURES_MODELO]
).first().asDict()
con_nulos = {c: n for c, n in nulos.items() if n > 0}
assert not con_nulos, f"Features con valores nulos: {con_nulos}"

spark.sql(f"DROP TABLE IF EXISTS {TBL_ENRIQUECIDO}")
spark.sql(f"DROP TABLE IF EXISTS {TBL_ZONAS}")

print("Capa Gold construida y verificada.")

# COMMAND ----------

display(spark.table(config.TBL_FEATURES).limit(10))

# COMMAND ----------

