# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Aprovisionamiento del entorno
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Objetivo** | Crear catalogo, esquemas y volumenes antes de ejecutar el pipeline |
# MAGIC | **Entradas** | Ninguna |
# MAGIC | **Salidas** | Estructura de Unity Catalog lista para la ingesta |
# MAGIC | **Depende de** | Permisos para crear catalogo en el metastore |
# MAGIC
# MAGIC **Proceso**
# MAGIC 1. Determinar el catalogo destino segun el tipo de espacio de trabajo
# MAGIC 2. Crear los tres esquemas del patron medallion
# MAGIC 3. Crear el volumen de aterrizaje y el de artefactos
# MAGIC 4. Verificar la estructura resultante
# MAGIC
# MAGIC **Portabilidad entre espacios de trabajo.** En Databricks Free Edition el
# MAGIC unico catalogo escribible es `workspace`. En un espacio con metastore
# MAGIC propio —Azure Databricks, por ejemplo— conviene un catalogo dedicado. El
# MAGIC codigo lee el nombre de la variable de entorno `NYC_TAXI_CATALOG`, de modo
# MAGIC que cambiar de entorno no exige modificar ningun archivo fuente.

# COMMAND ----------

dbutils.widgets.text("p_catalogo", "workspace", "Catalogo destino")
dbutils.widgets.dropdown("p_crear_catalogo", "no", ["si", "no"], "Crear el catalogo")

CATALOGO = dbutils.widgets.get("p_catalogo").strip()
CREAR_CATALOGO = dbutils.widgets.get("p_crear_catalogo") == "si"

print(f"Catalogo destino : {CATALOGO}")
print(f"Crear catalogo   : {CREAR_CATALOGO}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Catalogo
# MAGIC
# MAGIC Solo se crea cuando el espacio de trabajo lo permite. En Free Edition la
# MAGIC creacion de catalogos esta restringida y debe usarse `workspace`, que ya
# MAGIC existe.

# COMMAND ----------

if CREAR_CATALOGO:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOGO}")
    spark.sql(f"COMMENT ON CATALOG {CATALOGO} IS "
              f"'Prediccion de duracion de viajes en taxi de Nueva York'")
    print(f"Catalogo {CATALOGO} disponible")
else:
    print(f"Se usara el catalogo existente {CATALOGO}")

spark.sql(f"USE CATALOG {CATALOGO}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Esquemas del patron medallion
# MAGIC
# MAGIC Un esquema por capa, con la responsabilidad de cada una documentada en el
# MAGIC propio metastore para que sea consultable sin abrir el codigo.

# COMMAND ----------

ESQUEMAS = {
    "bronze": "Ingesta cruda con trazabilidad. Los valores no se alteran; solo se anexan columnas de auditoria.",
    "silver": "Datos tipados y validados. Lo que no supera las reglas de calidad se aisla en cuarentena, no se borra.",
    "gold": "Tablas de consumo. Una por consumidor: analitica de detalle, agregados para visualizacion y features para el modelo.",
}

for nombre, descripcion in ESQUEMAS.items():
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOGO}.{nombre}")
    spark.sql(f"COMMENT ON SCHEMA {CATALOGO}.{nombre} IS '{descripcion}'")
    print(f"  {nombre:<8} listo")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Volumenes
# MAGIC
# MAGIC Dos volumenes con propositos distintos. `bronze.landing` recibe los
# MAGIC archivos de origen; `gold.artifacts` guarda los resultados de analisis
# MAGIC —figuras y hallazgos serializados— que no son tablas y por tanto no deben
# MAGIC ocupar espacio en el esquema de consumo.

# COMMAND ----------

VOLUMENES = {
    "bronze.landing": "Zona de aterrizaje de los archivos de origen sin procesar.",
    "gold.artifacts": "Artefactos de analisis: figuras y hallazgos serializados.",
}

for ruta, descripcion in VOLUMENES.items():
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOGO}.{ruta}")
    spark.sql(f"COMMENT ON VOLUME {CATALOGO}.{ruta} IS '{descripcion}'")
    print(f"  /Volumes/{CATALOGO}/{ruta.replace('.', '/')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verificacion

# COMMAND ----------

print(f"Esquemas en {CATALOGO}:")
for fila in spark.sql(f"SHOW SCHEMAS IN {CATALOGO}").collect():
    print(f"  {fila[0]}")

print("\nVolumenes:")
for esquema in ["bronze", "gold"]:
    for fila in spark.sql(f"SHOW VOLUMES IN {CATALOGO}.{esquema}").collect():
        print(f"  {esquema}.{fila['volume_name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Siguientes pasos
# MAGIC
# MAGIC 1. Subir `train.csv` y `test.csv` a `/Volumes/<catalogo>/bronze/landing/`
# MAGIC 2. Si el catalogo no es `workspace`, definir la variable de entorno en el
# MAGIC    entorno de ejecucion o ajustar el valor por defecto en `config.py`
# MAGIC 3. Ejecutar `00_tests`, y despues el pipeline en orden
# MAGIC
# MAGIC La celda siguiente deja constancia de la ruta exacta a la que subir los
# MAGIC archivos y del valor que debe tomar la variable de entorno.

# COMMAND ----------

print("Subir los archivos de origen a:")
print(f"  /Volumes/{CATALOGO}/bronze/landing/train.csv")
print(f"  /Volumes/{CATALOGO}/bronze/landing/test.csv")

if CATALOGO != "workspace":
    print(f"\nDefinir antes de ejecutar el pipeline:")
    print(f'  os.environ["NYC_TAXI_CATALOG"] = "{CATALOGO}"')
    print("  o modificar el valor por defecto en src/nyc_taxi/config.py")
