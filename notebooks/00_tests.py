# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 00 · Tests unitarios
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Objetivo** | Verificar la logica pura del proyecto antes de gastar computo en el pipeline |
# MAGIC | **Entradas** | `src/nyc_taxi/` y `tests/unit/` |
# MAGIC | **Salidas** | Reporte de ejecucion. No produce ni modifica datos |
# MAGIC | **Depende de** | Nada. Es el primer notebook a ejecutar |
# MAGIC
# MAGIC **Proceso**
# MAGIC 1. Instalar pytest en el entorno de la sesion
# MAGIC 2. Localizar la raiz del repositorio de forma dinamica
# MAGIC 3. Ejecutar la suite completa
# MAGIC 4. Fallar el notebook si algun test no pasa
# MAGIC
# MAGIC Las pruebas cubren geometria, reglas de validacion, tipado, split, metricas
# MAGIC y deteccion de deriva. Son logica pura, sin sesion de Spark, y corren en
# MAGIC menos de un segundo.

# COMMAND ----------

# MAGIC %pip install pytest --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import subprocess
import sys

# El sistema de archivos de /Workspace no permite crear directorios __pycache__,
# y pytest intenta escribir ahí el caché de reescritura de asserts. Desactivar la
# escritura de bytecode evita el OSError [Errno 95] Operation not supported.
ENTORNO = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


# Misma localización dinámica que usan los demás notebooks: sube desde el
# directorio actual hasta encontrar la carpeta que contiene src/. Nada de rutas
# fijas, para que el repo pueda vivir a cualquier profundidad del workspace.
def raiz_repo(marcador="src", max_niveles=10):
    ruta = os.getcwd()
    for _ in range(max_niveles):
        if os.path.isdir(os.path.join(ruta, marcador)):
            return ruta
        padre = os.path.dirname(ruta)
        if padre == ruta:
            break
        ruta = padre
    raise RuntimeError(
        f"No se encontró la raíz del repo (carpeta con '{marcador}/') "
        f"partiendo de {os.getcwd()}."
    )


RAIZ = raiz_repo()
print(f"Raíz del repo: {RAIZ}")

# COMMAND ----------

resultado = subprocess.run(
    [sys.executable, "-m", "pytest", "-v", "--tb=short", "-p", "no:cacheprovider"],
    cwd=RAIZ,
    env=ENTORNO,
    capture_output=True,
    text=True,
)

print(resultado.stdout)
if resultado.stderr:
    print("--- stderr ---")
    print(resultado.stderr)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resultado
# MAGIC
# MAGIC El `assert` hace que el notebook falle si algún test falla, de modo que
# MAGIC pueda encadenarse como primera tarea de un job y detener el pipeline antes
# MAGIC de tocar el dato.

# COMMAND ----------

assert resultado.returncode == 0, (
    f"La suite de tests falló (código {resultado.returncode}). "
    "Revisar la salida de la celda anterior."
)

print("Todos los tests pasaron.")

# COMMAND ----------

