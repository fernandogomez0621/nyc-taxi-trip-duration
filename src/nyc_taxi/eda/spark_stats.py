"""
Estadistica descriptiva calculada en Spark sobre la poblacion completa.

Existe para que los tamanos de efecto reportados no dependan de una muestra. Las
funciones equivalentes de `stats.py` operan sobre secuencias en memoria y sirven
para pruebas automaticas y para calculos sobre muestras pequenas; estas recorren
la tabla entera de forma distribuida y no traen al driver mas que un punado de
filas agregadas.

Definicion empleada:

    eta cuadrado = suma de cuadrados entre grupos / suma de cuadrados total

que equivale a la proporcion de la varianza de la variable de interes que
explica la pertenencia a cada grupo.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def eta_cuadrado_spark(df: DataFrame, col_valor: str, col_grupo: str) -> float:
    """
    Razon de correlacion entre una variable numerica y un factor categorico,
    calculada sobre todas las filas del DataFrame.

    Requiere dos pasadas: una para la media y la varianza globales, otra para las
    medias por grupo. El resultado que viaja al driver son tantas filas como
    grupos tenga el factor, no la tabla completa.
    """
    global_ = df.select(
        F.avg(col_valor).alias("media"),
        F.count(F.lit(1)).alias("n"),
        F.var_pop(col_valor).alias("var"),
    ).first()

    if not global_["var"]:
        return 0.0

    ss_total = global_["var"] * global_["n"]

    ss_entre = (
        df.groupBy(col_grupo)
        .agg(F.count(F.lit(1)).alias("n_g"), F.avg(col_valor).alias("media_g"))
        .select(F.sum(F.col("n_g") * F.pow(F.col("media_g") - F.lit(global_["media"]), 2)).alias("ss"))
        .first()["ss"]
    )

    return float(ss_entre / ss_total) if ss_total else 0.0


def coeficientes_regresion_simple(df: DataFrame, col_x: str, col_y: str) -> tuple[float, float, float]:
    """
    Ajusta y = intercepto + pendiente * x mediante momentos, en una sola pasada.

    Devuelve (intercepto, pendiente, r2). Se usa para remover el efecto del
    predictor dominante antes de medir efectos parciales.
    """
    m = df.select(
        F.avg(col_x).alias("mx"), F.avg(col_y).alias("my"),
        F.covar_pop(col_x, col_y).alias("cov"),
        F.var_pop(col_x).alias("vx"), F.var_pop(col_y).alias("vy"),
    ).first()

    if not m["vx"]:
        return 0.0, 0.0, 0.0

    pendiente = m["cov"] / m["vx"]
    intercepto = m["my"] - pendiente * m["mx"]
    r2 = (m["cov"] ** 2) / (m["vx"] * m["vy"]) if m["vy"] else 0.0
    return float(intercepto), float(pendiente), float(r2)


def agregar_residuos(df: DataFrame, col_x: str, col_y: str, col_salida: str = "residuo"):
    """
    Anexa la columna de residuos de `col_y` tras descontar el efecto lineal de
    `col_x`, y devuelve tambien los coeficientes del ajuste.

    El eta cuadrado calculado sobre esta columna mide el efecto **parcial** de un
    factor: cuanto explica una vez removido el predictor dominante. Sin ese
    ajuste, un factor puede parecer irrelevante solo porque su relacion con la
    variable de interes queda enmascarada por otro con el que esta
    correlacionado.
    """
    intercepto, pendiente, r2 = coeficientes_regresion_simple(df, col_x, col_y)
    df_res = df.withColumn(
        col_salida,
        F.col(col_y) - (F.lit(intercepto) + F.lit(pendiente) * F.col(col_x)),
    )
    return df_res, {"intercepto": intercepto, "pendiente": pendiente, "r2": r2}


def tabla_efectos(df: DataFrame, col_valor: str, col_residuo: str, factores: list[str]) -> list[dict]:
    """
    Calcula el efecto marginal y el parcial de cada factor, ordenados por el
    parcial de mayor a menor.
    """
    filas = []
    for f in factores:
        filas.append({
            "factor": f,
            "eta2_marginal": round(eta_cuadrado_spark(df, col_valor, f), 6),
            "eta2_parcial": round(eta_cuadrado_spark(df, col_residuo, f), 6),
        })
    return sorted(filas, key=lambda r: -r["eta2_parcial"])
