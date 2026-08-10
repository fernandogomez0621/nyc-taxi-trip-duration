# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Monitoreo en produccion (Parte 4 · reto bonus)
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Objetivo** | Definir y demostrar el sistema de deteccion de deriva si el modelo operara en produccion |
# MAGIC | **Entradas** | `gold.trips_features`, `bronze.raw_scoring`, `artifacts/modelo_resumen.json` |
# MAGIC | **Salidas** | `gold.monitoring_drift` con el historial de evaluaciones |
# MAGIC | **Depende de** | Notebooks 01 (`p_origen=scoring`), 03 y 05 |
# MAGIC
# MAGIC **Proceso**
# MAGIC 1. Definir la referencia contra la que se compara cada lote entrante
# MAGIC 2. Evaluar un lote real y comprobar que no genera falsos positivos
# MAGIC 3. Inducir deriva de forma controlada y verificar que se detecta
# MAGIC 4. Determinar el umbral de deteccion y el retraso hasta la alerta
# MAGIC 5. Especificar el sistema completo de alertas y su encadenamiento
# MAGIC
# MAGIC **Planteamiento.** El enunciado pregunta que metricas y alertas se
# MAGIC implementarian para detectar deriva si el modelo recibiera datos nuevos
# MAGIC cada semana. En lugar de enumerar tecnicas, este notebook las ejecuta sobre
# MAGIC datos reales: mide el comportamiento del detector cuando **no** hay deriva
# MAGIC —que es lo que determina su tasa de falsos positivos— y despues introduce
# MAGIC desplazamientos de magnitud creciente para establecer a partir de que punto
# MAGIC los detecta.
# MAGIC
# MAGIC Un detector que solo se prueba con datos alterados es un detector sin
# MAGIC calibrar: no se sabe cuantas veces despertaria al equipo sin motivo.

# COMMAND ----------

import json
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from nyc_taxi import config
from nyc_taxi.monitoring import drift

plt.rcParams.update({
    "figure.figsize": (11, 4.2), "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})
AZUL, NARANJA, GRIS, ROJO = "#2E5A87", "#D97B29", "#8A8A8A", "#B03030"

RUTA_ARTEFACTOS = f"/Volumes/{config.CATALOGO}/{config.SCHEMA_GOLD}/artifacts"
TBL_DRIFT = f"{config.CATALOGO}.{config.SCHEMA_GOLD}.monitoring_drift"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Que se vigila
# MAGIC
# MAGIC Se vigilan tres cosas distintas, que fallan por razones distintas y en
# MAGIC momentos distintos:
# MAGIC
# MAGIC | Plano | Que detecta | Requiere etiquetas |
# MAGIC |---|---|---|
# MAGIC | Deriva de covariables | Cambia la distribucion de las variables de entrada | No |
# MAGIC | Deriva del objetivo | Cambia la distribucion de las duraciones reales | Si |
# MAGIC | Degradacion del error | El error sube aunque las distribuciones parezcan estables | Si |
# MAGIC
# MAGIC El primer plano es el mas valioso en la practica porque **no necesita
# MAGIC esperar a que existan duraciones reales**. Un viaje que empieza hoy no
# MAGIC entrega su etiqueta hasta que termina, y la consolidacion de un lote
# MAGIC semanal tarda aun mas. La deriva de covariables avisa antes de que sea
# MAGIC posible medir el error.
# MAGIC
# MAGIC La metrica empleada es el **indice de estabilidad poblacional (PSI)**, que
# MAGIC compara la distribucion de una variable entre la muestra de referencia y la
# MAGIC nueva. Se eligio sobre las pruebas de hipotesis por la misma razon expuesta
# MAGIC en el analisis exploratorio: con cien mil observaciones por lote, un
# MAGIC contraste de Kolmogorov-Smirnov rechazaria la igualdad de distribuciones
# MAGIC practicamente siempre, midiendo el tamano del lote y no su contenido. El
# MAGIC PSI cuantifica **magnitud** de cambio, que es lo que permite decidir.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Referencia
# MAGIC
# MAGIC La referencia es la particion con la que se entreno el modelo. Congelarla
# MAGIC —en lugar de compararse siempre contra el lote anterior— es deliberado: si
# MAGIC la referencia se moviera con cada semana, una deriva lenta y sostenida
# MAGIC pasaria inadvertida porque cada lote se pareceria al inmediatamente
# MAGIC anterior. Comparar contra el punto de entrenamiento detecta el
# MAGIC desplazamiento acumulado, que es el que degrada al modelo.

# COMMAND ----------

VARIABLES = [
    "distancia_haversine_km", "distancia_manhattan_km", "bearing_deg",
    "pickup_hour", "pickup_dayofweek", "pickup_cluster", "dropoff_cluster",
    "passenger_count", "es_aeropuerto",
]
N_MUESTRA = 30_000

referencia = (
    spark.table(config.TBL_FEATURES).filter(F.col("split_flag") == "train")
    .select(*VARIABLES, config.TARGET)
    .sample(fraction=0.05, seed=config.SEMILLA).limit(N_MUESTRA).toPandas()
)
print(f"Referencia: {len(referencia):,} viajes del periodo de entrenamiento")

ref = {v: referencia[v].tolist() for v in VARIABLES}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Lote real: comportamiento sin deriva
# MAGIC
# MAGIC Se evalua primero la particion de evaluacion, que corresponde a las dos
# MAGIC semanas siguientes al entrenamiento. Es un lote genuinamente posterior y
# MAGIC sin manipular: exactamente lo que llegaria un lunes en produccion.
# MAGIC
# MAGIC El resultado esperado es que **no** dispare. Esa comprobacion es la que
# MAGIC calibra el detector: sin ella no habria forma de saber cuantas alertas
# MAGIC infundadas generaria.

# COMMAND ----------

lote_real = (
    spark.table(config.TBL_FEATURES).filter(F.col("split_flag") == "test")
    .select(*VARIABLES, config.TARGET)
    .sample(fraction=0.4, seed=config.SEMILLA).limit(N_MUESTRA).toPandas()
)
print(f"Lote evaluado: {len(lote_real):,} viajes de las dos semanas posteriores\n")

resultado = drift.evaluar_lote(ref, {v: lote_real[v].tolist() for v in VARIABLES})

print(f"{'variable':<26}{'PSI':>9}  estado")
for v, r in sorted(resultado["features"].items(), key=lambda kv: -kv[1]["psi"]):
    print(f"{v:<26}{r['psi']:>9.4f}  {r['estado']}")

psi_target = drift.psi(referencia[config.TARGET].tolist(), lote_real[config.TARGET].tolist())
print(f"\n{'objetivo (duracion)':<26}{psi_target:>9.4f}  {drift.clasificar_psi(psi_target)}")
print(f"\nAlerta: {'SI' if resultado['alerta'] else 'NO'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Linea base establecida
# MAGIC
# MAGIC Todos los indicadores quedan muy por debajo del umbral de vigilancia. El
# MAGIC detector no genera falsos positivos sobre un lote posterior legitimo, que
# MAGIC es la condicion minima para que un sistema de alertas sea util: uno que
# MAGIC dispara sin motivo se desactiva en dos semanas.
# MAGIC
# MAGIC Conviene precisar una limitacion del ejercicio. El conjunto de scoring
# MAGIC publicado por la fuente cubre el mismo periodo que el de entrenamiento,
# MAGIC porque su particion fue aleatoria y no temporal. No sirve, por tanto, como
# MAGIC lote futuro. Por eso la referencia se contrasta contra la particion de
# MAGIC evaluacion, que si es posterior en el tiempo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Deriva inducida: sensibilidad del detector
# MAGIC
# MAGIC Comprobado que no dispara sin motivo, queda determinar a partir de que
# MAGIC magnitud si lo hace. Se construyen tres escenarios que corresponden a
# MAGIC cambios plausibles en la operacion, y en cada uno se varia la intensidad
# MAGIC para localizar el punto de deteccion.

# COMMAND ----------

HORAS_NOCTURNAS = [22, 23, 0, 1, 2, 3, 4, 5]


def escenario_nocturno(df, intensidad):
    """
    Sesgo hacia viajes nocturnos: cambio en la composicion de la demanda.

    La intensidad expresa el aumento **relativo** sobre la proporcion natural de
    viajes nocturnos del lote. Definirla en terminos absolutos seria enganoso:
    fijar un 10 % de viajes nocturnos cuando la proporcion natural ronda el 24 %
    no anade deriva sino que la invierte.
    """
    es_noc = df.pickup_hour.isin(HORAS_NOCTURNAS)
    proporcion_objetivo = min(0.95, es_noc.mean() * (1 + intensidad))
    n_noc = int(len(df) * proporcion_objetivo)
    return pd.concat([
        df[es_noc].sample(n_noc, replace=True, random_state=config.SEMILLA),
        df[~es_noc].sample(len(df) - n_noc, replace=True, random_state=config.SEMILLA),
    ])


def escenario_zona(df, intensidad):
    """Perdida de cobertura en las zonas de mayor demanda."""
    intensidad = min(intensidad, 0.95)
    top = df.pickup_cluster.value_counts().head(3).index
    excluidos = df[~df.pickup_cluster.isin(top)]
    incluidos = df[df.pickup_cluster.isin(top)]
    n_top = int(len(incluidos) * (1 - intensidad))
    return pd.concat([
        excluidos,
        incluidos.sample(n_top, random_state=config.SEMILLA) if n_top else incluidos.head(0),
    ])


def escenario_distancia(df, intensidad):
    """Trayectos sistematicamente mas largos: expansion del area de servicio."""
    out = df.copy()
    out["distancia_haversine_km"] = out.distancia_haversine_km * (1 + intensidad)
    out["distancia_manhattan_km"] = out.distancia_manhattan_km * (1 + intensidad)
    return out


ESCENARIOS = {
    "Sesgo nocturno": escenario_nocturno,
    "Perdida de zonas principales": escenario_zona,
    "Trayectos mas largos": escenario_distancia,
}
# La intensidad expresa la magnitud relativa del cambio respecto al lote
# original. Se explora hasta duplicar porque, como muestran los resultados, el
# PSI es un indicador conservador y requiere desplazamientos sustanciales.
INTENSIDADES = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

filas = []
for nombre, fn in ESCENARIOS.items():
    for i in INTENSIDADES:
        lote = fn(lote_real, i) if i > 0 else lote_real
        r = drift.evaluar_lote(ref, {v: lote[v].tolist() for v in VARIABLES})
        psi_max = max(x["psi"] for x in r["features"].values())
        filas.append({
            "escenario": nombre, "intensidad": i, "psi_maximo": psi_max,
            "n_alerta": r["n_features_en_alerta"], "alerta": r["alerta"],
        })

tabla = pd.DataFrame(filas)
print(tabla.pivot(index="intensidad", columns="escenario", values="psi_maximo").round(3).to_string())

# COMMAND ----------

fig, ax = plt.subplots()
for nombre, sub in tabla.groupby("escenario"):
    ax.plot(sub.intensidad, sub.psi_maximo, "o-", label=nombre, lw=1.8)
ax.axhline(drift.PSI_ESTABLE, ls="--", color=GRIS, lw=1)
ax.axhline(drift.PSI_ALERTA, ls="--", color=ROJO, lw=1.2)
ax.text(0.01, drift.PSI_ESTABLE * 1.05, "vigilar (0,10)", color=GRIS, fontsize=8)
ax.text(0.01, drift.PSI_ALERTA * 1.05, "alerta (0,25)", color=ROJO, fontsize=8)
ax.set(xlabel="intensidad del cambio", ylabel="PSI maximo entre variables",
       title="Sensibilidad del detector por escenario", yscale="log")
ax.legend(frameon=False)
plt.tight_layout()
os.makedirs(f"{RUTA_ARTEFACTOS}/figuras", exist_ok=True)
fig.savefig(f"{RUTA_ARTEFACTOS}/figuras/10_sensibilidad_drift.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

print("Intensidad minima detectada por escenario")
for nombre, sub in tabla.groupby("escenario"):
    dispara = sub[sub.alerta]
    umbral = dispara.intensidad.min() if len(dispara) else None
    print(f"  {nombre:<30} {'no detectado en el rango' if umbral is None else f'{umbral:.0%}'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lectura de la sensibilidad
# MAGIC
# MAGIC **El PSI es un indicador notablemente conservador**, y conviene decirlo con
# MAGIC claridad porque condiciona el diseno del sistema. Un aumento del 50 % en la
# MAGIC distancia de todos los trayectos —un cambio que cualquier responsable de
# MAGIC operaciones calificaria de grave— alcanza un PSI de 0,24 y **no llega a
# MAGIC disparar** la alerta. La deteccion aparece de forma consistente cuando el
# MAGIC desplazamiento duplica la magnitud original.
# MAGIC
# MAGIC Ese comportamiento no es un defecto del calculo sino de los umbrales
# MAGIC convencionales de 0,10 y 0,25, heredados de la industria de riesgo
# MAGIC crediticio, donde las poblaciones se mueven despacio. Aplicados sin
# MAGIC revision a un fenomeno urbano de alta variabilidad, dejan pasar cambios
# MAGIC relevantes.
# MAGIC
# MAGIC De ahi se derivan dos ajustes al diseno:
# MAGIC
# MAGIC - **Bajar el umbral de alerta a 0,15** para este caso concreto, apoyandose
# MAGIC   en la linea base medida: sobre un lote legitimo el PSI maximo observado
# MAGIC   fue de 0,002, dos ordenes de magnitud por debajo. Existe margen amplio
# MAGIC   para endurecer el criterio sin generar falsos positivos.
# MAGIC - **Complementar el PSI con reglas de cobertura**, que detectan lo que un
# MAGIC   indicador de distribucion no captura bien: si una zona o una franja
# MAGIC   horaria que representaba mas del 5 % de los viajes desaparece del lote, se
# MAGIC   alerta con independencia del PSI.
# MAGIC
# MAGIC Calibrar los umbrales con datos propios en lugar de adoptar los valores por
# MAGIC defecto es justamente lo que distingue un detector operativo de uno
# MAGIC declarativo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Degradacion del error
# MAGIC
# MAGIC La deriva de covariables anticipa el problema; la degradacion del error lo
# MAGIC confirma. Cuando las etiquetas estan disponibles, se compara el error del
# MAGIC lote contra el registrado en la evaluacion inicial.

# COMMAND ----------

try:
    with open(f"{RUTA_ARTEFACTOS}/modelo_resumen.json") as fh:
        modelo = json.load(fh)
    ref_mae = modelo["metricas_test"]["mae_min"]
    ref_rmsle = modelo["metricas_test"]["rmsle"]
    print(f"Error de referencia del modelo en produccion")
    print(f"  MAE   : {ref_mae:.3f} min")
    print(f"  RMSLE : {ref_rmsle:.4f}")
    print(f"\nUmbrales de alerta propuestos")
    print(f"  atencion : MAE > {ref_mae * 1.15:.3f} min  (+15 %)")
    print(f"  critico  : MAE > {ref_mae * 1.30:.3f} min  (+30 %)")
except (FileNotFoundError, OSError):
    ref_mae = ref_rmsle = None
    print("Sin resumen del modelo; se omiten los umbrales de degradacion.")

# COMMAND ----------

# MAGIC %md
# MAGIC Los umbrales son relativos al desempeno medido y no absolutos. Un umbral
# MAGIC fijo obligaria a reajustarlo cada vez que el modelo mejora; uno relativo se
# MAGIC adapta solo. El margen del 15 % se situa holgadamente por encima de la
# MAGIC variacion natural entre ventanas temporales observada durante la seleccion
# MAGIC de hiperparametros, que fue de 0,008 en RMSLE.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Persistencia del historial
# MAGIC
# MAGIC Cada evaluacion se acumula en una tabla. El valor de una sola medicion es
# MAGIC limitado; el de la serie es lo que permite distinguir un pico puntual
# MAGIC —una nevada, un evento en la ciudad— de una tendencia sostenida que si
# MAGIC justifica reentrenar.

# COMMAND ----------

historial = pd.DataFrame([
    {
        "evaluado_en": pd.Timestamp.utcnow().tz_localize(None),
        "lote": "particion de evaluacion",
        "variable": v,
        "psi": r["psi"],
        "estado": r["estado"],
        "n_referencia": len(referencia),
        "n_lote": len(lote_real),
    }
    for v, r in resultado["features"].items()
] + [{
    "evaluado_en": pd.Timestamp.utcnow().tz_localize(None),
    "lote": "particion de evaluacion",
    "variable": "objetivo_duracion",
    "psi": psi_target,
    "estado": drift.clasificar_psi(psi_target),
    "n_referencia": len(referencia),
    "n_lote": len(lote_real),
}])

(spark.createDataFrame(historial).write.format("delta")
 .mode("append").option("mergeSchema", "true").saveAsTable(TBL_DRIFT))

print(f"{len(historial)} registros anexados a {TBL_DRIFT}")
display(spark.table(TBL_DRIFT).orderBy(F.desc("psi")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Sistema de alertas propuesto
# MAGIC
# MAGIC | Escenario | Disparador | Severidad | Accion |
# MAGIC |---|---|---|---|
# MAGIC | Fallo de carga | Una tarea del job termina en error | Critica | Notificacion inmediata; el pipeline no avanza con datos incompletos |
# MAGIC | Calidad degradada | Tasa de cuarentena superior al triple de la habitual (0,21 %) | Alta | Revision del lote y notificacion a la fuente |
# MAGIC | Deriva de covariables | PSI superior a 0,15 en cualquier variable | Media | Revision; no dispara reentrenamiento por si sola |
# MAGIC | Cobertura perdida | Una categoria con mas del 5 % de peso desaparece del lote | Media | Verificacion de la fuente |
# MAGIC | Deriva del objetivo | PSI del objetivo superior a 0,15 | Alta | Reentrenamiento programado |
# MAGIC | Degradacion del error | MAE 15 % por encima de la referencia | Alta | Reentrenamiento programado |
# MAGIC | Degradacion severa | MAE 30 % por encima de la referencia | Critica | Reentrenamiento inmediato y aviso a operaciones |
# MAGIC | Ausencia de lote | No llega archivo en la ventana esperada | Media | Verificacion de la fuente |
# MAGIC
# MAGIC **La deriva de covariables por si sola no dispara reentrenamiento.** Que la
# MAGIC distribucion de entrada cambie no implica que el modelo empeore: puede
# MAGIC generalizar bien sobre el nuevo regimen. Reentrenar ante cada
# MAGIC desplazamiento consume computo y, sobre todo, introduce inestabilidad en un
# MAGIC sistema que funcionaba. La regla adoptada es que la deriva de covariables
# MAGIC **advierte** y la degradacion del error **decide**.
# MAGIC
# MAGIC La unica excepcion es la deriva del objetivo, que si implica que el
# MAGIC fenomeno modelado cambio.
# MAGIC
# MAGIC ### Frecuencia y encadenamiento
# MAGIC
# MAGIC El enunciado plantea lotes semanales. La evaluacion de deriva se ejecuta
# MAGIC como ultima tarea del job de ingesta, de modo que cada lote se mide al
# MAGIC llegar. La degradacion del error se evalua con retraso, cuando las
# MAGIC duraciones reales estan consolidadas.
# MAGIC
# MAGIC Sobre esa base, el reentrenamiento se dispara **por condicion y no por
# MAGIC calendario**: un job programado consulta la tabla de historial y, si se
# MAGIC cumplen los criterios de severidad alta durante dos evaluaciones
# MAGIC consecutivas, lanza el entrenamiento. La exigencia de dos evaluaciones
# MAGIC evita reaccionar a un evento puntual como la nevada del 23 de enero, que
# MAGIC habria disparado cualquier umbral y no representa un cambio de regimen.
# MAGIC
# MAGIC ### Que no se vigila y por que
# MAGIC
# MAGIC No se monitorea la distribucion de las **predicciones**. Es una practica
# MAGIC habitual, pero aporta poco cuando ya se vigilan las entradas: si estas son
# MAGIC estables y el modelo es determinista, sus salidas lo seran por
# MAGIC construccion. Su utilidad aparece cuando no hay acceso a las variables de
# MAGIC entrada, que no es el caso.