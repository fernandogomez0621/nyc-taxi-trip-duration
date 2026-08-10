"""
Definicion de las variables del modelo y de la validacion temporal.

Deliberadamente agnostico del framework de entrenamiento: declara que columnas
entran, en que orden y cuales son categoricas, sin depender de Spark ni de
scikit-learn. Eso permite probar la coherencia de las listas sin arrancar
ninguna sesion.
"""

from __future__ import annotations

# Variables que son etiquetas y no magnitudes. Un modelo de arboles las trata
# por particion y no supone orden entre ellas, de modo que no requieren
# expansion one-hot; un modelo lineal si, porque de lo contrario interpretaria
# que la hora 14 vale el doble que la 7 o que la zona 18 es mayor que la 9.
CATEGORICAS = [
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "pickup_cluster",
    "dropoff_cluster",
    "vendor_id",
]

# Variables continuas o binarias, que entran sin transformar.
NUMERICAS = [
    "distancia_haversine_km",
    "distancia_manhattan_km",
    "bearing_deg",
    "passenger_count",
    "store_and_fwd_flag",
    "is_weekend",
    "es_aeropuerto",
]

# Orden canonico de las columnas del modelo. Fijarlo en un unico sitio evita
# que el vector de importancias se asocie a las variables equivocadas.
COLUMNAS_MODELO = NUMERICAS + CATEGORICAS


def indices_categoricas(columnas: list[str] = None) -> list[int]:
    """Posiciones de las variables categoricas dentro del orden canonico."""
    columnas = columnas or COLUMNAS_MODELO
    return [columnas.index(c) for c in CATEGORICAS]


def ventanas_temporales(fecha_min: str, fecha_max: str, n_ventanas: int = 3) -> list[dict]:
    """
    Genera ventanas de validacion hacia adelante sobre el conjunto de
    entrenamiento.

    Cada ventana entrena con todo lo anterior a un corte y valida sobre el
    periodo siguiente. Es el equivalente temporal de la validacion cruzada: el
    K-fold habitual mezcla periodos y permitiria entrenar con datos posteriores
    a los de validacion, exactamente el problema que el split temporal existe
    para evitar.
    """
    from datetime import datetime, timedelta

    inicio = datetime.strptime(fecha_min, "%Y-%m-%d")
    fin = datetime.strptime(fecha_max, "%Y-%m-%d")
    paso = (fin - inicio).days // (n_ventanas + 1)

    ventanas = []
    for i in range(1, n_ventanas + 1):
        ventanas.append({
            "ventana": i,
            "entrena_hasta": (inicio + timedelta(days=paso * i)).strftime("%Y-%m-%d"),
            "valida_hasta": (inicio + timedelta(days=paso * (i + 1))).strftime("%Y-%m-%d"),
        })
    return ventanas
