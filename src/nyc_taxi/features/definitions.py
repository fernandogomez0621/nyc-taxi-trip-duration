"""
Feature engineering: definiciones puras.

Lógica sin Spark, testeable con pytest sin necesidad de una sesión. Las
versiones en columnas de Spark viven en `transformations.py` y deben producir
exactamente el mismo resultado que estas; los tests verifican esa equivalencia.
"""

from __future__ import annotations

import math

RADIO_TIERRA_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distancia de círculo máximo entre dos puntos, en kilómetros.

    Es la distancia en línea recta sobre la superficie terrestre. Subestima el
    trayecto real de un taxi, que está obligado a seguir las calles, pero es la
    base sobre la que se construye la aproximación de Manhattan.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * RADIO_TIERRA_KM * math.asin(math.sqrt(a))


def manhattan_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distancia de trayecto en cuadrícula, en kilómetros.

    Suma el desplazamiento norte-sur y el este-oeste por separado, cada uno
    calculado con haversine. Aproxima mejor el recorrido real que la línea recta
    porque Manhattan está trazada como retícula y un taxi no puede cortar en
    diagonal a través de las manzanas.
    """
    d_lat = haversine_km(lat1, lon1, lat2, lon1)
    d_lon = haversine_km(lat1, lon1, lat1, lon2)
    return d_lat + d_lon


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Rumbo inicial del trayecto en grados (0 = norte, 90 = este).

    Captura la dirección del viaje, que en NYC no es simétrica: entrar a
    Manhattan en hora pico y salir de ella tienen perfiles de congestión
    distintos aunque la distancia sea idéntica.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def es_fin_de_semana(dia_semana: int) -> int:
    """
    Bandera de fin de semana.

    Usa la convención de `pyspark.sql.functions.dayofweek`, donde 1 = domingo y
    7 = sábado. Ojo: NO es la convención ISO de `datetime.isoweekday()`, donde
    1 = lunes. Mezclarlas es un error silencioso que desplaza la bandera dos
    días y no rompe nada visiblemente.
    """
    return int(dia_semana in (1, 7))


def velocidad_kmh(distancia_km: float, duracion_seg: float) -> float:
    """
    Velocidad media del trayecto.

    ATENCIÓN: deriva del target (duracion). Es válida para el análisis
    exploratorio y para detectar registros absurdos, pero NUNCA puede entrar
    como variable de entrada del modelo. Ver `config.COLUMNAS_PROHIBIDAS`.
    """
    if duracion_seg <= 0:
        return 0.0
    return distancia_km / (duracion_seg / 3600.0)


def esta_cerca_de(lat: float, lon: float, ref_lat: float, ref_lon: float,
                  radio_km: float, km_lat: float = 111.0, km_lon: float = 84.0) -> bool:
    """
    Indica si un punto cae dentro de un radio alrededor de una referencia.

    Usa la escala local de grados en lugar de la distancia de circulo maximo.
    Sobre distancias de pocos kilometros la diferencia es despreciable y el
    calculo resulta considerablemente mas barato, lo que importa al evaluarlo
    sobre millones de filas.
    """
    dx = (lat - ref_lat) * km_lat
    dy = (lon - ref_lon) * km_lon
    return (dx * dx + dy * dy) <= radio_km * radio_km


def toca_aeropuerto(lat_o: float, lon_o: float, lat_d: float, lon_d: float,
                    aeropuertos: dict, radio_km: float) -> int:
    """
    Bandera de trayecto de aeropuerto: cualquiera de los dos extremos dentro del
    radio de una terminal.

    Se evalúan ambos extremos porque el regimen es simetrico: tanto la llegada a
    la terminal como la salida desde ella recorren autopista y no retícula.
    """
    for ref_lat, ref_lon in aeropuertos.values():
        if esta_cerca_de(lat_o, lon_o, ref_lat, ref_lon, radio_km):
            return 1
        if esta_cerca_de(lat_d, lon_d, ref_lat, ref_lon, radio_km):
            return 1
    return 0
