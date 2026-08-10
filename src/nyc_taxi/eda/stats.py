"""
Estadistica descriptiva para el analisis exploratorio.

Funciones puras sobre secuencias numericas, sin dependencia de Spark ni de
pandas, de modo que puedan probarse de forma automatica.

Nota metodologica que atraviesa todo el modulo: con 1,45 millones de
observaciones las pruebas de significancia clasicas pierden utilidad. A ese
tamano muestral cualquier diferencia, por trivial que sea, resulta
estadisticamente significativa. Por eso se trabaja con **tamanos de efecto**
—que cuantifican la magnitud— y no con valores p.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence


def eta_cuadrado(valores: Sequence[float], grupos: Sequence) -> float:
    """
    Razon de correlacion: proporcion de la varianza de `valores` que explica la
    pertenencia a cada grupo de `grupos`.

    Se usa en lugar del coeficiente de Pearson para variables categoricas y
    ciclicas. Pearson mide asociacion **lineal**, de modo que sobre una variable
    como la hora del dia —donde las 23 y las 0 son contiguas— arroja un valor
    cercano a cero aunque el efecto sea fuerte. El eta cuadrado no supone
    ninguna forma funcional.

    Rango de 0 a 1.
    """
    n = len(valores)
    if n < 2:
        return 0.0

    media = sum(valores) / n
    suma_grupo = defaultdict(float)
    conteo_grupo = defaultdict(int)
    for v, g in zip(valores, grupos):
        suma_grupo[g] += v
        conteo_grupo[g] += 1

    entre = sum(
        conteo_grupo[g] * (suma_grupo[g] / conteo_grupo[g] - media) ** 2 for g in conteo_grupo
    )
    total = sum((v - media) ** 2 for v in valores)
    return entre / total if total else 0.0


def asimetria(valores: Sequence[float]) -> float:
    """
    Coeficiente de asimetria de Fisher.

    Un valor positivo alto indica cola derecha pesada, situacion en la que el
    error cuadratico queda dominado por las observaciones extremas.
    """
    n = len(valores)
    media = sum(valores) / n
    m2 = sum((v - media) ** 2 for v in valores) / n
    m3 = sum((v - media) ** 3 for v in valores) / n
    return m3 / m2 ** 1.5 if m2 else 0.0


def curtosis_exceso(valores: Sequence[float]) -> float:
    """Curtosis en exceso sobre la normal. Cuantifica el peso de las colas."""
    n = len(valores)
    media = sum(valores) / n
    m2 = sum((v - media) ** 2 for v in valores) / n
    m4 = sum((v - media) ** 4 for v in valores) / n
    return m4 / m2 ** 2 - 3 if m2 else 0.0


def cuantil(valores: Sequence[float], q: float) -> float:
    """Cuantil por interpolacion lineal."""
    ordenados = sorted(valores)
    if not ordenados:
        return 0.0
    pos = q * (len(ordenados) - 1)
    inf = int(math.floor(pos))
    sup = min(inf + 1, len(ordenados) - 1)
    return ordenados[inf] + (ordenados[sup] - ordenados[inf]) * (pos - inf)


def dispersion_por_tramo(valores: Sequence[float]) -> dict:
    """
    Resume la dispersion de un tramo mediante cuantiles.

    Se reporta el ancho absoluto entre los percentiles 10 y 90 y tambien su
    razon, porque miden cosas distintas: el ancho indica cuanto margen habria
    que comunicar al usuario, y la razon indica si el error **relativo** crece o
    se mantiene.
    """
    p10, p50, p90 = cuantil(valores, 0.1), cuantil(valores, 0.5), cuantil(valores, 0.9)
    return {
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "ancho_p90_p10": p90 - p10,
        "razon_p90_p10": p90 / p10 if p10 else 0.0,
    }


def r2_regresion_simple(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Coeficiente de determinacion de una regresion lineal simple de `y` sobre `x`.

    Se emplea para cuantificar cuanta varianza explica un unico predictor antes
    de evaluar el aporte incremental de los demas.
    """
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return (sxy ** 2) / (sxx * syy) if sxx and syy else 0.0


def residuos_regresion_simple(x: Sequence[float], y: Sequence[float]) -> list[float]:
    """
    Residuos de `y` tras remover el efecto lineal de `x`.

    Permite calcular efectos **parciales**: el eta cuadrado sobre estos residuos
    mide cuanto explica un factor una vez descontado el predictor dominante, lo
    que evita atribuirle a un factor el efecto de otro con el que esta
    correlacionado.
    """
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    pendiente = sxy / sxx if sxx else 0.0
    intercepto = my - pendiente * mx
    return [b - (intercepto + pendiente * a) for a, b in zip(x, y)]
