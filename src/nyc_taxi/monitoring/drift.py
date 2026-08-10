"""
Detección de data drift (reto bonus).

Responde a la pregunta de qué métricas y alertas se implementarían si el modelo
corriera en producción alimentado por datos nuevos cada semana.

Se vigilan tres cosas distintas, que fallan por razones distintas:
  1. Deriva de covariables : cambia la distribución de las features de entrada.
  2. Deriva del target     : cambia la distribución de las duraciones reales.
  3. Degradación           : el error sube aunque las distribuciones se vean bien.

La primera se detecta sin esperar etiquetas, que es lo valioso: avisa antes de
que existan duraciones reales con las que medir el error.
"""

from __future__ import annotations

import math
from typing import Sequence

# Umbrales convencionales de PSI en la industria de riesgo, de donde viene la
# métrica. Son heurísticos, no una prueba estadística: sirven para priorizar
# revisión, no para concluir por sí solos.
PSI_ESTABLE = 0.10
PSI_ALERTA = 0.25


def _histograma(valores: Sequence[float], cortes: Sequence[float]) -> list[float]:
    total = len(valores)
    if total == 0:
        return [0.0] * (len(cortes) - 1)

    conteos = [0] * (len(cortes) - 1)
    for v in valores:
        for i in range(len(cortes) - 1):
            limite_sup = cortes[i + 1]
            if v < limite_sup or (i == len(cortes) - 2 and v <= limite_sup):
                conteos[i] += 1
                break
    return [c / total for c in conteos]


def cortes_por_cuantiles(referencia: Sequence[float], n_bins: int = 10) -> list[float]:
    """
    Define los bordes de los bins sobre la muestra de referencia.

    Por cuantiles y no por ancho fijo: con distribuciones sesgadas como la de
    duración, los bins de ancho fijo dejan casi todo en el primero y el PSI deja
    de discriminar.
    """
    ordenados = sorted(referencia)
    n = len(ordenados)
    cortes = [ordenados[0]]
    for i in range(1, n_bins):
        cortes.append(ordenados[int(n * i / n_bins)])
    cortes.append(ordenados[-1])
    return sorted(set(cortes))


def psi(referencia: Sequence[float], actual: Sequence[float], n_bins: int = 10) -> float:
    """
    Population Stability Index entre la muestra de referencia y la nueva.

    Mide cuánto se movió la distribución. Cerca de cero, nada cambió.
    """
    cortes = cortes_por_cuantiles(referencia, n_bins)
    p_ref = _histograma(referencia, cortes)
    p_act = _histograma(actual, cortes)

    epsilon = 1e-6
    total = 0.0
    for pr, pa in zip(p_ref, p_act):
        pr = max(pr, epsilon)
        pa = max(pa, epsilon)
        total += (pa - pr) * math.log(pa / pr)
    return total


def clasificar_psi(valor: float) -> str:
    if valor < PSI_ESTABLE:
        return "estable"
    if valor < PSI_ALERTA:
        return "vigilar"
    return "alerta"


def evaluar_lote(
    referencia: dict[str, Sequence[float]],
    actual: dict[str, Sequence[float]],
    n_bins: int = 10,
) -> dict:
    """
    Evalúa la deriva de cada feature de un lote nuevo contra la referencia.

    Devuelve el PSI por variable, su clasificación, y si el lote en conjunto
    debería disparar una alerta.
    """
    resultados = {}
    for feature, valores_ref in referencia.items():
        if feature not in actual:
            continue
        valor = psi(valores_ref, actual[feature], n_bins)
        resultados[feature] = {"psi": round(valor, 4), "estado": clasificar_psi(valor)}

    hay_alerta = any(r["estado"] == "alerta" for r in resultados.values())
    return {
        "features": resultados,
        "alerta": hay_alerta,
        "n_features_en_alerta": sum(1 for r in resultados.values() if r["estado"] == "alerta"),
    }
