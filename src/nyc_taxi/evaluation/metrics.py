"""
Métricas de evaluación y su lectura en términos de negocio.

La prueba pide explícitamente explicar qué tan bueno es el modelo en términos de
negocio y no solo estadísticos. `traducir_a_negocio` existe para que esa
traducción sea parte del código y no una frase improvisada al final.
"""

from __future__ import annotations

import math
from typing import Sequence


def rmse(y_real: Sequence[float], y_pred: Sequence[float]) -> float:
    n = len(y_real)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_real, y_pred)) / n)


def mae(y_real: Sequence[float], y_pred: Sequence[float]) -> float:
    n = len(y_real)
    return sum(abs(a - b) for a, b in zip(y_real, y_pred)) / n


def r2(y_real: Sequence[float], y_pred: Sequence[float]) -> float:
    media = sum(y_real) / len(y_real)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_real, y_pred))
    ss_tot = sum((a - media) ** 2 for a in y_real)
    return 1 - ss_res / ss_tot if ss_tot else 0.0


def rmsle(y_real: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Error logarítmico cuadrático medio: la métrica oficial de la competencia.

    Penaliza el error relativo y no el absoluto, lo cual tiene sentido aquí:
    equivocarse dos minutos en un viaje de cinco es grave, en uno de cuarenta
    es irrelevante.
    """
    n = len(y_real)
    return math.sqrt(
        sum((math.log1p(max(b, 0)) - math.log1p(max(a, 0))) ** 2 for a, b in zip(y_real, y_pred)) / n
    )


def traducir_a_negocio(mae_seg: float, mediana_real_seg: float) -> dict:
    """
    Convierte el error del modelo en una frase que un directivo pueda usar.

    Un MAE en segundos no le dice nada a nadie. El error relativo sobre el viaje
    mediano sí: indica si se puede prometer un tiempo puntual al pasajero o si
    hay que comunicar una ventana.
    """
    error_min = mae_seg / 60
    mediana_min = mediana_real_seg / 60
    error_relativo = mae_seg / mediana_real_seg if mediana_real_seg else 0.0

    return {
        "error_medio_min": round(error_min, 2),
        "viaje_mediano_min": round(mediana_min, 2),
        "error_relativo_pct": round(error_relativo * 100, 1),
        "ventana_sugerida_min": (
            round(max(0.0, mediana_min - error_min), 1),
            round(mediana_min + error_min, 1),
        ),
    }


def diagnostico_overfitting(metrica_train: float, metrica_test: float) -> dict:
    """
    Compara el desempeño en entrenamiento y evaluación.

    No basta con reportar que el modelo generaliza: hay que mostrar la evidencia.
    La brecha relativa es esa evidencia.
    """
    brecha = metrica_test - metrica_train
    brecha_rel = brecha / metrica_train if metrica_train else 0.0
    return {
        "train": round(metrica_train, 4),
        "test": round(metrica_test, 4),
        "brecha_absoluta": round(brecha, 4),
        "brecha_relativa_pct": round(brecha_rel * 100, 1),
    }
