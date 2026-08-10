"""
Localización de la raíz del proyecto.

Los notebooks pueden vivir a cualquier profundidad dentro del repo (o moverse de
sitio a mitad del trabajo), así que en vez de fijar un número de saltos con
`../../src` se busca hacia arriba la carpeta que contiene `src/`. Es inmune a la
estructura de subcarpetas y falla con un mensaje explícito en vez de un
ModuleNotFoundError genérico.
"""

from __future__ import annotations

import os
import sys

MARCADOR = "src"
MAX_NIVELES = 10


def raiz_repo(desde: str | None = None) -> str:
    """Devuelve la primera carpeta hacia arriba que contenga `src/`."""
    ruta = os.path.abspath(desde or os.getcwd())
    for _ in range(MAX_NIVELES):
        if os.path.isdir(os.path.join(ruta, MARCADOR)):
            return ruta
        padre = os.path.dirname(ruta)
        if padre == ruta:  # llegamos a la raíz del sistema de archivos
            break
        ruta = padre
    raise RuntimeError(
        f"No se encontró la raíz del repo (carpeta con '{MARCADOR}/') "
        f"partiendo de {os.path.abspath(desde or os.getcwd())}. "
        "Verificar que el notebook esté dentro del repositorio."
    )


def preparar_path(verbose: bool = True) -> str:
    """Agrega `<raiz>/src` al sys.path. Idempotente."""
    ruta_src = os.path.join(raiz_repo(), MARCADOR)
    if ruta_src not in sys.path:
        sys.path.insert(0, ruta_src)
    if verbose:
        print(f"src en path: {ruta_src}")
    return ruta_src
