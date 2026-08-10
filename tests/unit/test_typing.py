"""Tests del tipado de la capa Silver."""

import pytest

from nyc_taxi.data_prep import typing as tp


COLS_TRAIN = [
    "id", "vendor_id", "pickup_datetime", "dropoff_datetime", "passenger_count",
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "store_and_fwd_flag", "trip_duration", "_row_hash", "_ingested_at",
]
COLS_SCORING = [c for c in COLS_TRAIN if c not in ("dropoff_datetime", "trip_duration")]


class TestMapaDeTipos:
    def test_coordenadas_son_double_no_float(self):
        # Un float de 32 bits perdería precisión suficiente para desplazar el
        # punto decenas de metros y contaminar la distancia calculada.
        for c in ["pickup_latitude", "pickup_longitude",
                  "dropoff_latitude", "dropoff_longitude"]:
            assert tp.TIPOS[c] == "double"

    def test_identificadores_siguen_siendo_texto(self):
        assert tp.TIPOS["id"] == "string"
        assert tp.TIPOS["vendor_id"] == "string"

    def test_duracion_es_entero(self):
        assert tp.TIPOS["trip_duration"] == "int"


class TestColumnasATipar:
    def test_train_incluye_las_dos_columnas_del_target(self):
        cols = tp.columnas_a_tipar(COLS_TRAIN)
        assert "dropoff_datetime" in cols
        assert "trip_duration" in cols

    def test_scoring_las_omite_sin_reventar(self):
        # La fuente retira ambas de test.csv; la misma función debe servir para
        # los dos conjuntos sin lógica condicional en el notebook.
        cols = tp.columnas_a_tipar(COLS_SCORING)
        assert "dropoff_datetime" not in cols
        assert "trip_duration" not in cols
        assert "pickup_datetime" in cols

    def test_ignora_columnas_de_auditoria(self):
        cols = tp.columnas_a_tipar(COLS_TRAIN)
        assert not any(c.startswith("_") for c in cols)


class TestFlag:
    @pytest.mark.parametrize("valor,esperado", [
        ("Y", 1), ("N", 0), ("y", 1), ("n", 0), (" Y ", 1),
    ])
    def test_mapeo_valido(self, valor, esperado):
        assert tp.mapear_flag(valor) == esperado

    def test_nulo_se_propaga(self):
        assert tp.mapear_flag(None) is None

    def test_valor_fuera_de_dominio_no_se_asume_cero(self):
        # Devolver 0 ante un valor inesperado escondería un problema de calidad.
        assert tp.mapear_flag("X") is None
