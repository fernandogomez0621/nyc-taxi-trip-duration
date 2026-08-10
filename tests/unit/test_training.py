"""Tests de la construccion de pipelines y de la validacion temporal."""

import pytest

from nyc_taxi import config
from nyc_taxi.training import pipelines


class TestListasDeVariables:
    def test_cubren_exactamente_las_features_declaradas(self):
        # Si alguien agrega una feature a config y olvida clasificarla como
        # categorica o numerica, quedaria fuera del modelo en silencio.
        declaradas = set(config.FEATURES_MODELO)
        usadas = set(pipelines.CATEGORICAS) | set(pipelines.NUMERICAS)
        assert usadas == declaradas, f"desajuste: {declaradas ^ usadas}"

    def test_ninguna_variable_esta_en_ambas_listas(self):
        assert not (set(pipelines.CATEGORICAS) & set(pipelines.NUMERICAS))

    def test_las_horas_y_zonas_son_categoricas(self):
        # Tratarlas como numericas impondria un orden inexistente.
        for v in ["pickup_hour", "pickup_cluster", "dropoff_cluster"]:
            assert v in pipelines.CATEGORICAS

    def test_las_distancias_son_numericas(self):
        for v in ["distancia_haversine_km", "distancia_manhattan_km"]:
            assert v in pipelines.NUMERICAS

    def test_orden_canonico_es_numericas_y_luego_categoricas(self):
        assert pipelines.COLUMNAS_MODELO == pipelines.NUMERICAS + pipelines.CATEGORICAS
        assert len(pipelines.COLUMNAS_MODELO) == len(config.FEATURES_MODELO)

    def test_indices_categoricas_apuntan_a_las_columnas_correctas(self):
        # Un desajuste aqui asociaria las importancias a variables equivocadas.
        idx = pipelines.indices_categoricas()
        assert [pipelines.COLUMNAS_MODELO[i] for i in idx] == pipelines.CATEGORICAS


class TestVentanasTemporales:
    def test_genera_el_numero_pedido(self):
        v = pipelines.ventanas_temporales("2016-01-01", "2016-06-17", 3)
        assert len(v) == 3

    def test_las_ventanas_avanzan_en_el_tiempo(self):
        v = pipelines.ventanas_temporales("2016-01-01", "2016-06-17", 3)
        for i in range(len(v) - 1):
            assert v[i]["entrena_hasta"] < v[i + 1]["entrena_hasta"]

    def test_validacion_siempre_posterior_al_entrenamiento(self):
        # La condicion que distingue esta validacion del K-fold habitual.
        for v in pipelines.ventanas_temporales("2016-01-01", "2016-06-17", 3):
            assert v["valida_hasta"] > v["entrena_hasta"]

    def test_se_mantienen_dentro_del_rango(self):
        v = pipelines.ventanas_temporales("2016-01-01", "2016-06-17", 3)
        assert v[0]["entrena_hasta"] >= "2016-01-01"
        assert v[-1]["valida_hasta"] <= "2016-06-17"
