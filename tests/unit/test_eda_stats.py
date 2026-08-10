"""Tests de las funciones estadisticas del analisis exploratorio."""

import pytest

from nyc_taxi.eda import stats


class TestEtaCuadrado:
    def test_separacion_perfecta_da_uno(self):
        # Sin varianza dentro de cada grupo, toda la varianza es entre grupos.
        assert stats.eta_cuadrado([1, 1, 1, 5, 5, 5], ["a", "a", "a", "b", "b", "b"]) == pytest.approx(1.0)

    def test_grupos_identicos_dan_cero(self):
        assert stats.eta_cuadrado([1, 5, 1, 5], ["a", "a", "b", "b"]) == pytest.approx(0.0)

    def test_siempre_en_rango(self):
        v = [3, 1, 4, 1, 5, 9, 2, 6]
        g = [0, 1, 0, 1, 2, 2, 1, 0]
        assert 0.0 <= stats.eta_cuadrado(v, g) <= 1.0

    def test_captura_relacion_ciclica_que_pearson_pierde(self):
        # Patron en forma de U: alto en los extremos, bajo en el centro. Una
        # correlacion lineal daria practicamente cero; el eta cuadrado no.
        valores = [10, 2, 1, 2, 10] * 4
        horas = [0, 6, 12, 18, 23] * 4
        assert stats.eta_cuadrado(valores, horas) > 0.9


class TestFormaDistribucion:
    def test_simetrica_sin_asimetria(self):
        assert stats.asimetria([1, 2, 3, 4, 5]) == pytest.approx(0.0, abs=1e-9)

    def test_cola_derecha_da_asimetria_positiva(self):
        assert stats.asimetria([1, 1, 1, 1, 2, 2, 3, 20]) > 1.0

    def test_curtosis_de_cola_pesada_es_positiva(self):
        assert stats.curtosis_exceso([1, 1, 1, 1, 1, 1, 1, 50]) > 0


class TestCuantiles:
    def test_mediana(self):
        assert stats.cuantil([1, 2, 3, 4, 5], 0.5) == pytest.approx(3.0)

    def test_extremos(self):
        assert stats.cuantil([1, 2, 3], 0.0) == 1.0
        assert stats.cuantil([1, 2, 3], 1.0) == 3.0

    def test_dispersion_reporta_ancho_y_razon(self):
        d = stats.dispersion_por_tramo(list(range(1, 101)))
        assert d["ancho_p90_p10"] == pytest.approx(d["p90"] - d["p10"])
        assert d["razon_p90_p10"] > 1


class TestRegresion:
    def test_relacion_perfecta(self):
        assert stats.r2_regresion_simple([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)

    def test_sin_relacion(self):
        assert stats.r2_regresion_simple([1, 2, 3, 4], [5, 5, 5, 5]) == pytest.approx(0.0)

    def test_residuos_de_ajuste_perfecto_son_cero(self):
        r = stats.residuos_regresion_simple([1, 2, 3, 4], [2, 4, 6, 8])
        assert all(abs(x) < 1e-9 for x in r)

    def test_residuos_remueven_el_efecto_lineal(self):
        x = [1, 2, 3, 4, 5, 6]
        y = [2, 4, 6, 8, 10, 12]
        r = stats.residuos_regresion_simple(x, y)
        assert stats.r2_regresion_simple(x, r) == pytest.approx(0.0, abs=1e-9)
