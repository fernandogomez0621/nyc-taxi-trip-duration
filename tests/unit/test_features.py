"""Tests de las funciones puras de feature engineering."""

import math

import pytest

from nyc_taxi.features import definitions as fd


class TestHaversine:
    def test_distancia_cero_mismo_punto(self):
        assert fd.haversine_km(40.7679, -73.9822, 40.7679, -73.9822) == pytest.approx(0.0)

    def test_un_grado_de_latitud(self):
        # Un grado de latitud son ~111,19 km en cualquier meridiano.
        d = fd.haversine_km(0.0, 0.0, 1.0, 0.0)
        assert d == pytest.approx(111.19, abs=0.1)

    def test_es_simetrica(self):
        ida = fd.haversine_km(40.7679, -73.9822, 40.7656, -73.9646)
        vuelta = fd.haversine_km(40.7656, -73.9646, 40.7679, -73.9822)
        assert ida == pytest.approx(vuelta)

    def test_viaje_real_del_dataset(self):
        # Primera fila de train.csv: recorrido corto dentro de Midtown.
        d = fd.haversine_km(40.767936706542969, -73.982154846191406,
                            40.765602111816406, -73.964630126953125)
        assert 1.0 < d < 2.0


class TestManhattan:
    def test_nunca_menor_que_haversine(self):
        args = (40.7679, -73.9822, 40.7061, -74.0087)
        assert fd.manhattan_km(*args) >= fd.haversine_km(*args)

    def test_igual_a_haversine_en_trayecto_puramente_norte_sur(self):
        # Sin componente este-oeste, la cuadrícula coincide con la línea recta.
        args = (40.70, -74.00, 40.80, -74.00)
        assert fd.manhattan_km(*args) == pytest.approx(fd.haversine_km(*args), abs=1e-6)


class TestBearing:
    def test_norte(self):
        assert fd.bearing_deg(40.70, -74.00, 40.80, -74.00) == pytest.approx(0.0, abs=0.1)

    def test_este(self):
        assert fd.bearing_deg(40.70, -74.00, 40.70, -73.90) == pytest.approx(90.0, abs=0.1)

    def test_siempre_en_rango(self):
        for lat, lon in [(40.8, -74.1), (40.6, -73.8), (40.75, -73.99)]:
            b = fd.bearing_deg(40.70, -74.00, lat, lon)
            assert 0.0 <= b < 360.0


class TestFinDeSemana:
    @pytest.mark.parametrize("dia,esperado", [(1, 1), (7, 1), (2, 0), (4, 0), (6, 0)])
    def test_convencion_spark(self, dia, esperado):
        # Convención de F.dayofweek: 1 = domingo, 7 = sábado.
        assert fd.es_fin_de_semana(dia) == esperado


class TestVelocidad:
    def test_calculo_basico(self):
        # 10 km en media hora son 20 km/h.
        assert fd.velocidad_kmh(10.0, 1800) == pytest.approx(20.0)

    def test_duracion_cero_no_revienta(self):
        assert fd.velocidad_kmh(5.0, 0) == 0.0


class TestAeropuerto:
    JFK = (40.6413, -73.7781)
    LGA = (40.7769, -73.8740)
    AEROPUERTOS = {"JFK": JFK, "LGA": LGA}
    RADIO = 2.0

    def test_el_propio_punto_esta_dentro(self):
        assert fd.esta_cerca_de(*self.JFK, *self.JFK, self.RADIO)

    def test_midtown_esta_fuera_de_ambos(self):
        assert not fd.esta_cerca_de(40.7679, -73.9822, *self.JFK, self.RADIO)
        assert not fd.esta_cerca_de(40.7679, -73.9822, *self.LGA, self.RADIO)

    def test_recogida_en_terminal_marca_el_viaje(self):
        assert fd.toca_aeropuerto(*self.JFK, 40.7679, -73.9822,
                                  self.AEROPUERTOS, self.RADIO) == 1

    def test_destino_en_terminal_marca_el_viaje(self):
        # El regimen es simetrico: importa tocar la terminal, no en que sentido.
        assert fd.toca_aeropuerto(40.7679, -73.9822, *self.LGA,
                                  self.AEROPUERTOS, self.RADIO) == 1

    def test_trayecto_urbano_no_se_marca(self):
        assert fd.toca_aeropuerto(40.7679, -73.9822, 40.7505, -73.9934,
                                  self.AEROPUERTOS, self.RADIO) == 0
