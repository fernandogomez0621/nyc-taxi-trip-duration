"""Tests de las reglas de calidad, el split y las métricas."""

import pytest

from nyc_taxi import config
from nyc_taxi.data_prep import splits, validation
from nyc_taxi.evaluation import metrics
from nyc_taxi.monitoring import drift


class TestValidaciones:
    @pytest.mark.parametrize("seg,ok", [(0, False), (1, True), (455, True),
                                        (21600, True), (21601, False), (86400, False)])
    def test_duracion(self, seg, ok):
        assert validation.es_duracion_valida(seg) is ok

    @pytest.mark.parametrize("n,ok", [(0, False), (1, True), (6, True), (7, False)])
    def test_pasajeros(self, n, ok):
        assert validation.es_pasajeros_valido(n) is ok

    def test_coordenada_de_manhattan_es_valida(self):
        assert validation.esta_en_nyc(40.7679, -73.9822)

    def test_coordenada_en_el_atlantico_no_lo_es(self):
        # (0, 0) es el fallback típico de un GPS sin señal.
        assert not validation.esta_en_nyc(0.0, 0.0)

    def test_trayecto_requiere_ambos_extremos(self):
        assert not validation.es_trayecto_en_nyc(40.7679, -73.9822, 0.0, 0.0)

    def test_lista_de_reglas_no_esta_vacia(self):
        assert len(validation.NOMBRES_REGLAS) >= 5


class TestSplit:
    def test_antes_del_corte_es_train(self):
        assert splits.asignar_split("2016-01-15 08:30:00") == splits.SPLIT_TRAIN

    def test_desde_el_corte_es_test(self):
        assert splits.asignar_split("2016-06-20 08:30:00") == splits.SPLIT_TEST

    def test_el_dia_del_corte_cae_en_test(self):
        assert splits.asignar_split(config.FECHA_CORTE_SPLIT) == splits.SPLIT_TEST

    def test_corte_dentro_del_rango_del_dataset(self):
        assert "2016-01-01" < config.FECHA_CORTE_SPLIT < "2016-06-30"


class TestMetricas:
    def test_prediccion_perfecta(self):
        y = [100.0, 200.0, 300.0]
        assert metrics.rmse(y, y) == pytest.approx(0.0)
        assert metrics.mae(y, y) == pytest.approx(0.0)
        assert metrics.r2(y, y) == pytest.approx(1.0)

    def test_mae_conocido(self):
        assert metrics.mae([10.0, 20.0], [12.0, 24.0]) == pytest.approx(3.0)

    def test_traduccion_a_negocio(self):
        r = metrics.traducir_a_negocio(mae_seg=180, mediana_real_seg=660)
        assert r["error_medio_min"] == 3.0
        assert r["viaje_mediano_min"] == 11.0
        assert r["error_relativo_pct"] == pytest.approx(27.3, abs=0.1)

    def test_diagnostico_overfitting_detecta_brecha(self):
        d = metrics.diagnostico_overfitting(metrica_train=100.0, metrica_test=150.0)
        assert d["brecha_relativa_pct"] == pytest.approx(50.0)


class TestDrift:
    def test_sin_deriva_el_psi_es_bajo(self):
        base = [float(i % 100) for i in range(1000)]
        assert drift.psi(base, base) < drift.PSI_ESTABLE

    def test_con_desplazamiento_fuerte_el_psi_dispara(self):
        base = [float(i % 100) for i in range(1000)]
        movido = [v + 200 for v in base]
        assert drift.clasificar_psi(drift.psi(base, movido)) == "alerta"

    def test_evaluar_lote_reporta_por_feature(self):
        base = {"distancia_haversine_km": [float(i % 50) for i in range(500)]}
        nuevo = {"distancia_haversine_km": [float(i % 50) for i in range(500)]}
        r = drift.evaluar_lote(base, nuevo)
        assert r["alerta"] is False
        assert "distancia_haversine_km" in r["features"]


class TestBootstrap:
    def test_encuentra_la_raiz_desde_una_subcarpeta_profunda(self, tmp_path, monkeypatch):
        from nyc_taxi import bootstrap

        raiz = tmp_path / "repo"
        (raiz / "src").mkdir(parents=True)
        profundo = raiz / "notebooks" / "pipeline" / "etl" / "v2"
        profundo.mkdir(parents=True)

        monkeypatch.chdir(profundo)
        assert bootstrap.raiz_repo() == str(raiz)

    def test_falla_con_mensaje_claro_si_no_hay_repo(self, tmp_path, monkeypatch):
        import pytest as _pytest

        from nyc_taxi import bootstrap

        suelto = tmp_path / "suelto"
        suelto.mkdir()
        monkeypatch.chdir(suelto)
        with _pytest.raises(RuntimeError, match="No se encontró la raíz del repo"):
            bootstrap.raiz_repo()
