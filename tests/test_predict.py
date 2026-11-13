"""Pruebas unitarias del contrato predecir() (ejecutar: python -m pytest tests/ -v)."""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.predict import predecir, predecir_lote  # noqa: E402

BATERIA_RIESGO = {  # fila EV109237 del test set (falla real = 1)
    "vehicle_brand": "Tesla", "battery_health_percent": 50.03, "state_of_health": 54.31,
    "cell_temperature_max": 56.58, "internal_resistance": 0.956, "BMS_warning_count": 2,
}
CLAVES = {"prediccion", "clase_etiqueta", "confianza_o_score", "vida_util_restante_ciclos",
          "top_factores", "tiempo_ms", "status"}


def test_contrato_de_salida():
    r = predecir(BATERIA_RIESGO)
    assert r["status"] == "success"
    assert CLAVES <= r.keys()
    assert 0 <= r["prediccion"] <= 1
    assert r["clase_etiqueta"] in {"Riesgo ALTO", "Riesgo MEDIO", "Riesgo BAJO"}
    assert len(r["top_factores"]) == 5
    assert r["vida_util_restante_ciclos"] >= 0


def test_bateria_degradada_es_riesgo_alto():
    assert predecir(BATERIA_RIESGO)["nivel_riesgo"] == "ALTO"


def test_entrada_invalida_no_rompe():
    r = predecir({"battery_health_percent": 120, "cycle_count": "mucho"})
    assert r["status"] == "error"
    assert len(r["errores"]) == 2


def test_entrada_vacia():
    assert predecir({})["status"] == "error"


def test_latencia_individual_menor_a_3s():
    predecir(BATERIA_RIESGO)  # calienta SHAP
    t0 = time.perf_counter()
    predecir(BATERIA_RIESGO)
    assert time.perf_counter() - t0 < 3


def test_lote_20000_menor_a_10s():
    df = pd.read_csv(ROOT / "data/raw/ev_battery_failure_prediction_Dataset.csv")
    t0 = time.perf_counter()
    out = predecir_lote(df)
    assert time.perf_counter() - t0 < 10
    assert len(out) == 20000
    assert out["probabilidad_falla"].is_monotonic_decreasing


def test_rechaza_booleanos_infinitos_y_limites_fisicos():
    r_bool = predecir({"battery_health_percent": True})
    assert r_bool["status"] == "error"

    r_inf = predecir({"cycle_count": float("inf")})
    assert r_inf["status"] == "error"

    r_phys = predecir({"average_speed": -10, "humidity": 120, "daily_distance": -5})
    assert r_phys["status"] == "error"
    assert len(r_phys["errores"]) == 3


def test_rechaza_solo_claves_desconocidas_y_umbral_invalido():
    assert predecir({"variable_inexistente": 123})["status"] == "error"
    assert predecir({"battery_health_percent": float("nan")})["status"] == "error"
    assert predecir(BATERIA_RIESGO, umbral_alto=1.5)["status"] == "error"
    assert predecir(BATERIA_RIESGO, umbral_alto=True)["status"] == "error"


def test_lote_vacio_y_parcial_con_na_categorico():
    vacio = predecir_lote(pd.DataFrame())
    assert vacio.empty
    assert {"probabilidad_falla", "nivel_riesgo", "vida_util_restante_ciclos"} <= set(vacio.columns)

    parcial = pd.DataFrame({
        "vehicle_brand": ["Tesla", pd.NA, 123],
        "battery_health_percent": [50.0, "no_valido", 92.0],
    })
    out = predecir_lote(parcial)
    assert len(out) == 3
    assert out["probabilidad_falla"].between(0, 1).all()
