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
