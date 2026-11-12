"""
BatteryGuard AI - Funcion de inferencia desacoplada (contrato con el dashboard, RNF-03)

El dashboard (Entrega 4) solo hace:
    from src.models.predict import predecir, predecir_lote
    resultado = predecir({"battery_health_percent": 61.2, "internal_resistance": 0.86, ...})

- predecir(entrada: dict) -> dict      una bateria (formulario), con explicacion SHAP (top 5)
- predecir_lote(df: DataFrame) -> df   toda la flota (tabla priorizada), sin SHAP para ser rapido

Las variables que no vengan en la entrada se completan con la mediana / moda del train (RF-01).
Las entradas invalidas devuelven status "error" con mensajes claros en vez de romper (RNF-02).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

# 1. Carga de artefactos serializados (una sola vez en memoria al importar el modulo)
MODELO_FALLA = joblib.load(MODELS_DIR / "modelo_falla.joblib")
MODELO_VIDA = joblib.load(MODELS_DIR / "modelo_vida_util.joblib")
IMPUTADOR = joblib.load(MODELS_DIR / "imputer.joblib")
CONFIG = json.loads((MODELS_DIR / "model_config.json").read_text(encoding="utf-8"))
SCHEMA = json.loads((MODELS_DIR / "feature_schema.json").read_text(encoding="utf-8"))

FEATURES: list[str] = CONFIG["features"]
NUM: list[str] = CONFIG["numeric_features"]
CAT: list[str] = CONFIG["categorical_features"]
UMBRAL_ALTO: float = CONFIG["umbral_alto"]
UMBRAL_MEDIO: float = CONFIG["umbral_medio"]

# Limites fisicos duros (fuera de ellos la entrada es imposible -> error)
LIMITES_DUROS = {
    "battery_health_percent": (0, 100), "state_of_health": (0, 100), "state_of_charge": (0, 100),
    "depth_of_discharge": (0, 100), "charge_efficiency": (0, 100), "discharge_efficiency": (0, 100),
    "fast_charge_ratio": (0, 1), "slow_charge_ratio": (0, 1), "overnight_charging_ratio": (0, 1),
    "home_charging_ratio": (0, 1), "highway_driving_ratio": (0, 1), "city_driving_ratio": (0, 1),
}
NO_NEGATIVAS = [c for c in NUM if any(k in c for k in ("count", "faults", "events", "cycle", "km",
                                                          "years", "interruptions", "days", "updates",
                                                          "resistance", "capacity"))]

_explainer = None  # SHAP se inicializa la primera vez que se necesita


def _get_explainer():
    global _explainer
    if _explainer is None:
        import shap
        _explainer = shap.TreeExplainer(MODELO_FALLA.named_steps["model"])
    return _explainer


def nivel_riesgo(prob: float, umbral_alto: float | None = None) -> str:
    """ALTO / MEDIO / BAJO. El dashboard puede pasar su propio umbral (slider, RF-04)."""
    alto = UMBRAL_ALTO if umbral_alto is None else umbral_alto
    medio = min(UMBRAL_MEDIO, alto)
    if prob >= alto:
        return "ALTO"
    if prob >= medio:
        return "MEDIO"
    return "BAJO"


def validar_entrada(entrada: dict) -> tuple[dict, list[str], list[str]]:
    """Devuelve (entrada_limpia, errores, advertencias)."""
    errores, advertencias, limpia = [], [], {}
    if not isinstance(entrada, dict) or not entrada:
        return {}, ["La entrada debe ser un diccionario con al menos una variable."], []
    desconocidas = [k for k in entrada if k not in FEATURES]
    if desconocidas:
        advertencias.append(f"Se ignoraron variables que el modelo no usa: {', '.join(desconocidas)}")
    for col in NUM:
        if col not in entrada or entrada[col] is None or entrada[col] == "":
            continue
        try:
            v = float(entrada[col])
        except (TypeError, ValueError):
            errores.append(f"'{col}' debe ser numerico (recibido: {entrada[col]!r}).")
            continue
        if np.isnan(v) or np.isinf(v):
            continue
        if col in LIMITES_DUROS:
            lo, hi = LIMITES_DUROS[col]
            if not lo <= v <= hi:
                errores.append(f"'{col}' = {v} esta fuera del rango valido [{lo}, {hi}].")
                continue
        if col in NO_NEGATIVAS and v < 0:
            errores.append(f"'{col}' no puede ser negativo (recibido: {v}).")
            continue
        lo, hi = SCHEMA["numeric_ranges"][col]
        if not lo <= v <= hi:
            advertencias.append(f"'{col}' = {v} esta fuera del rango visto en el historial [{lo:g}, {hi:g}]; "
                                f"la prediccion puede ser menos confiable.")
        limpia[col] = v
    for col in CAT:
        if col not in entrada or entrada[col] in (None, ""):
            continue
        val = str(entrada[col])
        if val not in SCHEMA["categories"][col]:
            advertencias.append(f"'{col}' = '{val}' no aparece en el historial; se trata como categoria desconocida.")
        limpia[col] = val
    faltantes = [c for c in FEATURES if c not in limpia]
    if faltantes and not errores:
        advertencias.append(f"{len(faltantes)} de {len(FEATURES)} variables no se capturaron y se "
                            f"completaron con la mediana / moda del historial.")
    return limpia, errores, advertencias


def _preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Mismo preprocesamiento que en entrenamiento: columnas en orden + imputacion del train."""
    df = df.reindex(columns=FEATURES)
    df[NUM] = df[NUM].apply(pd.to_numeric, errors="coerce")
    X = IMPUTADOR.transform(df)
    return X[FEATURES]


def _top_factores(X: pd.DataFrame, k: int = 5) -> list[dict]:
    """Top-k variables por |SHAP|, agregando las columnas one-hot a su variable original."""
    prep = MODELO_FALLA.named_steps["prep"]
    Xt = prep.transform(X)
    sv = np.asarray(_get_explainer().shap_values(Xt))[0]
    nombres = prep.get_feature_names_out()
    contrib: dict[str, float] = {}
    for nombre, val in zip(nombres, sv):
        base = next((c for c in CAT if nombre.startswith(c + "_")), nombre)
        contrib[base] = contrib.get(base, 0.0) + float(val)
    top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    fila = X.iloc[0]
    return [{"variable": v, "valor": (fila[v].item() if hasattr(fila[v], "item") else fila[v]),
             "contribucion_shap": round(c, 4),
             "efecto": "aumenta el riesgo" if c > 0 else "reduce el riesgo"} for v, c in top]


def predecir(entrada: dict, umbral_alto: float | None = None, explicar: bool = True) -> dict:
    """
    Recibe las variables de una bateria como diccionario plano (formulario del dashboard)
    y retorna la inferencia estructurada lista para mostrar en pantalla.

    Parametros:
        entrada (dict): variables de la bateria (nombres de columna del CSV). Las faltantes se imputan.
        umbral_alto (float, opcional): umbral del slider; por defecto el calibrado en entrenamiento.
        explicar (bool): si True calcula los 5 factores SHAP.
    Retorna:
        dict con prediccion (probabilidad 0-1), clase_etiqueta, confianza_o_score,
        vida_util_restante_ciclos, top_factores, tiempo_ms, status y mensajes.
    """
    t0 = time.perf_counter()
    limpia, errores, advertencias = validar_entrada(entrada)
    if errores:
        return {"status": "error", "errores": errores, "advertencias": advertencias,
                "tiempo_ms": round((time.perf_counter() - t0) * 1000, 1)}
    try:
        # 2. DataFrame con el orden y nombres de columnas del entrenamiento + imputacion
        X = _preparar(pd.DataFrame([limpia]))
        # 3. Inferencia de ambos modelos
        prob = float(MODELO_FALLA.predict_proba(X)[0, 1])
        vida = float(MODELO_VIDA.predict(X)[0])
        nivel = nivel_riesgo(prob, umbral_alto)
        factores = _top_factores(X) if explicar else []
    except Exception as exc:  # el dashboard nunca debe romperse (RNF-02)
        return {"status": "error", "errores": [f"No se pudo calcular el riesgo: {exc}"],
                "advertencias": advertencias, "tiempo_ms": round((time.perf_counter() - t0) * 1000, 1)}
    # 4. Salida estructurada (contrato de la Entrega 2)
    return {
        "prediccion": round(prob, 4),
        "clase_predicha": int(nivel == "ALTO"),
        "clase_etiqueta": f"Riesgo {nivel}",
        "nivel_riesgo": nivel,
        "confianza_o_score": round(prob, 4),
        "probabilidad_falla_pct": round(prob * 100, 2),
        "vida_util_restante_ciclos": int(round(max(vida, 0))),
        "top_factores": [f["variable"] for f in factores],
        "top_factores_detalle": factores,
        "umbral_usado": round(UMBRAL_ALTO if umbral_alto is None else umbral_alto, 4),
        "tiempo_ms": round((time.perf_counter() - t0) * 1000, 1),
        "advertencias": advertencias,
        "status": "success",
    }


def predecir_lote(df: pd.DataFrame, umbral_alto: float | None = None) -> pd.DataFrame:
    """Puntua toda la flota (p. ej. las 20,000 baterias) para la tabla priorizada. Meta: <= 10 s."""
    X = _preparar(df.copy())
    prob = MODELO_FALLA.predict_proba(X)[:, 1]
    vida = np.clip(MODELO_VIDA.predict(X), 0, None).round().astype(int)
    out = df.copy()
    out["probabilidad_falla"] = prob.round(4)
    out["nivel_riesgo"] = [nivel_riesgo(p, umbral_alto) for p in prob]
    out["vida_util_restante_ciclos"] = vida
    return out.sort_values("probabilidad_falla", ascending=False)


if __name__ == "__main__":
    # Demo rapida: python src/models/predict.py
    demo = {"vehicle_brand": "Tata", "vehicle_type": "Truck", "battery_chemistry": "NCA",
            "fleet_or_private": "Fleet", "cycle_count": 1425, "battery_health_percent": 61.24,
            "internal_resistance": 0.861, "cell_temperature_max": 27.26, "BMS_warning_count": 1,
            "previous_faults": 1}
    print(json.dumps(predecir(demo), indent=2, ensure_ascii=False))
