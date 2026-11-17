"""
BatteryGuard AI - Pipeline de preprocesamiento (Entrega 3)

Pasos (en este orden, para no filtrar informacion del test al train):
  1. Carga el CSV crudo desde data/raw/ y valida columnas y target.
  2. Elimina duplicados y anonimiza identificadores (vehicle_id, battery_serial).
  3. Excluye las variables con fuga de datos (lista aprobada en la Entrega 2, riesgo R2).
  4. Particion estratificada 75 % train / 25 % test con random_state = 42.
  5. Ajusta la imputacion SOLO con train (mediana en numericas, moda en categoricas)
     y la aplica a train y test -> 0 vacios en predictoras (CA-01).
  6. Guarda data/processed/train.csv, data/processed/test.csv y models/imputer.joblib.

Uso:
    python src/data/preprocess.py --data data/raw/ev_battery_failure_prediction_Dataset.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

_log = logging.getLogger("batteryguard")


def _say(msg: str) -> None:
    """Escribe en el log de entrenamiento si existe; si no, en consola."""
    (_log.info if _log.handlers else print)(msg)


# Raiz del repositorio (src/data/preprocess.py -> ../../)
ROOT = Path(__file__).resolve().parents[2]

RANDOM_STATE = 42
TEST_SIZE = 0.25

TARGET_CLF = "battery_failure"
TARGET_REG = "predicted_remaining_life_cycles"
ID_COLS = ["vehicle_id", "battery_serial"]

# Variables con fuga de datos (Entrega 2, seccion 4C): puntajes compuestos que ya
# resumen el estado de falla, o son transformaciones directas de otra variable.
LEAKAGE_COLS = [
    "thermal_runaway_risk",   # puntaje compuesto; thermal_runaway_risk + thermal_health_score ~ 100
    "thermal_health_score",
    "aging_score",
    "battery_stress_index",
    "capacity_loss_percent",  # = 100 - battery_health_percent (redundante exacta)
]

CATEGORICAL_COLS = [
    "vehicle_brand", "vehicle_model", "vehicle_type", "battery_manufacturer",
    "battery_chemistry", "drive_type", "fleet_or_private", "terrain_type",
]


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(ID_COLS + [TARGET_CLF, TARGET_REG] + CATEGORICAL_COLS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {sorted(missing)}")
    if df[TARGET_CLF].isna().any():
        raise ValueError("El target battery_failure tiene vacios.")
    if not set(df[TARGET_CLF].unique()).issubset({0, 1}):
        raise ValueError("El target battery_failure debe contener solo valores binarios 0 y 1.")
    return df


def anonymize(value: str) -> str:
    """ID anonimo y estable (mismo valor -> mismo codigo) para mostrar en el tablero."""
    return "BAT-" + hashlib.sha256(str(value).encode()).hexdigest()[:8].upper()


def feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = set(ID_COLS + LEAKAGE_COLS + [TARGET_CLF, TARGET_REG, "battery_id_anon"])
    cat = [c for c in CATEGORICAL_COLS if c in df.columns]
    num = [c for c in df.columns if c not in excluded and c not in cat]
    return num, cat


def build_imputer(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    imp = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", SimpleImputer(strategy="most_frequent"), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    imp.set_output(transform="pandas")
    return imp


def run(data_path: Path, out_dir: Path, models_dir: Path) -> dict:
    _say(f"[INFO] Cargando dataset crudo: {data_path.relative_to(ROOT) if data_path.is_relative_to(ROOT) else data_path}")
    df = load_raw(data_path)
    n0 = len(df)
    df = df.drop_duplicates()
    _say(f"[INFO] {n0:,} registros, {df.shape[1]} columnas | duplicados eliminados: {n0 - len(df)}")
    _say(f"[INFO] Fallas: {int(df[TARGET_CLF].sum()):,} ({df[TARGET_CLF].mean():.2%})")
    _say(f"[INFO] Vacios por columna: min {df.isna().mean().min():.2%} - max {df.isna().mean().max():.2%}")

    df["battery_id_anon"] = df["battery_serial"].map(anonymize)
    num_cols, cat_cols = feature_lists(df)
    _say(f"[INFO] Variables excluidas por fuga de datos: {LEAKAGE_COLS}")
    _say(f"[INFO] Predictoras: {len(num_cols)} numericas + {len(cat_cols)} categoricas = {len(num_cols) + len(cat_cols)}")

    train, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET_CLF], random_state=RANDOM_STATE
    )
    _say(f"[INFO] Particion estratificada (random_state={RANDOM_STATE}): "
          f"Train {len(train):,} ({int(train[TARGET_CLF].sum())} fallas) | "
          f"Test {len(test):,} ({int(test[TARGET_CLF].sum())} fallas)")

    def _normalize_inputs(part: pd.DataFrame) -> pd.DataFrame:
        sub = part[num_cols + cat_cols].copy()
        for c in cat_cols:
            # Evita pd.NA o tipos mixtos en pandas 3.x al pasar por SimpleImputer
            sub[c] = sub[c].where(sub[c].notna(), np.nan)
            mask = sub[c].notna()
            if mask.any():
                sub.loc[mask, c] = sub.loc[mask, c].astype(str)
        return sub

    imputer = build_imputer(num_cols, cat_cols)
    imputer.fit(_normalize_inputs(train))  # SOLO train

    def transform(part: pd.DataFrame) -> pd.DataFrame:
        X = imputer.transform(_normalize_inputs(part))
        X.index = part.index
        meta = part[["battery_id_anon", TARGET_CLF, TARGET_REG]]
        return pd.concat([meta, X], axis=1)

    train_p, test_p = transform(train), transform(test)
    nulls = int(train_p[num_cols + cat_cols].isna().sum().sum() + test_p[num_cols + cat_cols].isna().sum().sum())
    assert nulls == 0, "Quedaron vacios en predictoras"
    _say(f"[INFO] Imputacion ajustada con train (mediana / moda). Vacios restantes en predictoras: {nulls}")

    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    train_p.to_csv(out_dir / "train.csv", index=False)
    test_p.to_csv(out_dir / "test.csv", index=False)
    joblib.dump(imputer, models_dir / "imputer.joblib")

    # Rangos validos por variable (para RNF-02: validar entradas en predecir()).
    # Se calculan SOLO con train (antes se usaba df completo, filtrando min/max/categorias del test).
    ranges = {c: [float(train[c].min()), float(train[c].max())] for c in num_cols}
    categories = {c: sorted(train[c].dropna().astype(str).unique().tolist()) for c in cat_cols}
    schema = {
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "leakage_excluded": LEAKAGE_COLS,
        "numeric_ranges": ranges,
        "categories": categories,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
    (models_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False))
    _say(f"[SUCCESS] data/processed/train.csv y test.csv guardados")
    _say(f"[SUCCESS] Imputador guardado en: models/imputer.joblib | esquema: models/feature_schema.json")
    return schema


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocesamiento BatteryGuard AI")
    p.add_argument("--data", default="data/raw/ev_battery_failure_prediction_Dataset.csv")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--models", default="models")
    a = p.parse_args()
    run(ROOT / a.data, ROOT / a.out, ROOT / a.models)


if __name__ == "__main__":
    main()
