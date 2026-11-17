"""
BatteryGuard AI - Entrenamiento y experimentacion (Entrega 3)

Entrena y compara, sobre el MISMO test set de la Entrega 2 (25 %, estratificado, random_state=42):

  Clasificacion (riesgo de falla, OE1):
    Exp 0  Baseline ingenuo  - DummyClassifier (siempre "sin falla")
    Exp 1  Regresion Logistica (StandardScaler + One-Hot, class_weight="balanced")
    Exp 2  Random Forest por defecto (class_weight="balanced")
    Exp 3  Gradient Boosting por defecto (HistGradientBoosting, sin pesos)
    Exp 4  Gradient Boosting ajustado (pesos de clase + regularizacion, GridSearchCV 5-fold)
    Exp 5  Exp 4 + variables con fuga (solo control, NO se usa en produccion)

  Regresion (vida util restante en ciclos, OE2):
    R1  Ridge (baseline lineal)     R2  HistGradientBoostingRegressor

El umbral de decision de cada clasificador se elige SIN mirar el test: con predicciones
out-of-fold (5-fold) en train, el umbral de mayor precision que logra recall >= 0.87
(margen sobre la meta de 0.85).

Uso (desde la raiz del repositorio, un solo comando; si faltan los datos procesados
corre primero src/data/preprocess.py automaticamente):
    python src/models/train.py --data data/processed --output models/ --preprocess
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import joblib

if "MPLCONFIGDIR" not in os.environ:
    _mpl_home = Path.home() / ".matplotlib"
    _mpl_parent = _mpl_home if _mpl_home.exists() else Path.home()
    if not os.access(_mpl_parent, os.W_OK):
        os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score, make_scorer,
                             mean_absolute_error, precision_recall_curve, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.preprocess import LEAKAGE_COLS, RANDOM_STATE  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
TARGET_CLF, TARGET_REG = "battery_failure", "predicted_remaining_life_cycles"
RECALL_TARGET_CV = 0.87      # recall buscado en validacion cruzada (meta en test: >= 0.85)
RECALL_MEDIUM_CV = 0.97      # recall con el que una bateria pasa a riesgo MEDIO
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

log = logging.getLogger("batteryguard")


# --------------------------------------------------------------------------- utilidades
def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(message)s")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def make_encoder(num: list[str], cat: list[str], scale: bool) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler() if scale else "passthrough", num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        verbose_feature_names_out=False,
    )


def threshold_for_recall(y: pd.Series, proba: np.ndarray, recall_min: float) -> float:
    """Umbral con mayor precision entre los que logran recall >= recall_min."""
    p, r, th = precision_recall_curve(y, proba)
    ok = r[:-1] >= recall_min
    if not ok.any():
        return float(th[0])
    return float(th[np.argmax(np.where(ok, p[:-1], -1))])


def clf_metrics(y, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": float((pred == y).mean()),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(proba)) > 1 else 0.5,
        "pr_auc": average_precision_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "flagged": int(pred.sum()),
    }


def diagnose(train_m: dict, test_m: dict) -> str:
    if test_m["recall"] == 0:
        return "Underfitting total (no detecta fallas)"
    gap = train_m["pr_auc"] - test_m["pr_auc"]
    if gap > 0.20:
        return f"Overfitting severo (PR-AUC train {train_m['pr_auc']:.2f} vs test {test_m['pr_auc']:.2f})"
    if gap > 0.10:
        return f"Overfitting moderado (PR-AUC train {train_m['pr_auc']:.2f} vs test {test_m['pr_auc']:.2f})"
    if test_m["recall"] < 0.85 or test_m["precision"] < 0.55:
        return f"Generaliza (brecha {gap:.2f}) pero no cumple la meta"
    return f"Optimo: cumple meta y brecha controlada ({gap:.2f})"


# --------------------------------------------------------------------------- experimentos
def run_classifier(name, model, Xtr, ytr, Xte, yte) -> dict:
    log.info(f"[INFO] {name}: validacion cruzada 5-fold para elegir umbral...")
    oof = cross_val_predict(model, Xtr, ytr, cv=CV, method="predict_proba")[:, 1]
    thr = threshold_for_recall(ytr, oof, RECALL_TARGET_CV) if oof.std() > 0 else 0.5
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    p_te = model.predict_proba(Xte)[:, 1]
    t_inf = (time.perf_counter() - t0) / len(Xte) * 1000
    p_tr = model.predict_proba(Xtr)[:, 1]
    m_tr, m_te = clf_metrics(ytr, p_tr, thr), clf_metrics(yte, p_te, thr)
    res = {
        "experimento": name, "umbral": thr, "train": m_tr, "test": m_te,
        "t_entrenamiento_s": t_train, "t_inferencia_ms_por_bateria": t_inf,
        "diagnostico": diagnose(m_tr, m_te), "oof": oof,
        "cv_precision_at_recall": precision_at_recall(ytr, oof) if oof.std() > 0 else 0.0, "proba_test": p_te, "model": model,
    }
    log.info(f"        umbral={thr:.3f} | TEST recall={m_te['recall']:.3f} precision={m_te['precision']:.3f} "
             f"F1={m_te['f1']:.3f} PR-AUC={m_te['pr_auc']:.3f} ROC-AUC={m_te['roc_auc']:.3f} "
             f"Brier={m_te['brier']:.4f} | TRAIN recall={m_tr['recall']:.3f} "
             f"precision={m_tr['precision']:.3f} PR-AUC={m_tr['pr_auc']:.3f} | {t_train:.2f} s")
    log.info(f"        Diagnostico: {res['diagnostico']}")
    return res


def precision_at_recall(y_true, proba, recall_min: float = RECALL_TARGET_CV) -> float:
    """Metrica del proyecto: mejor precision alcanzable con recall >= recall_min."""
    p, r, _ = precision_recall_curve(y_true, proba)
    ok = r[:-1] >= recall_min
    return float(p[:-1][ok].max()) if ok.any() else 0.0


def tune_gradient_boosting(enc, Xtr, ytr) -> tuple[Pipeline, dict]:
    # Sin class_weight="balanced": con desbalance 6.9%, pesar 13.5x infla las probabilidades
    # predichas (Brier ~0.050 vs ~0.029) sin mejorar el ordenamiento. Entrenar sin pesos mantiene
    # probabilidades calibradas y SHAP TreeExplainer compatible; el umbral se ajusta via CV.
    pipe = Pipeline([("prep", enc), ("model", HistGradientBoostingClassifier(
        class_weight=None, max_iter=500, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=RANDOM_STATE))])
    grid = {
        "model__learning_rate": [0.03, 0.05],
        "model__max_leaf_nodes": [5, 7],
        "model__min_samples_leaf": [50, 100],
        "model__l2_regularization": [5.0, 10.0],
    }
    scorer = make_scorer(precision_at_recall, response_method="predict_proba")
    log.info(f"[INFO] GridSearchCV Gradient Boosting: {np.prod([len(v) for v in grid.values()])} "
             f"combinaciones x 5 folds (metrica: precision con recall >= {RECALL_TARGET_CV})...")
    gs = GridSearchCV(pipe, grid, scoring=scorer, cv=CV, n_jobs=-1)
    t0 = time.perf_counter()
    gs.fit(Xtr, ytr)
    best = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
    best["class_weight"] = None
    best["n_iter (early stopping)"] = int(gs.best_estimator_.named_steps["model"].n_iter_)
    log.info(f"        Mejores hiperparametros: {best} (precision@recall0.87 CV={gs.best_score_:.3f}, "
             f"{time.perf_counter()-t0:.1f} s)")
    return gs.best_estimator_, best


def tune_gradient_boosting_regressor(enc, Xr, yr, cv_reg) -> tuple[Pipeline, dict]:
    """Ajusta HistGradientBoostingRegressor con early stopping y validacion cruzada 5-fold (sin mirar test)."""
    pipe = Pipeline([("prep", enc), ("model", HistGradientBoostingRegressor(
        max_iter=500, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=RANDOM_STATE))])
    grid = {
        "model__learning_rate": [0.03, 0.05],
        "model__max_leaf_nodes": [15, 31],
        "model__min_samples_leaf": [20, 30],
        "model__l2_regularization": [1.0, 5.0],
    }
    gs = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error", cv=cv_reg, n_jobs=-1)
    gs.fit(Xr, yr)
    best = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
    best["n_iter (early stopping)"] = int(gs.best_estimator_.named_steps["model"].n_iter_)
    log.info(f"[INFO] GridSearchCV RUL Regressor: {best} (CV MAE={-gs.best_score_:.1f} ciclos)")
    return gs.best_estimator_, best


# --------------------------------------------------------------------------- graficas
def plot_confusion(y, proba, thr, title, path):
    cm = confusion_matrix(y, (proba >= thr).astype(int))
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay(cm, display_labels=["Sin falla (0)", "Falla (1)"]).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Prediccion del modelo")
    ax.set_ylabel("Valor real")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return cm


def plot_pr_curves(results, yte, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for r in results:
        if r["experimento"].startswith(("Exp 0", "Exp 5")):
            continue
        p, rc, _ = precision_recall_curve(yte, r["proba_test"])
        ax.plot(rc, p, label=f"{r['experimento']} (PR-AUC {r['test']['pr_auc']:.2f})")
    ax.axvline(0.85, ls="--", c="gray", lw=1)
    ax.axhline(0.55, ls="--", c="gray", lw=1)
    ax.text(0.86, 0.95, "meta: recall >= 0.85,\nprecision >= 0.55", fontsize=8, color="gray", va="top")
    ax.set_xlabel("Recall (fallas detectadas)")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall en el test set (5,000 baterias)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_capture(yte, proba, thr, path):
    order = np.argsort(-proba)
    captured = np.cumsum(yte.values[order]) / yte.sum()
    n = np.arange(1, len(yte) + 1)
    k = int((proba >= thr).sum())
    pct_fleet = k / len(yte) * 100
    pct_captured = (captured[k - 1] * 100) if k > 0 else 0.0
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(n / len(yte) * 100, captured * 100, lw=2, label="Gradient Boosting")
    ax.plot([0, 100], [0, 100], ls=":", c="gray", label="Inspeccion al azar")
    ax.scatter([pct_fleet], [pct_captured], c="red", zorder=5,
               label=f"Umbral {thr:.2f}: inspeccionar {k} ({k/len(yte):.1%}) -> {pct_captured:.1f}% fallas")
    ax.set_xlabel("% de la flota inspeccionada (ordenada por riesgo)")
    ax.set_ylabel("% de fallas capturadas")
    ax.set_title("Curva de captura de fallas (grafica 3 del dashboard)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Entrenamiento BatteryGuard AI")
    ap.add_argument("--data", default="data/processed", help="carpeta con train.csv y test.csv")
    ap.add_argument("--output", default="models/", help="carpeta para los modelos")
    ap.add_argument("--reports", default="reports/")
    ap.add_argument("--raw", default="data/raw/ev_battery_failure_prediction_Dataset.csv",
                    help="CSV crudo (entrada del preprocesamiento y del experimento de control con fuga)")
    ap.add_argument("--preprocess", action="store_true", help="fuerza a rerun del preprocesamiento")
    a = ap.parse_args()
    data_dir, out_dir, rep_dir = ROOT / a.data, ROOT / a.output, ROOT / a.reports
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(rep_dir / "train_log.txt")
    t_global = time.perf_counter()

    # Si no existen los datos procesados (repo recien clonado) o se pide --preprocess, corre el pipeline
    if a.preprocess or not (data_dir / "train.csv").exists() or not (out_dir / "feature_schema.json").exists():
        from src.data.preprocess import run as run_preprocess
        log.info("[INFO] Ejecutando pipeline de preprocesamiento (src/data/preprocess.py)...")
        run_preprocess(ROOT / a.raw, data_dir, out_dir)
    schema = json.loads((out_dir / "feature_schema.json").read_text())
    num, cat = schema["numeric_features"], schema["categorical_features"]
    tr = pd.read_csv(data_dir / "train.csv")
    te = pd.read_csv(data_dir / "test.csv")
    log.info("=" * 100)
    log.info("BatteryGuard AI - Entrenamiento de modelos (Entrega 3)")
    log.info("=" * 100)
    log.info(f"[INFO] Cargando dataset procesado: {a.data}/train.csv + test.csv ({len(tr)+len(te):,} registros)")
    log.info(f"[INFO] Conjunto Train: {len(tr):,} muestras ({int(tr[TARGET_CLF].sum())} fallas) | "
             f"Conjunto Test: {len(te):,} muestras ({int(te[TARGET_CLF].sum())} fallas)")
    log.info(f"[INFO] Predictoras: {len(num)} numericas + {len(cat)} categoricas | "
             f"excluidas por fuga: {', '.join(LEAKAGE_COLS)}")
    Xtr, ytr, Xte, yte = tr[num + cat], tr[TARGET_CLF], te[num + cat], te[TARGET_CLF]

    # ---------------- Clasificacion
    log.info("-" * 100)
    log.info("PARTE 1 - CLASIFICACION DEL RIESGO DE FALLA")
    log.info("-" * 100)
    results = []
    results.append(run_classifier("Exp 0: Baseline ingenuo (siempre 'sin falla')",
                                  DummyClassifier(strategy="most_frequent"), Xtr, ytr, Xte, yte))
    results.append(run_classifier("Exp 1: Regresion Logistica",
                                  Pipeline([("prep", make_encoder(num, cat, scale=True)),
                                            ("model", LogisticRegression(C=1.0, class_weight="balanced",
                                                                         max_iter=5000, random_state=RANDOM_STATE))]),
                                  Xtr, ytr, Xte, yte))
    results.append(run_classifier("Exp 2: Random Forest (por defecto)",
                                  Pipeline([("prep", make_encoder(num, cat, scale=False)),
                                            ("model", RandomForestClassifier(
                                                n_estimators=300, class_weight="balanced", n_jobs=-1,
                                                random_state=RANDOM_STATE))]),
                                  Xtr, ytr, Xte, yte))
    results.append(run_classifier("Exp 3: Gradient Boosting (por defecto)",
                                  Pipeline([("prep", make_encoder(num, cat, scale=False)),
                                            ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE))]),
                                  Xtr, ytr, Xte, yte))
    best_gb, best_params = tune_gradient_boosting(make_encoder(num, cat, scale=False), Xtr, ytr)
    results.append(run_classifier("Exp 4: Gradient Boosting ajustado", best_gb, Xtr, ytr, Xte, yte))
    results[-1]["hiperparametros"] = best_params

    # Exp 5 (control de fuga): exactamente mismas filas, orden e imputacion que Exp 1-4, agregando LEAKAGE_COLS
    from src.data.preprocess import anonymize
    raw = pd.read_csv(ROOT / a.raw).drop_duplicates()
    raw["battery_id_anon"] = raw["battery_serial"].map(anonymize)
    raw_idx = raw.set_index("battery_id_anon")
    leak_tr = raw_idx.loc[tr["battery_id_anon"]].reset_index()
    leak_te = raw_idx.loc[te["battery_id_anon"]].reset_index()
    for col in cat:
        for part in (leak_tr, leak_te):
            part[col] = part[col].where(part[col].notna(), np.nan)
            mask = part[col].notna()
            if mask.any():
                part.loc[mask, col] = part.loc[mask, col].astype(str)
    num_leak = num + LEAKAGE_COLS
    leak_params = {k: v for k, v in best_params.items() if not k.startswith("n_iter")}
    gb_leak = Pipeline([("prep", ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_leak),
        ("cat", Pipeline([("imp", _mode_imputer()), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])),
        ("model", HistGradientBoostingClassifier(max_iter=500,
                                                 random_state=RANDOM_STATE,
                                                 early_stopping=True, n_iter_no_change=20,
                                                 **leak_params))])
    results.append(run_classifier("Exp 5: Control CON variables con fuga (no se usa)", gb_leak,
                                  leak_tr[num_leak + cat], leak_tr[TARGET_CLF],
                                  leak_te[num_leak + cat], leak_te[TARGET_CLF]))

    # ---------------- Seleccion del ganador (solo modelos validos, sin fuga)
    candidates = [r for r in results if r["experimento"].startswith(("Exp 1", "Exp 2", "Exp 3", "Exp 4"))]
    # Criterio SIN mirar el test: mayor precision con recall >= 0.87 en validacion cruzada (train).
    winner = max(candidates, key=lambda r: r["cv_precision_at_recall"])
    for r in candidates:
        log.info(f"[INFO] CV (train) precision con recall>=0.87 -> {r['experimento']}: {r['cv_precision_at_recall']:.3f}")
    log.info("-" * 100)
    log.info(f"[RESULTADO] Modelo ganador: {winner['experimento']}")
    thr_high = winner["umbral"]
    thr_med = threshold_for_recall(ytr, winner["oof"], RECALL_MEDIUM_CV)
    log.info(f"[INFO] Umbral riesgo ALTO = {thr_high:.3f} | umbral riesgo MEDIO = {thr_med:.3f} (recall CV {RECALL_MEDIUM_CV})")
    log.info("[RESULTADOS EN TEST SET - modelo ganador, clase 1 = falla]")
    report = classification_report(yte, (winner["proba_test"] >= thr_high).astype(int),
                                   target_names=["0 sin falla", "1 falla"], digits=3)
    for line in report.splitlines():
        log.info("    " + line)

    cm = plot_confusion(yte, winner["proba_test"], thr_high,
                        f"Matriz de confusion - {winner['experimento']}\n(test, umbral {thr_high:.2f})",
                        rep_dir / "matriz_confusion_ganador.png")
    lr = next(r for r in results if r["experimento"].startswith("Exp 1"))
    cm_lr = plot_confusion(yte, lr["proba_test"], lr["umbral"],
                           f"Matriz de confusion - Regresion Logistica\n(test, umbral {lr['umbral']:.2f})",
                           rep_dir / "matriz_confusion_logistica.png")
    plot_pr_curves(results, yte, rep_dir / "curvas_precision_recall.png")
    plot_capture(yte, winner["proba_test"], thr_high, rep_dir / "curva_captura_fallas.png")
    log.info(f"[INFO] Matriz de confusion (TN, FP / FN, TP): {cm.ravel().tolist()}")

    # ---------------- Regresion de vida util
    log.info("-" * 100)
    log.info("PARTE 2 - REGRESION DE VIDA UTIL RESTANTE (ciclos)")
    log.info("-" * 100)
    rtr, rte = tr.dropna(subset=[TARGET_REG]), te.dropna(subset=[TARGET_REG])
    log.info(f"[INFO] Filas con target de vida util: train {len(rtr):,} | test {len(rte):,} "
             f"(se omiten {len(tr)-len(rtr)+len(te)-len(rte)} filas sin target; el target NO se imputa)")
    meta_mae = 0.10 * rtr[TARGET_REG].mean()
    cv_reg = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_r2_pipe, _ = tune_gradient_boosting_regressor(
        make_encoder(num, cat, scale=False), rtr[num + cat], rtr[TARGET_REG], cv_reg
    )
    reg_results = []
    for name, model in [
        ("R1: Ridge (baseline lineal)", Pipeline([("prep", make_encoder(num, cat, scale=True)),
                                                   ("model", Ridge(alpha=1.0))])),
        ("R2: Gradient Boosting Regressor", best_r2_pipe),
    ]:
        oof_reg = cross_val_predict(model, rtr[num + cat], rtr[TARGET_REG], cv=cv_reg, n_jobs=-1)
        cv_mae = mean_absolute_error(rtr[TARGET_REG], oof_reg)
        t0 = time.perf_counter()
        model.fit(rtr[num + cat], rtr[TARGET_REG])
        tt = time.perf_counter() - t0
        p_tr, p_te = model.predict(rtr[num + cat]), model.predict(rte[num + cat])
        r = {"experimento": name, "model": model, "t_entrenamiento_s": tt, "cv_mae": cv_mae,
             "train": {"mae": mean_absolute_error(rtr[TARGET_REG], p_tr), "r2": r2_score(rtr[TARGET_REG], p_tr)},
             "test": {"mae": mean_absolute_error(rte[TARGET_REG], p_te), "r2": r2_score(rte[TARGET_REG], p_te)}}
        reg_results.append(r)
        log.info(f"[INFO] {name}: CV MAE={cv_mae:.1f} | TEST MAE={r['test']['mae']:.1f} ciclos R2={r['test']['r2']:.3f} | "
                 f"TRAIN MAE={r['train']['mae']:.1f} R2={r['train']['r2']:.3f} | {tt:.2f} s")
    # Seleccion SIN mirar el test: menor MAE en validacion cruzada 5-fold sobre train
    reg_winner = min(reg_results, key=lambda r: r["cv_mae"])
    log.info(f"[RESULTADO] Modelo de vida util ganador: {reg_winner['experimento']} "
             f"(MAE {reg_winner['test']['mae']:.1f} ciclos; meta <= {meta_mae:.0f}) -> "
             f"{'CUMPLE' if reg_winner['test']['mae'] <= meta_mae else 'NO CUMPLE'}")

    # ---------------- Serializacion
    log.info("-" * 100)
    joblib.dump(winner["model"], out_dir / "modelo_battery_guard.joblib")
    joblib.dump(reg_winner["model"], out_dir / "modelo_battery_guard_vida_util.joblib")
    joblib.dump(lr["model"], out_dir / "modelo_baseline_logistica.joblib")
    joblib.dump(lr["model"].named_steps["prep"].named_transformers_["num"], out_dir / "scaler.joblib")
    config = {
        "modelo_falla": winner["experimento"], "hiperparametros": winner.get("hiperparametros", {}),
        "umbral_alto": thr_high, "umbral_medio": thr_med,
        "modelo_vida_util": reg_winner["experimento"],
        "features": num + cat, "numeric_features": num, "categorical_features": cat,
        "test_metrics": winner["test"], "test_mae_vida_util": reg_winner["test"]["mae"],
    }
    (out_dir / "model_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, default=float))
    log.info(f"[SUCCESS] Modelo de falla serializado en: {a.output.rstrip('/')}/modelo_battery_guard.joblib")
    log.info(f"[SUCCESS] Modelo de vida util serializado en: {a.output.rstrip('/')}/modelo_battery_guard_vida_util.joblib")
    log.info(f"[SUCCESS] Baseline + escalador guardados en: {a.output.rstrip('/')}/modelo_baseline_logistica.joblib, scaler.joblib")
    log.info(f"[SUCCESS] Umbrales y configuracion en: {a.output.rstrip('/')}/model_config.json")

    # ---------------- Tabla de experimentos
    rows = []
    for r in results:
        rows.append({"experimento": r["experimento"], "umbral": round(r["umbral"], 3),
                     **{f"test_{k}": round(v, 4) for k, v in r["test"].items()},
                     **{f"train_{k}": round(v, 4) for k, v in r["train"].items() if k in ("recall", "precision", "pr_auc")},
                     "cv_precision_at_recall_087": round(r["cv_precision_at_recall"], 4),
                     "t_entrenamiento_s": round(r["t_entrenamiento_s"], 2),
                     "t_inferencia_ms_por_bateria": round(r["t_inferencia_ms_por_bateria"], 4),
                     "diagnostico": r["diagnostico"]})
    pd.DataFrame(rows).to_csv(rep_dir / "experimentos_clasificacion.csv", index=False)
    pd.DataFrame([{"experimento": r["experimento"], "cv_mae": round(r["cv_mae"], 1),
                   "test_mae": round(r["test"]["mae"], 1),
                   "test_r2": round(r["test"]["r2"], 4), "train_mae": round(r["train"]["mae"], 1),
                   "train_r2": round(r["train"]["r2"], 4), "t_entrenamiento_s": round(r["t_entrenamiento_s"], 2)}
                  for r in reg_results]).to_csv(rep_dir / "experimentos_vida_util.csv", index=False)
    (rep_dir / "matrices_confusion.json").write_text(json.dumps(
        {"ganador": cm.tolist(), "logistica": cm_lr.tolist()}, indent=2))
    log.info(f"[SUCCESS] Reportes en {a.reports}: experimentos_*.csv, matriz_confusion_*.png, curvas_*.png")
    log.info(f"[INFO] Tiempo total: {time.perf_counter() - t_global:.1f} s")


def _mode_imputer():
    return SimpleImputer(strategy="most_frequent")


if __name__ == "__main__":
    main()
