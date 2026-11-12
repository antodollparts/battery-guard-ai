# BatteryGuard AI — Motor de IA (Entrega 3)

Sistema de IA que estima el **riesgo de falla** y la **vida útil restante (ciclos)** de las baterías de una flota de vehículos eléctricos.
Grupo LITIID71 · Fundamentos de Inteligencia Artificial.

## Estructura

```
data/raw/ev_battery_failure_prediction_Dataset.csv   CSV original (20,000 × 70), sin modificar
data/processed/train.csv, test.csv                   75 % / 25 % estratificado, imputado, sin fuga
src/data/preprocess.py                               pipeline de preprocesamiento
src/models/train.py                                  entrenamiento + experimentos + reportes
src/models/predict.py                                predecir() / predecir_lote() (contrato con el dashboard)
models/modelo_falla.joblib                           Gradient Boosting ajustado (ganador, clasificación)
models/modelo_vida_util.joblib                       Gradient Boosting Regressor (vida útil)
models/modelo_baseline_logistica.joblib, scaler.joblib   baseline + StandardScaler
models/imputer.joblib, feature_schema.json, model_config.json   imputación, esquema y umbrales
reports/                                             log, tablas de experimentos, matrices y curvas
tests/test_predict.py                                pruebas del contrato de inferencia
```

## Reproducir (un solo comando)

```bash
pip install -r requirements.txt
python src/models/train.py --data data/processed --output models/ --preprocess
```

Corre el preprocesamiento y el entrenamiento completo en ~4 min (CPU) y deja los modelos en `models/` y el log en `reports/train_log.txt`. Las rutas son relativas a la raíz del repo y la semilla es fija (`random_state=42`).

## Usar el modelo

```python
from src.models.predict import predecir, predecir_lote
predecir({"battery_health_percent": 50.0, "internal_resistance": 0.95, "cell_temperature_max": 56.6})
```

```bash
python src/models/predict.py      # demo
python -m pytest tests/ -v        # pruebas
```

> Los `.joblib` se serializaron con scikit-learn 1.9.1. Si instalan otra versión, vuelvan a correr el comando de entrenamiento.
