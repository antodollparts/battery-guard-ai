# BatteryGuard AI — Modelo de IA

Sistema de IA que estima el **riesgo de falla** y la **vida útil restante (ciclos)** de las baterías de una flota de vehículos eléctricos.
Grupo LITIID71 · Fundamentos de Inteligencia Artificial.

## Integrantes del Equipo

* **Antonella Junco** - [@antodollparts](https://github.com/antodollparts/)
* **Amerika Carbajal** - [@Amerikita](https://github.com/Amerikita)
* **Victor Canché** - [@usuario3](https://github.com/victor332-af)
* **Diego Alexander** - [@DiegoMarin025](https://github.com/DiegoMarin025)
* **Leobardo Mex** -[@YondaimeMex](https://github.com/YondaimeMex)

## Estructura

```
.
├── data/
│   ├── raw/
│   │   └── ev_battery_failure_prediction_Dataset.csv   # CSV original (20,000 × 70), sin modificar
│   └── processed/
│       ├── train.csv                                    # 75 % estratificado, imputado, sin fuga
│       └── test.csv                                     # 25 % estratificado, imputado, sin fuga
├── models/
│   ├── feature_schema.json                              # esquema de variables
│   ├── imputer.joblib                                   # imputación
│   ├── model_config.json                                # umbrales y configuración
│   ├── modelo_baseline_logistica.joblib                 # baseline (regresión logística)
│   ├── modelo_battery_guard.joblib                      # Gradient Boosting ajustado (ganador, clasificación)
│   ├── modelo_battery_guard_vida_util.joblib            # Gradient Boosting Regressor (vida útil)
│   └── scaler.joblib                                    # StandardScaler del baseline
├── reports/                                             # log, tablas de experimentos, matrices y curvas
│   ├── curva_captura_fallas.*
│   ├── curvas_precision_recall.*
│   ├── experimentos_clasificacion.*
│   ├── experimentos_vida_util.*
│   ├── matrices_confusion.json
│   ├── matriz_confusion_*.*
│   └── train_log.txt
├── src/
│   ├── data/
│   │   └── preprocess.py                                # pipeline de preprocesamiento
│   └── models/
│       ├── train.py                                     # entrenamiento + experimentos + reportes
│       └── predict.py                                   # predecir() / predecir_lote() (contrato con el dashboard)
├── tests/
│   └── test_predict.py                                  # pruebas del contrato de inferencia
├── .gitignore
├── README.md
└── requirements.txt
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
