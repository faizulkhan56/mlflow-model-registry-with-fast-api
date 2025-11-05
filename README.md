# Serving a ML Model from MLflow Model Registry with FastAPI

This lab shows how to train, register, and serve a Scikit‑learn model using **MLflow** and **FastAPI**.

## Project Structure
```
wine_quality_mlflow_lab/
├── dataset/
│   └── winequality-red.csv         # (you add this file; see below)
├── mlruns/                         # created automatically by MLflow
├── README.md
├── requirements.txt
├── .gitignore
├── train_v1.py                     # Baseline RF
├── train_v2.py                     # Tuned RF with CV search
├── serve.py                        # FastAPI server loading from Registry
└── predict.py                      # CLI test loading from Registry
```

> The dataset is **semicolon-delimited**. Download from UCI:  
> https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv  
> Save to: `dataset/winequality-red.csv`

## 1) Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Run MLflow Tracking Server

Local file store:
```bash
mlflow server \
  --host 0.0.0.0 --port 5000 \
  --allowed-hosts '*' \  --cors-allowed-origins '*'
```
Open MLflow UI: `http://<host>:5000`

## 3) Train & Register the Model

### Option A — Baseline (`train_v1.py`)
```bash
python train_v1.py
```
- Simple RandomForest (`n_estimators=100, max_depth=5`).
- Logs RMSE + params.
- Registers as **WineQuality-RandomForest-Model** (Version 1 if first time).

### Option B — Tuned (`train_v2.py`)
```bash
python train_v2.py
```
- **RandomizedSearchCV** (5-fold) over RF hyperparameters.
- Logs RMSE/MAE/R², best params, OOB score.
- Registers **best** model (becomes Version 2, then 3, etc.).

### Promote to Production
In the MLflow UI:
- **Models → WineQuality-RandomForest-Model → Version X → Stage → Promote to Production**

## 4) Serve the Model (FastAPI)

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```
- Swagger UI: `http://<host>:8000/docs`
- Endpoints:
  - `GET /` → health + model URI
  - `GET /test` → quick fixed-row prediction
  - `POST /predict/` → provide JSON body:
```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

> `serve.py` loads `models:/WineQuality-RandomForest-Model/Production` by default.  
> If you haven't promoted any version yet, temporarily pin a version number, e.g. `.../2`.

## 5) CLI Prediction

```bash
python predict.py
```

## Differences: `train_v1.py` vs `train_v2.py`
| Aspect | `train_v1.py` (Version 1) | `train_v2.py` (Version 2) |
|---|---|---|
| Goal | Baseline reproducible model | Higher quality via hyperparameter tuning |
| Algorithm | RandomForestRegressor | RandomForestRegressor |
| Hyperparams | Fixed (`n_estimators=100`, `max_depth=5`) | **Searched** trees/depth/min samples/max_features |
| CV | None | 5‑fold `RandomizedSearchCV` (n_iter=40) |
| Metrics | RMSE | RMSE, MAE, R², OOB score |
| Artifacts | Model only | Model + `feature_info.json`, input example, signature |
| Registry | Registers a version | Registers **best** model as new version |
| Best Use | Quick baseline / sanity check | Production candidate |

## Notes
- Keep inference inputs within realistic ranges; extreme values unseen in training tend to yield near‑mean predictions.
- Focus on **lower RMSE / better generalization**, not “increasing the number” of predictions.
- Prefer loading by **stage** (`/Production`) in serving so promotions take effect without code changes.
