# Serving a ML Model from MLflow Model Registry with FastAPI

This lab shows how to train, register, and serve a Scikit‑learn model using **MLflow** and **FastAPI**.

## Project Structure
```
mlflow-model-registry-with-fast-api/
├── dataset/
│   ├── README.md                   # Dataset instructions
│   └── winequality-red.csv         # (you add this file; see below)
├── mlruns/                         # created automatically by MLflow
├── artifacts/                      # created by train_v2.py (gitignored)
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

## Quick Start (Deployment Checklist)

To deploy this project from scratch, follow these steps in order:

1. ✅ **Prerequisites**: Python 3.8+, download dataset to `dataset/winequality-red.csv`
2. ✅ **Environment Setup**: Create venv, install `requirements.txt`
3. ✅ **Start MLflow Server**: Run `mlflow server --host 0.0.0.0 --port 5000` (keep it running)
4. ✅ **Train Model**: Run `train_v1.py` or `train_v2.py` to register a model
5. ✅ **Promote to Production**: In MLflow UI (http://localhost:5000), promote a model version to Production stage
6. ✅ **Serve API**: Run `uvicorn serve:app --host 0.0.0.0 --port 8000`
7. ✅ **Test**: Visit `http://localhost:8000/docs` or run `python predict.py`

## Prerequisites

- **Python 3.8+** installed
- **MLflow Tracking Server** running (see Section 2)
- **Dataset file** downloaded: `dataset/winequality-red.csv` (see project structure above)

## 1) Environment Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import mlflow, fastapi, sklearn; print('All packages installed successfully')"
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
- **Experiment:** `WineQuality-Baseline`
- **Model:** Simple RandomForest with fixed hyperparameters (`n_estimators=100, max_depth=5, random_state=42`)
- **Metrics:** RMSE only
- **Artifacts:** Model with signature
- **Output:** Prints Run ID and Test RMSE
- **Registry:** Registers as **WineQuality-RandomForest-Model** (Version 1 if first time)

### Option B — Tuned (`train_v2.py`)
```bash
python train_v2.py
```
- **Experiment:** `WineQuality-RandomForest-Tuned`
- **Model:** RandomForest with **RandomizedSearchCV** (5-fold KFold, 40 iterations) over hyperparameters
- **Base Model:** `n_estimators=400`, `bootstrap=True`, `oob_score=True`
- **Hyperparameter Search:** `n_estimators` (300-1000), `max_depth` (None, 5-30), `min_samples_split` (2-20), `min_samples_leaf` (1-5), `max_features` (sqrt, log2, None)
- **Metrics:** RMSE, MAE, R², OOB score
- **Artifacts:** Model with signature, input example, `feature_info.json`
- **Output:** Prints detailed summary with Run ID, best params, and all metrics; displays registered model version
- **Registry:** Registers the **best** model from search (becomes Version 2, then 3, etc.)

### Promote Model to Production Stage

**Important:** The `serve.py` application uses `models:/WineQuality-RandomForest-Model/Production` to load the model. This means you **must** promote at least one model version to the **Production** stage before running the serving API.

#### Option A: Promote via MLflow UI (Recommended)

1. Open MLflow UI: `http://localhost:5000`
2. Navigate to **Models** tab
3. Click on **WineQuality-RandomForest-Model**
4. Select a model version (e.g., Version 1, 2, etc.)
5. Click on the **Stage** dropdown (shows "None" by default)
6. Select **Production** from the dropdown
7. Confirm the promotion

**Screenshot path:** `Models → WineQuality-RandomForest-Model → [Select Version] → Stage → Production`

#### Option B: Promote via MLflow CLI

```bash
# Promote a specific version to Production
mlflow models transition-stage \
  --model-name WineQuality-RandomForest-Model \
  --version <VERSION_NUMBER> \
  --stage Production \
  --tracking-uri http://localhost:5000

# Example: Promote version 2 to Production
mlflow models transition-stage \
  --model-name WineQuality-RandomForest-Model \
  --version 2 \
  --stage Production \
  --tracking-uri http://localhost:5000
```

#### Option C: Promote via Python API

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5000")
client.transition_model_version_stage(
    name="WineQuality-RandomForest-Model",
    version=<VERSION_NUMBER>,
    stage="Production"
)
```

#### Understanding Model Stages

MLflow Model Registry supports three stages:
- **None** (default): Newly registered models start here
- **Staging**: For testing/validation before production
- **Production**: Active production models (used by `serve.py`)

**Why use Production stage?**
- `serve.py` automatically loads the latest model in Production stage
- No code changes needed when promoting new versions
- Ensures only validated models are served

**Alternative: Using Version Numbers**

If you haven't promoted any version to Production yet, you can temporarily modify `serve.py` to use a specific version:

```python
# Instead of: models:/WineQuality-RandomForest-Model/Production
MODEL_URI = f"models:/{MODEL_NAME}/2"  # Use version 2
```

## 4) Serve the Model (FastAPI)

**Prerequisites:** Ensure at least one model version is promoted to **Production** stage (see Section 3 above).

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

**Note:** `serve.py` loads the model using `models:/WineQuality-RandomForest-Model/Production`. If you get an error that the model is not found, ensure you've promoted a version to Production stage (see Section 3) or temporarily modify `serve.py` to use a specific version number (e.g., `models:/WineQuality-RandomForest-Model/2`).

## 5) CLI Prediction

**Prerequisites:** Ensure at least one model version is promoted to **Production** stage (see Section 3 above).

```bash
python predict.py
```

- Loads the model from `models:/WineQuality-RandomForest-Model/Production`
- Performs a sample prediction with predefined wine features
- Prints the prediction result
- Includes error handling with helpful tips if the model is not found

## Differences: `train_v1.py` vs `train_v2.py`

| Aspect | `train_v1.py` | `train_v2.py` |
|---|---|---|
| **Goal** | Baseline reproducible model | Higher quality via hyperparameter tuning |
| **Experiment Name** | `WineQuality-Baseline` | `WineQuality-RandomForest-Tuned` |
| **Algorithm** | RandomForestRegressor | RandomForestRegressor |
| **Base Config** | `n_estimators=100`, `max_depth=5`, `random_state=42` | `n_estimators=400`, `bootstrap=True`, `oob_score=True`, `n_jobs=-1` |
| **Hyperparameters** | Fixed values | **RandomizedSearchCV** over 5 hyperparameters |
| **Search Space** | N/A | `n_estimators`: 300-1000<br>`max_depth`: None, 5-30<br>`min_samples_split`: 2-20<br>`min_samples_leaf`: 1-5<br>`max_features`: sqrt, log2, None |
| **Cross-Validation** | None | 5-fold `KFold` with `RandomizedSearchCV` (n_iter=40) |
| **Metrics Logged** | RMSE | RMSE, MAE, R², OOB score |
| **Parameters Logged** | Fixed params | Best params from search + CV splits + n_iter |
| **Artifacts** | Model + signature | Model + signature + input example + `feature_info.json` |
| **Additional Files** | None | Creates `artifacts/feature_info.json` (gitignored) |
| **Output** | Run ID, Test RMSE | Detailed summary: Run ID, best params, RMSE, MAE, R², version info |
| **Registry Behavior** | Registers model as new version | Registers **best** model from search as new version |
| **Execution Time** | Fast (~seconds) | Slower (hyperparameter search) |
| **Best Use Case** | Quick baseline / sanity check / initial model | Production candidate / optimized model |

## Troubleshooting

### Common Issues

**1. Model not found error when running `serve.py` or `predict.py`:**
- **Error**: `Failed to load model from models:/WineQuality-RandomForest-Model/Production`
- **Solution**: 
  - Ensure MLflow server is running: `mlflow server --host 0.0.0.0 --port 5000`
  - Verify you've trained and registered a model (run `train_v1.py` or `train_v2.py`)
  - **Most importantly**: Promote a model version to Production stage (see Section 3)
  - Alternative: Temporarily modify `MODEL_URI` in `serve.py`/`predict.py` to use a specific version: `models:/WineQuality-RandomForest-Model/2`

**2. Dataset file not found:**
- **Error**: `FileNotFoundError` when running training scripts
- **Solution**: Download `winequality-red.csv` from UCI and place it in `dataset/winequality-red.csv`

**3. MLflow server connection error:**
- **Error**: Connection refused or timeout
- **Solution**: 
  - Ensure MLflow server is running on port 5000
  - Check firewall settings
  - Verify `MLFLOW_URI` in all scripts matches your server URL

**4. Port already in use:**
- **Error**: `Address already in use` when starting MLflow server or FastAPI
- **Solution**: 
  - Change port in the command (e.g., `--port 5001`)
  - Update `MLFLOW_URI` in all scripts to match
  - Or stop the process using the port

**5. Import errors:**
- **Error**: `ModuleNotFoundError`
- **Solution**: 
  - Ensure virtual environment is activated
  - Run `pip install -r requirements.txt`
  - Verify installation: `python -c "import mlflow, fastapi, sklearn"`

## Notes
- Keep inference inputs within realistic ranges; extreme values unseen in training tend to yield near‑mean predictions.
- Focus on **lower RMSE / better generalization**, not "increasing the number" of predictions.
- **Important:** Always promote a model version to **Production** stage before running `serve.py`, as it uses `models:/WineQuality-RandomForest-Model/Production` to load the model.
- Prefer loading by **stage** (`/Production`) in serving so promotions take effect without code changes. This allows you to update the production model by simply promoting a new version in MLflow, without modifying `serve.py`.
- All scripts use consistent constants:
  - `MLFLOW_URI = "http://localhost:5000"`
  - `MODEL_NAME / REGISTERED_MODEL_NAME = "WineQuality-RandomForest-Model"`
