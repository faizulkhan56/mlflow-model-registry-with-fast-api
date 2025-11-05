import os
import sys
import json
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint

MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "WineQuality-RandomForest-Tuned"
REGISTERED_MODEL_NAME = "WineQuality-RandomForest-Model"
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset", "winequality-red.csv")

CV_SPLITS = 5
N_ITER = 40
RANDOM_STATE = 42
N_JOBS = -1

def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH, sep=";")
    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    base_model = RandomForestRegressor(
        n_estimators=400,
        bootstrap=True,
        oob_score=True,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    param_distributions = {
        "n_estimators": randint(300, 1001),
        "max_depth": [None] + list(range(5, 31, 5)),
        "min_samples_split": randint(2, 21),
        "min_samples_leaf": randint(1, 6),
        "max_features": ["sqrt", "log2", None],
    }

    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=N_JOBS,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True,
    )

    with mlflow.start_run() as run:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_params(search.best_params_)
        mlflow.log_param("cv_splits", CV_SPLITS)
        mlflow.log_param("randomized_search_n_iter", N_ITER)

        if hasattr(best_model, "oob_score_") and best_model.oob_score_ is not None:
            mlflow.log_metric("oob_score", float(best_model.oob_score_))

        feature_info = {
            "feature_names": list(X.columns),
            "target_name": "quality",
        }
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/feature_info.json", "w") as f:
            json.dump(feature_info, f, indent=2)
        mlflow.log_artifact("artifacts/feature_info.json")

        signature = infer_signature(X_test, y_pred)
        input_example = X_test.iloc[:1]

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="wine-quality-model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print("\n============================")
        print(f"Run ID:      {run.info.run_id}")
        print(f"Best params: {search.best_params_}")
        print(f"Test RMSE:   {rmse:.4f}")
        print(f"Test MAE:    {mae:.4f}")
        print(f"Test R2:     {r2:.4f}")
        print("============================\n")

        try:
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            versions_sorted = sorted(versions, key=lambda mv: int(mv.version))
            latest = versions_sorted[-1]
            print(f"Registered model '{REGISTERED_MODEL_NAME}' now at version: {latest.version}")
        except Exception as e:
            print(f"(Info) Could not list model versions: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
