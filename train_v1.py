import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

MLFLOW_URI = "http://localhost:5000"
REGISTERED_MODEL_NAME = "WineQuality-RandomForest-Model"
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset", "winequality-red.csv")

def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("WineQuality-Baseline")

    df = pd.read_csv(DATA_PATH, sep=";")
    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        signature = infer_signature(X_test, y_pred)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="wine-quality-model",
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Test RMSE: {rmse:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
