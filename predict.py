import sys
import mlflow
import mlflow.sklearn
import pandas as pd

MLFLOW_URI = "http://localhost:5000"
MODEL_NAME = "WineQuality-RandomForest-Model"
MODEL_URI = f"models:/{MODEL_NAME}/Production"  # or pin to a version e.g., /2

mlflow.set_tracking_uri(MLFLOW_URI)

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print(f"Model loaded successfully from: {MODEL_URI}")
    if hasattr(model, 'feature_names_in_'):
        print("Expected features:", list(model.feature_names_in_))
except Exception as e:
    print(f"ERROR: Failed to load model from {MODEL_URI}: {e}", file=sys.stderr)
    print("\nTip: Ensure you have:", file=sys.stderr)
    print("  1. Started the MLflow server (mlflow server --host 0.0.0.0 --port 5000)", file=sys.stderr)
    print("  2. Trained and registered a model (train_v1.py or train_v2.py)", file=sys.stderr)
    print("  3. Promoted a model version to Production stage", file=sys.stderr)
    print("  4. Or modify MODEL_URI to use a specific version (e.g., /2)", file=sys.stderr)
    sys.exit(1)

sample = pd.DataFrame([{
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}])

try:
    prediction = model.predict(sample)
    print("Prediction:", float(prediction[0]))
except Exception as e:
    print(f"ERROR: Prediction failed: {e}", file=sys.stderr)
    sys.exit(1)
