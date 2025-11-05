import mlflow
import mlflow.sklearn
import pandas as pd

MLFLOW_URI = "http://localhost:5000"
MODEL_NAME = "WineQuality-RandomForest-Model"
MODEL_URI = f"models:/{MODEL_NAME}/Production"  # or pin to a version e.g., /2

mlflow.set_tracking_uri(MLFLOW_URI)
model = mlflow.sklearn.load_model(MODEL_URI)

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

print("Prediction:", float(model.predict(sample)[0]))
