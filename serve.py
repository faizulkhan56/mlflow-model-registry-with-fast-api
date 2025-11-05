from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd

app = FastAPI(title="Wine Quality Prediction API")

MLFLOW_URI = "http://localhost:5000"
MODEL_NAME = "WineQuality-RandomForest-Model"
MODEL_URI = f"models:/{MODEL_NAME}/Production"  # or pin a version: models:/.../2

mlflow.set_tracking_uri(MLFLOW_URI)

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print("Model loaded successfully from:", MODEL_URI)
    if hasattr(model, 'feature_names_in_'):
        print("Expected features:", list(model.feature_names_in_))
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_URI}: {e}")

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def root():
    return {"message": "Wine Quality Prediction API", "model_uri": MODEL_URI}

@app.get("/test")
def test_prediction():
    try:
        test_data = pd.DataFrame([{
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
        pred = model.predict(test_data)
        return {"test_prediction": float(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test prediction failed: {e}")

@app.post("/predict/")
def predict(wine: WineFeatures):
    try:
        row = pd.DataFrame([{
            "fixed acidity": wine.fixed_acidity,
            "volatile acidity": wine.volatile_acidity,
            "citric acid": wine.citric_acid,
            "residual sugar": wine.residual_sugar,
            "chlorides": wine.chlorides,
            "free sulfur dioxide": wine.free_sulfur_dioxide,
            "total sulfur dioxide": wine.total_sulfur_dioxide,
            "density": wine.density,
            "pH": wine.pH,
            "sulphates": wine.sulphates,
            "alcohol": wine.alcohol
        }])
        pred = model.predict(row)
        return {"prediction": float(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
