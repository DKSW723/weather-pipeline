import joblib
import pandas as pd
from ml.features import FEATURE_COLS

def load_model(path: str = "models/model.pkl"):
    return joblib.load(path)

def predict(features: dict) -> float:
    model = load_model()
    df = pd.DataFrame([features])
    df = df[FEATURE_COLS]
    prediction = model.predict(df)
    return round(float(prediction[0]), 2)