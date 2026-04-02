import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from etl.load import read_from_db
from ml.features import FEATURE_COLS, TARGET_COL

def train():
    df = read_from_db()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Kein shuffle – wir haben Zeitreihendaten!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    mlflow.set_experiment("weather-forecast")

    with mlflow.start_run():
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})
        mlflow.sklearn.log_model(model, "model")

        print(f"MAE:  {mae:.2f} °C")
        print(f"RMSE: {rmse:.2f} °C")
        print(f"R2:   {r2:.3f}")

        # Modell lokal speichern
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        print("Model saved to models/model.pkl")

if __name__ == "__main__":
    train()