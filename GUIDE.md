# Step-by-Step Development Guide

This guide walks you through building the Weather Forecast ML Pipeline from scratch.
Each step builds on the previous one — follow them in order.

---

## Phase 1: Project Setup

### Step 1.1 – Initialize the project

```bash
mkdir weather-pipeline
cd weather-pipeline
git init
python -m venv venv
source venv/bin/activate
```

### Step 1.2 – Create the folder structure

```bash
mkdir -p data etl ml api notebooks tests models mlruns
touch etl/__init__.py ml/__init__.py api/__init__.py
touch etl/extract.py etl/transform.py etl/load.py
touch ml/features.py ml/train.py ml/predict.py
touch api/main.py
touch requirements.txt .gitignore .env.example
```

### Step 1.3 – Create requirements.txt

```
requests
pandas
numpy
scikit-learn
xgboost
mlflow
fastapi
uvicorn
sqlalchemy
python-dotenv
pytest
httpx
joblib
```

Install:
```bash
pip install -r requirements.txt
```

### Step 1.4 – Create .gitignore

```
venv/
__pycache__/
*.pyc
data/weather.db
models/*.pkl
mlruns/
.env
.DS_Store
```

---

## Phase 2: ETL Pipeline

### Step 2.1 – Extract (etl/extract.py)

Fetch historical weather data from Open-Meteo API for Berlin.

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_weather(latitude=52.52, longitude=13.41, days=365):
    """Fetch weather data from Open-Meteo API."""
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "sunshine_duration"
        ],
        "timezone": "Europe/Berlin",
        "start_date": start_date,
        "end_date": end_date
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    print(f"Fetched {len(df)} rows from Open-Meteo API.")
    return df

if __name__ == "__main__":
    df = fetch_weather()
    print(df.head())
```

Run and test:
```bash
python -m etl.extract
```

---

### Step 2.2 – Transform (etl/transform.py)

Clean data and engineer features.

```python
import pandas as pd

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and add features to the weather dataframe."""
    df = df.copy()

    # Drop rows with missing target
    df = df.dropna(subset=["temperature_2m_max"])

    # Fill remaining NaN with column median
    df = df.fillna(df.median(numeric_only=True))

    # Feature engineering
    df["dayofyear"] = df["time"].dt.dayofyear
    df["month"] = df["time"].dt.month
    df["weekday"] = df["time"].dt.weekday
    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]

    # Lag features (yesterday's values)
    df["temp_max_lag1"] = df["temperature_2m_max"].shift(1)
    df["temp_max_lag7"] = df["temperature_2m_max"].shift(7)
    df["precip_lag1"] = df["precipitation_sum"].shift(1)

    # Rolling mean (7-day)
    df["temp_rolling_7d"] = df["temperature_2m_max"].rolling(7).mean()

    # Drop rows with NaN from lag features
    df = df.dropna()

    print(f"Transformed data: {len(df)} rows, {len(df.columns)} columns.")
    return df

if __name__ == "__main__":
    from etl.extract import fetch_weather
    df = fetch_weather()
    df_transformed = transform(df)
    print(df_transformed.head())
```

---

### Step 2.3 – Load (etl/load.py)

Save transformed data into SQLite.

```python
import sqlite3
import pandas as pd
import os

DB_PATH = "data/weather.db"

def load_to_db(df: pd.DataFrame, table_name: str = "weather"):
    """Load dataframe into SQLite database."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Loaded {len(df)} rows into '{table_name}' table in {DB_PATH}.")

def read_from_db(table_name: str = "weather") -> pd.DataFrame:
    """Read data from SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    df["time"] = pd.to_datetime(df["time"])
    return df

if __name__ == "__main__":
    from etl.extract import fetch_weather
    from etl.transform import transform
    df = fetch_weather()
    df = transform(df)
    load_to_db(df)
```

Run the full ETL pipeline:
```bash
python -m etl.extract && python -m etl.transform  # test individually
python -m etl.load                                 # runs all three internally
```

---

## Phase 3: Exploratory Data Analysis

### Step 3.1 – Create notebooks/01_EDA.ipynb

Open Jupyter:
```bash
pip install jupyter
jupyter notebook
```

Cover these topics in your notebook:
1. Load data from SQLite
2. Distribution of temperature_2m_max (histogram)
3. Temperature over time (line plot)
4. Correlation heatmap of all features
5. Seasonal patterns (boxplot per month)
6. Check for missing values and outliers

This notebook goes into your GitHub and shows data science skills.

---

## Phase 4: Machine Learning

### Step 4.1 – Feature definition (ml/features.py)

```python
FEATURE_COLS = [
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "sunshine_duration",
    "dayofyear",
    "month",
    "weekday",
    "temp_range",
    "temp_max_lag1",
    "temp_max_lag7",
    "precip_lag1",
    "temp_rolling_7d"
]

TARGET_COL = "temperature_2m_max"
```

---

### Step 4.2 – Train the model (ml/train.py)

```python
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from etl.load import read_from_db
from ml.features import FEATURE_COLS, TARGET_COL

def train():
    df = read_from_db()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # time series: no shuffle!
    )

    mlflow.set_experiment("weather-forecast")

    with mlflow.start_run():
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})
        mlflow.sklearn.log_model(model, "model")

        print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")

        # Save model locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        print("Model saved to models/model.pkl")

if __name__ == "__main__":
    train()
```

Run training:
```bash
python -m ml.train
mlflow ui        # open http://localhost:5000 to inspect results
```

---

### Step 4.3 – Prediction helper (ml/predict.py)

```python
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
```

---

## Phase 5: FastAPI

### Step 5.1 – Build the API (api/main.py)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from ml.predict import predict

app = FastAPI(
    title="Weather Forecast API",
    description="Predicts next day maximum temperature for Berlin",
    version="1.0"
)

class WeatherInput(BaseModel):
    temperature_2m_min: float
    precipitation_sum: float
    windspeed_10m_max: float
    sunshine_duration: float
    dayofyear: int
    month: int
    weekday: int
    temp_range: float
    temp_max_lag1: float
    temp_max_lag7: float
    precip_lag1: float
    temp_rolling_7d: float

class PredictionOutput(BaseModel):
    predicted_temperature: float
    unit: str
    model_version: str
    timestamp: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Weather Forecast API is running"}

@app.get("/info")
def model_info():
    return {
        "model": "RandomForestRegressor",
        "target": "temperature_2m_max (next day)",
        "location": "Berlin, Germany",
        "data_source": "Open-Meteo API"
    }

@app.post("/predict", response_model=PredictionOutput)
def make_prediction(data: WeatherInput):
    try:
        result = predict(data.dict())
        return PredictionOutput(
            predicted_temperature=result,
            unit="°C",
            model_version="1.0",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

Start the API:
```bash
uvicorn api.main:app --reload
```

Open the interactive docs: http://localhost:8000/docs

---

## Phase 6: Docker

### Step 6.1 – Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run ETL + training on build, then start API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 6.2 – docker-compose.yml

```yaml
version: "3.9"

services:
  pipeline:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./mlruns:/app/mlruns
    environment:
      - PYTHONUNBUFFERED=1

  mlflow:
    image: python:3.11-slim
    command: >
      bash -c "pip install mlflow && mlflow ui --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
```

Build and run:
```bash
docker-compose up --build
```

---

## Phase 7: Tests

### Step 7.1 – tests/test_etl.py

```python
import pytest
import pandas as pd
from etl.transform import transform

def test_transform_adds_features():
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=30),
        "temperature_2m_max": [10.0] * 30,
        "temperature_2m_min": [5.0] * 30,
        "precipitation_sum": [0.0] * 30,
        "windspeed_10m_max": [10.0] * 30,
        "sunshine_duration": [3600.0] * 30,
    })
    result = transform(df)
    assert "dayofyear" in result.columns
    assert "temp_max_lag1" in result.columns
    assert "temp_rolling_7d" in result.columns
    assert result.isna().sum().sum() == 0
```

### Step 7.2 – tests/test_api.py

```python
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_info():
    response = client.get("/info")
    assert response.status_code == 200
    assert "model" in response.json()
```

Run tests:
```bash
pytest tests/ -v
```

---

## Phase 8: GitHub

### Step 8.1 – Push to GitHub

```bash
git add .
git commit -m "feat: initial end-to-end weather forecast pipeline"
git branch -M main
git remote add origin https://github.com/DKSW723/weather-pipeline.git
git push -u origin main
```

### Step 8.2 – GitHub Repository Checklist

- [ ] README.md with architecture diagram
- [ ] requirements.txt
- [ ] .gitignore (no .env, no .db, no .pkl committed)
- [ ] notebooks/01_EDA.ipynb with visualizations
- [ ] Clean, modular code (etl/, ml/, api/)
- [ ] Tests passing
- [ ] Docker running

---

## Suggested Improvements (after v1.0)

| Feature | Complexity | Impact |
|---|---|---|
| Switch SQLite to PostgreSQL | Low | Medium |
| Add XGBoost and compare with RF in MLflow | Low | High |
| Schedule ETL with Airflow or cron | Medium | High |
| Add authentication to API (API Key) | Medium | Medium |
| Deploy to AWS EC2 or Render.com | Medium | Very High |
| Add GitHub Actions CI/CD | Medium | High |
| Add monitoring / model drift detection | High | Very High |

---

## What This Project Demonstrates

| Role | What They See |
|---|---|
| **Data Engineer** | ETL pipeline, database, API integration |
| **Data Scientist** | EDA notebook, feature engineering, model evaluation |
| **ML Engineer** | MLflow tracking, pipeline, model serving |
| **AI Engineer** | End-to-end system design, API, containerization |
| **DevOps** | Docker, Docker Compose, reproducible builds |
