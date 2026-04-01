# Weather Forecast ML Pipeline

A production-ready, end-to-end machine learning pipeline that fetches weather data from the Open-Meteo API, processes it through an ETL pipeline, trains a temperature forecasting model, and serves predictions via a FastAPI REST API — all containerized with Docker.

## Architecture

```
Open-Meteo API
      │
      ▼
 ETL Pipeline          ← fetch → transform → load into SQLite
      │
      ▼
 ML Training           ← feature engineering → train → MLflow tracking
      │
      ▼
 FastAPI REST API       ← serve predictions via /predict endpoint
      │
      ▼
 Docker Container       ← fully containerized & reproducible
```

## Tech Stack

| Layer | Technology |
|---|---|
| Data Source | Open-Meteo API (free, no key required) |
| ETL | Python, pandas, requests |
| Storage | SQLite (simple) / PostgreSQL (optional) |
| ML | scikit-learn (RandomForest, XGBoost) |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |
| Version Control | Git + GitHub |

## Project Structure

```
weather-pipeline/
│
├── data/
│   └── weather.db              # SQLite database (auto-created)
│
├── etl/
│   ├── __init__.py
│   ├── extract.py              # Fetch data from Open-Meteo API
│   ├── transform.py            # Clean and feature engineer
│   └── load.py                 # Load into SQLite
│
├── ml/
│   ├── __init__.py
│   ├── features.py             # Feature engineering
│   ├── train.py                # Model training + MLflow logging
│   └── predict.py              # Load model and run predictions
│
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI app
│
├── mlruns/                     # MLflow experiment tracking (auto-created)
│
├── models/                     # Saved model artifacts
│   └── model.pkl
│
├── notebooks/
│   └── 01_EDA.ipynb            # Exploratory Data Analysis
│
├── tests/
│   ├── test_etl.py
│   └── test_api.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/DKSW723/weather-pipeline.git
cd weather-pipeline
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the ETL pipeline

```bash
python -m etl.extract
python -m etl.transform
python -m etl.load
```

### 5. Train the model

```bash
python -m ml.train
```

### 6. Start the API

```bash
uvicorn api.main:app --reload
```

### 7. Run with Docker

```bash
docker-compose up --build
```

### 8. Test the API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"temperature_2m_max": 18.5, "precipitation_sum": 2.1, "windspeed_10m_max": 15.0, "dayofyear": 180}'
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/info` | Model info and last training date |
| POST | `/predict` | Predict next day temperature |

### Example Response

```json
{
  "predicted_temperature": 21.4,
  "unit": "°C",
  "model_version": "1.0",
  "timestamp": "2026-03-31T12:00:00"
}
```

---

## MLflow Tracking

Start the MLflow UI to compare experiments:

```bash
mlflow ui
```

Open: http://localhost:5000

Tracked metrics per run:
- MAE, RMSE, R²
- Feature importances
- Model parameters
- Training date and dataset size

---

## Step-by-Step Guide

See [GUIDE.md](GUIDE.md) for the full step-by-step development guide.

---

## Skills Demonstrated

- **Data Engineering:** API integration, ETL pipeline, database management
- **Data Science:** EDA, feature engineering, model selection, evaluation
- **ML Engineering:** MLflow tracking, model serialization, pipeline automation
- **Software Engineering:** REST API, Docker, modular code structure, testing
- **DevOps:** Containerization, Docker Compose, reproducible environments

---

## Author

**Dennis K. Sabatini Wien**
- GitHub: [github.com/DKSW723](https://github.com/DKSW723)
- LinkedIn: [linkedin.com/in/dennis-k-sabatini-wien](https://linkedin.com/in/dennis-k-sabatini-wien)
