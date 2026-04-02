from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from ml.predict import predict

app = FastAPI(
    title="Weather Forecast API",
    description="Predicts next day maximum temperature for Berlin",
    version="1.0"
)

# Input Schema
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

# Output Schema
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
        "data_source": "Open-Meteo Archive API"
    }

@app.post("/predict", response_model=PredictionOutput)
def make_prediction(data: WeatherInput):
    try:
        result = predict(data.model_dump())
        return PredictionOutput(
            predicted_temperature=result,
            unit="°C",
            model_version="1.0",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))