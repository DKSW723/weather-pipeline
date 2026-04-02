from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_info():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "location" in data

def test_predict_returns_temperature():
    payload = {
        "temperature_2m_min": 5.0,
        "precipitation_sum": 0.0,
        "windspeed_10m_max": 12.0,
        "sunshine_duration": 3600.0,
        "dayofyear": 90,
        "month": 3,
        "weekday": 1,
        "temp_range": 8.0,
        "temp_max_lag1": 13.0,
        "temp_max_lag7": 11.0,
        "precip_lag1": 0.0,
        "temp_rolling_7d": 12.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_temperature" in data
    assert isinstance(data["predicted_temperature"], float)
    assert -30 < data["predicted_temperature"] < 50  # plausible Temperatur

def test_predict_missing_field_returns_422():
    payload = {"temperature_2m_min": 5.0}  # fast alles fehlt
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
