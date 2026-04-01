import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_weather(latitude=52.52, longitude=13.41, days=365):
    end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")  # gestern, nicht heute
    start_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"  # ← geändert
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
    print(f"Fetched {len(df)} rows from Open-Meteo Archive API.")
    return df

if __name__ == "__main__":
    df = fetch_weather()
    print(df.head())