import pytest
import pandas as pd
from etl.transform import transform

def make_dummy_df(periods=30):
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=periods),
        "temperature_2m_max": [10.0] * periods,
        "temperature_2m_min": [5.0] * periods,
        "precipitation_sum": [0.0] * periods,
        "windspeed_10m_max": [10.0] * periods,
        "sunshine_duration": [3600.0] * periods,
    })

def test_transform_adds_features():
    df = make_dummy_df()
    result = transform(df)
    assert "dayofyear" in result.columns
    assert "month" in result.columns
    assert "temp_max_lag1" in result.columns
    assert "temp_rolling_7d" in result.columns

def test_transform_no_nulls():
    df = make_dummy_df()
    result = transform(df)
    assert result.isna().sum().sum() == 0

def test_transform_reduces_rows():
    df = make_dummy_df(30)
    result = transform(df)
    assert len(result) < 30  # Lag features entfernen erste Zeilen

def test_transform_adds_temp_range():
    df = make_dummy_df()
    result = transform(df)
    assert "temp_range" in result.columns
    assert (result["temp_range"] == 5.0).all()  # 10.0 - 5.0
