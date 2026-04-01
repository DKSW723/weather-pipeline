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
