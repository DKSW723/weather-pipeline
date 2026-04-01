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