from pathlib import Path
import pandas as pd

VALID_RISKS = {"Safe", "Moderate", "Dangerous"}

def validate_environmental_data(
    frame: pd.DataFrame,
    required_columns: list[str],
) -> pd.DataFrame:
    missing = set(required_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    clean = frame.copy()
    if clean[required_columns].isna().any().any():
        raise ValueError("Missing values found in required columns")
    rules = {
        "Temperature": clean["Temperature"].between(-20, 65),
        "Humidity": clean["Humidity"].between(0, 100),
        "WindSpeed": clean["WindSpeed"].ge(0),
        "PM2_5": clean["PM2_5"].ge(0),
        "PM10": clean["PM10"].ge(0),
        "NO2": clean["NO2"].ge(0),
        "AQI": clean["AQI"].ge(0),
        "Risk": clean["Risk"].isin(VALID_RISKS),
    }
    invalid_rows = ~pd.DataFrame(rules).all(axis=1)
    if invalid_rows.any():
        bad_indices = clean.index[invalid_rows].tolist()
        raise ValueError(f"Invalid sensor rows: {bad_indices}")
    return clean.drop_duplicates().reset_index(drop=True)