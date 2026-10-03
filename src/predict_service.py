from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
from joblib import load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FEATURES, MODEL_PATH, PREDICTION_LOG


def predict_risk(sensor_values: dict) -> dict:
    missing = set(FEATURES).difference(sensor_values)
    if missing:
        raise ValueError(f"Missing sensor fields: {sorted(missing)}")
    model = load(MODEL_PATH)
    sample = pd.DataFrame([sensor_values], columns=FEATURES)
    label = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    class_probabilities = {
        name: float(value)
        for name, value in zip(model.classes_, probabilities)
    }
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "risk": label,
        "confidence": max(class_probabilities.values()),
        "probabilities": class_probabilities,
    }
    log_prediction(sensor_values, result)
    return result
def log_prediction(sensor_values: dict, result: dict) -> None:
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {**sensor_values, **{
        "timestamp_utc": result["timestamp_utc"],
        "risk": result["risk"],
        "confidence": result["confidence"],
    }}
    pd.DataFrame([row]).to_csv(
        PREDICTION_LOG,
        mode="a",
        header=not PREDICTION_LOG.exists(),
        index=False,
    )