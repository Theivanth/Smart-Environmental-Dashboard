from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import validate_environmental_data


def test_rejects_invalid_humidity():
    frame = pd.DataFrame([{
        "Temperature": 30,
        "Humidity": 120,
        "WindSpeed": 10,
        "PM2_5": 25,
        "PM10": 45,
        "NO2": 20,
        "AQI": 70,
        "Risk": "Safe",
    }])
    with pytest.raises(ValueError):
        validate_environmental_data(frame, list(frame.columns))
# tests/test_alerts.py
from src.alerts import create_environmental_alert
def test_dangerous_risk_creates_critical_alert():
    alert = create_environmental_alert(
        risk="Dangerous",
        confidence=0.91,
        aqi=175,
    )
    assert alert["level"] == "CRITICAL"
def test_low_confidence_requests_review():
    alert = create_environmental_alert(
        risk="Safe",
        confidence=0.42,
        aqi=75,
    )
    assert alert["level"] == "REVIEW"