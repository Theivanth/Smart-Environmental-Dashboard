from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "environmental_data.csv"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "validated_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
MODEL_CARD = PROJECT_ROOT / "models" / "model_card.json"
PREDICTION_LOG = PROJECT_ROOT / "outputs" / "prediction_log.csv"
FEATURES = [
    "Temperature", "Humidity", "WindSpeed",
    "PM2_5", "PM10", "NO2", "AQI",
]
TARGET = "Risk"
RANDOM_STATE = 42
