from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FEATURES, PROCESSED_DATA, RAW_DATA, TARGET
from src.data_pipeline import validate_environmental_data
from src.evaluate import evaluate_model
from src.train_models import train_and_select


def main():
    raw = pd.read_csv(RAW_DATA)
    validated = validate_environmental_data(
        raw,
        required_columns=FEATURES + [TARGET],
    )
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(PROCESSED_DATA, index=False)
    model, X_test, y_test, results = train_and_select(validated)
    macro_f1, report = evaluate_model(
        model,
        X_test,
        y_test,
        output_dir=RAW_DATA.parents[2] / "outputs",
    )
    print("Training complete")
    print("Final macro F1:", macro_f1)
    print(report)
if __name__ == "__main__":
    main()