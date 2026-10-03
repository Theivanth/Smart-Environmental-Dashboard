from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)
def evaluate_model(model, X_test, y_test, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(X_test)
    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )
    (output_dir / "training_report.txt").write_text(
        f"Macro F1: {macro_f1:.3f}\n\n{report}",
        encoding="utf-8",
    )
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        cmap="Blues",
    )
    plt.title("Final Environmental Risk Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()
    return macro_f1, report