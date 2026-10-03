from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_training_interface_renders_parameter_controls():
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)

    assert not app.exception
    slider_labels = {slider.label for slider in app.get("slider")}
    assert "Holdout test fraction" in slider_labels
    assert "Cross-validation folds" in slider_labels
    assert "Number of trees" in slider_labels
    assert "Tree maximum depth (0 = unlimited)" in slider_labels
    assert "Number of neighbors" in slider_labels
    worker_selector = next(
        selector
        for selector in app.get("select_slider")
        if selector.label == "Parallel workers for Random Forest and KNN"
    )
    assert worker_selector.value == 1
    assert any(button.label == "Train, compare, and evaluate" for button in app.button)
    model_selector = next(
        selector for selector in app.multiselect if selector.label == "Models to compare"
    )
    assert set(model_selector.options) == {
        "Logistic regression",
        "Random forest",
        "Decision tree",
        "K-nearest neighbors (KNN)",
    }
