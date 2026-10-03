import json

import pandas as pd

import src.train_models as train_models
from config import FEATURES


def test_training_uses_interactive_model_parameters(tmp_path, monkeypatch):
    row_count = 40
    frame = pd.DataFrame({
        feature: [float((index * (feature_index + 1)) % 17) for index in range(row_count)]
        for feature_index, feature in enumerate(FEATURES)
    })
    frame["Risk"] = ["Safe" if index % 2 == 0 else "Dangerous" for index in range(row_count)]
    monkeypatch.setattr(train_models, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(train_models, "MODEL_CARD", tmp_path / "model_card.json")
    progress_updates = []

    model, _, _, results = train_models.train_and_select(
        frame,
        model_names=[
            "logistic_regression",
            "random_forest",
            "decision_tree",
            "knn",
        ],
        model_parameters={
            "logistic_regression": {
                "C": 0.5,
                "max_iter": 400,
                "class_weight": None,
            },
            "random_forest": {
                "n_estimators": 20,
                "max_depth": 5,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "n_jobs": 2,
            },
            "decision_tree": {
                "criterion": "entropy",
                "max_depth": 4,
                "min_samples_leaf": 2,
                "class_weight": None,
            },
            "knn": {
                "n_neighbors": 3,
                "weights": "distance",
                "p": 1,
                "n_jobs": 2,
            },
        },
        test_size=0.2,
        cv_folds=2,
        random_state=7,
        progress_callback=lambda value, message: progress_updates.append(
            (value, message)
        ),
    )

    logistic = train_models.candidate_models(
        ["logistic_regression"],
        {"logistic_regression": {"C": 0.5, "max_iter": 400, "class_weight": None}},
        random_state=7,
    )["logistic_regression"].named_steps["model"]
    forest = train_models.candidate_models(
        ["random_forest"],
        {"random_forest": {
            "n_estimators": 20,
            "max_depth": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "n_jobs": 2,
        }},
        random_state=7,
    )["random_forest"].named_steps["model"]
    decision_tree = train_models.candidate_models(
        ["decision_tree"],
        {"decision_tree": {
            "criterion": "entropy",
            "max_depth": 4,
            "min_samples_leaf": 2,
            "class_weight": None,
        }},
        random_state=7,
    )["decision_tree"].named_steps["model"]
    knn = train_models.candidate_models(
        ["knn"],
        {"knn": {"n_neighbors": 3, "weights": "distance", "p": 1, "n_jobs": 2}},
        random_state=7,
    )["knn"].named_steps["model"]

    assert logistic.C == 0.5
    assert logistic.max_iter == 400
    assert logistic.class_weight is None
    assert forest.n_estimators == 20
    assert forest.max_depth == 5
    assert forest.min_samples_leaf == 2
    assert forest.n_jobs == 2
    assert decision_tree.criterion == "entropy"
    assert decision_tree.max_depth == 4
    assert decision_tree.min_samples_leaf == 2
    assert decision_tree.class_weight is None
    assert knn.n_neighbors == 3
    assert knn.weights == "distance"
    assert knn.p == 1
    assert knn.n_jobs == 2
    default_forest = train_models.candidate_models(["random_forest"])[
        "random_forest"
    ].named_steps["model"]
    assert default_forest.n_jobs == 1
    assert len(results) == 4
    assert progress_updates[-1][0] == 1.0
    assert train_models.MODEL_PATH.is_file()
    saved_card = json.loads(train_models.MODEL_CARD.read_text(encoding="utf-8"))
    assert saved_card["training_parameters"]["cv_folds"] == 2
    assert saved_card["training_parameters"]["random_state"] == 7
    assert model is not None
