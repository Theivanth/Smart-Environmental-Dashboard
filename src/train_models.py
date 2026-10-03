import json
from pathlib import Path
import sys
from collections.abc import Callable, Mapping, Sequence

import pandas as pd
from joblib import dump
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FEATURES, MODEL_CARD, MODEL_PATH, RANDOM_STATE, TARGET


ModelParameters = Mapping[str, Mapping[str, object]]
ProgressCallback = Callable[[float, str], None]


def candidate_models(
    model_names: Sequence[str] = (
        "logistic_regression",
        "random_forest",
        "decision_tree",
        "knn",
    ),
    model_parameters: ModelParameters | None = None,
    random_state: int = RANDOM_STATE,
) -> dict[str, Pipeline]:
    parameters = model_parameters or {}
    available_models = {
        "logistic_regression": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=parameters.get("logistic_regression", {}).get("C", 1.0),
                max_iter=parameters.get("logistic_regression", {}).get("max_iter", 2000),
                class_weight=parameters.get("logistic_regression", {}).get(
                    "class_weight", "balanced"
                ),
                random_state=random_state,
            )),
        ]),
        "random_forest": lambda: Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=parameters.get("random_forest", {}).get(
                    "n_estimators", 300
                ),
                max_depth=parameters.get("random_forest", {}).get("max_depth", 8),
                min_samples_leaf=parameters.get("random_forest", {}).get(
                    "min_samples_leaf", 1
                ),
                class_weight=parameters.get("random_forest", {}).get(
                    "class_weight", "balanced"
                ),
                random_state=random_state,
                n_jobs=parameters.get("random_forest", {}).get("n_jobs", 1),
            )),
        ]),
        "decision_tree": lambda: Pipeline([
            ("model", DecisionTreeClassifier(
                criterion=parameters.get("decision_tree", {}).get(
                    "criterion", "gini"
                ),
                max_depth=parameters.get("decision_tree", {}).get("max_depth", 8),
                min_samples_leaf=parameters.get("decision_tree", {}).get(
                    "min_samples_leaf", 1
                ),
                class_weight=parameters.get("decision_tree", {}).get(
                    "class_weight", "balanced"
                ),
                random_state=random_state,
            )),
        ]),
        "knn": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=parameters.get("knn", {}).get("n_neighbors", 5),
                weights=parameters.get("knn", {}).get("weights", "uniform"),
                p=parameters.get("knn", {}).get("p", 2),
                n_jobs=parameters.get("knn", {}).get("n_jobs", 1),
            )),
        ]),
    }
    unknown_models = set(model_names).difference(available_models)
    if unknown_models:
        raise ValueError(f"Unknown model names: {sorted(unknown_models)}")
    if not model_names:
        raise ValueError("Select at least one model to train.")
    return {name: available_models[name]() for name in model_names}


def train_and_select(
    frame: pd.DataFrame,
    *,
    model_names: Sequence[str] = (
        "logistic_regression",
        "random_forest",
        "decision_tree",
        "knn",
    ),
    model_parameters: ModelParameters | None = None,
    test_size: float = 0.25,
    cv_folds: int = 4,
    random_state: int = RANDOM_STATE,
    progress_callback: ProgressCallback | None = None,
):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if TARGET not in frame or any(feature not in frame for feature in FEATURES):
        missing = [column for column in [*FEATURES, TARGET] if column not in frame]
        raise ValueError(f"Training data is missing required columns: {missing}")
    class_counts = frame[TARGET].value_counts()
    if len(class_counts) < 2:
        raise ValueError("Training data must contain at least two target classes.")
    if class_counts.min() < 2:
        raise ValueError(
            "Each target class needs at least two rows for a stratified holdout."
        )

    X = frame[FEATURES]
    y = frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    training_class_counts = y_train.value_counts()
    if training_class_counts.min() < cv_folds:
        raise ValueError(
            "Each target class must have at least as many training rows as "
            f"the number of cross-validation folds ({cv_folds})."
        )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = []
    models = candidate_models(model_names, model_parameters, random_state)
    cross_validation_progress = 0.8
    for index, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(
                cross_validation_progress * index / len(models),
                f"Cross-validating {name.replace('_', ' ')}...",
            )
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="f1_macro"
        )
        results.append({
            "name": name,
            "mean_f1_macro": float(scores.mean()),
            "std_f1_macro": float(scores.std()),
            "model": model,
        })
        if progress_callback:
            progress_callback(
                cross_validation_progress * (index + 1) / len(models),
                f"Finished cross-validation for {name.replace('_', ' ')}.",
            )
    best = max(results, key=lambda item: item["mean_f1_macro"])
    if progress_callback:
        progress_callback(0.85, f"Refitting the best model: {best['name']}...")
    best["model"].fit(X_train, y_train)
    if progress_callback:
        progress_callback(0.95, "Saving the trained model and model card...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(best["model"], MODEL_PATH)
    model_card = {
        "selected_model": best["name"],
        "cv_f1_macro": best["mean_f1_macro"],
        "features": FEATURES,
        "classes": sorted(y.unique().tolist()),
        "training_parameters": {
            "models": list(model_names),
            "model_parameters": model_parameters or {},
            "test_size": test_size,
            "cv_folds": cv_folds,
            "random_state": random_state,
        },
    }
    MODEL_CARD.write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    if progress_callback:
        progress_callback(1.0, "Model training and selection complete.")
    return best["model"], X_test, y_test, results