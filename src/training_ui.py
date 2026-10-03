from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    FEATURES,
    MODEL_CARD,
    MODEL_PATH,
    PREDICTION_LOG,
    PROCESSED_DATA,
    RAW_DATA,
    TARGET,
)
from src.data_pipeline import validate_environmental_data
from src.evaluate import evaluate_model
from src.train_models import train_and_select


MODEL_OPTIONS = {
    "Logistic regression": "logistic_regression",
    "Random forest": "random_forest",
    "Decision tree": "decision_tree",
    "K-nearest neighbors (KNN)": "knn",
}
OUTPUTS_DIR = PREDICTION_LOG.parent


def _load_training_data(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    if uploaded_file is not None:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file), uploaded_file.name
    if not RAW_DATA.is_file():
        st.error(f"Training data was not found: `{RAW_DATA}`. Upload a CSV to continue.")
        return None, str(RAW_DATA)
    return pd.read_csv(RAW_DATA), str(RAW_DATA)


def render_training_page() -> None:
    st.header("Train and evaluate a model")
    st.write(
        "Choose a dataset and training settings. A successful run updates the model "
        "used by the risk dashboard."
    )
    uploaded_file = st.file_uploader(
        "Training data (CSV; leave empty to use the project dataset)",
        type=["csv"],
        help=(
            "The CSV must include Temperature, Humidity, WindSpeed, PM2_5, PM10, "
            "NO2, AQI, and Risk columns."
        ),
    )

    try:
        raw_frame, source_name = _load_training_data(uploaded_file)
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as exc:
        st.error(f"Could not read the selected CSV: {exc}")
        return
    if raw_frame is None:
        return

    try:
        validated_frame = validate_environmental_data(
            raw_frame,
            required_columns=[*FEATURES, TARGET],
        )
    except (TypeError, ValueError) as exc:
        st.error(f"The selected dataset is not valid for training: {exc}")
        return

    metric_columns = st.columns(3)
    metric_columns[0].metric("Usable rows", f"{len(validated_frame):,}")
    metric_columns[1].metric("Risk classes", validated_frame[TARGET].nunique())
    metric_columns[2].metric("Source", Path(source_name).name)
    with st.expander("Preview validated training data"):
        st.dataframe(validated_frame.head(20), width="stretch")

    st.subheader("Training configuration")
    with st.form("training_configuration"):
        selected_labels = st.multiselect(
            "Models to compare",
            options=list(MODEL_OPTIONS),
            default=list(MODEL_OPTIONS),
            help="Each selected model is scored with stratified cross-validation.",
        )

        validation_col, cv_col, seed_col = st.columns(3)
        with validation_col:
            test_size = st.slider(
                "Holdout test fraction",
                min_value=0.10,
                max_value=0.40,
                value=0.25,
                step=0.05,
                help="Held-out fraction used for final evaluation, not model selection.",
            )
        with cv_col:
            cv_folds = st.slider(
                "Cross-validation folds",
                min_value=2,
                max_value=10,
                value=4,
                help="Higher fold counts increase training time and require more rows per class.",
            )
        with seed_col:
            random_state = st.number_input(
                "Random seed",
                min_value=0,
                max_value=2_147_483_647,
                value=42,
                step=1,
            )
        worker_count = st.select_slider(
            "Parallel workers for Random Forest and KNN",
            options=[1, 2, 4],
            value=1,
            help=(
                "One worker avoids process startup overhead on small datasets. "
                "Increase this for larger datasets if your machine has spare CPU."
            ),
        )

        st.markdown("#### Logistic regression")
        logistic_col1, logistic_col2, logistic_col3 = st.columns(3)
        with logistic_col1:
            logistic_c = st.slider(
                "Regularization strength (C)",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.01,
            )
        with logistic_col2:
            logistic_max_iter = st.slider(
                "Maximum iterations",
                min_value=100,
                max_value=5000,
                value=2000,
                step=100,
            )
        with logistic_col3:
            logistic_weight_label = st.selectbox(
                "Logistic class weighting",
                options=["Balanced", "None"],
                index=0,
            )

        st.markdown("#### Random forest")
        forest_col1, forest_col2, forest_col3, forest_col4 = st.columns(4)
        with forest_col1:
            forest_estimators = st.slider(
                "Number of trees",
                min_value=50,
                max_value=1000,
                value=300,
                step=50,
            )
        with forest_col2:
            forest_depth_value = st.slider(
                "Maximum tree depth (0 = unlimited)",
                min_value=0,
                max_value=30,
                value=8,
                step=1,
            )
        with forest_col3:
            forest_min_leaf = st.slider(
                "Minimum samples per leaf",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
            )
        with forest_col4:
            forest_weight_label = st.selectbox(
                "Forest class weighting",
                options=["Balanced", "None"],
                index=0,
            )

        st.markdown("#### Classification tree")
        tree_col1, tree_col2, tree_col3, tree_col4 = st.columns(4)
        with tree_col1:
            tree_criterion = st.selectbox(
                "Tree split criterion",
                options=["gini", "entropy", "log_loss"],
            )
        with tree_col2:
            tree_depth_value = st.slider(
                "Tree maximum depth (0 = unlimited)",
                min_value=0,
                max_value=30,
                value=8,
                step=1,
            )
        with tree_col3:
            tree_min_leaf = st.slider(
                "Tree minimum samples per leaf",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
            )
        with tree_col4:
            tree_weight_label = st.selectbox(
                "Tree class weighting",
                options=["Balanced", "None"],
                index=0,
            )

        st.markdown("#### K-nearest neighbors")
        knn_col1, knn_col2, knn_col3 = st.columns(3)
        with knn_col1:
            knn_neighbors = st.slider(
                "Number of neighbors",
                min_value=1,
                max_value=31,
                value=5,
                step=2,
            )
        with knn_col2:
            knn_weights_label = st.selectbox(
                "Neighbor weighting",
                options=["Uniform", "Distance"],
            )
        with knn_col3:
            knn_distance = st.selectbox(
                "Distance metric",
                options=["Euclidean", "Manhattan"],
            )

        submitted = st.form_submit_button(
            "Train, compare, and evaluate",
            type="primary",
            disabled=not selected_labels,
            width="stretch",
        )

    if submitted:
        selected_models = [MODEL_OPTIONS[label] for label in selected_labels]
        model_parameters = {
            "logistic_regression": {
                "C": logistic_c,
                "max_iter": logistic_max_iter,
                "class_weight": (
                    "balanced" if logistic_weight_label == "Balanced" else None
                ),
            },
            "random_forest": {
                "n_estimators": forest_estimators,
                "max_depth": forest_depth_value or None,
                "min_samples_leaf": forest_min_leaf,
                "class_weight": (
                    "balanced" if forest_weight_label == "Balanced" else None
                ),
                "n_jobs": worker_count,
            },
            "decision_tree": {
                "criterion": tree_criterion,
                "max_depth": tree_depth_value or None,
                "min_samples_leaf": tree_min_leaf,
                "class_weight": (
                    "balanced" if tree_weight_label == "Balanced" else None
                ),
            },
            "knn": {
                "n_neighbors": knn_neighbors,
                "weights": knn_weights_label.lower(),
                "p": 2 if knn_distance == "Euclidean" else 1,
                "n_jobs": worker_count,
            },
        }
        progress_bar = st.progress(0, text="Preparing the training run...")
        progress_text = st.empty()
        try:
            with st.status("Training in progress", expanded=True) as training_status:
                PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
                validated_frame.to_csv(PROCESSED_DATA, index=False)

                def report_progress(fraction: float, message: str) -> None:
                    progress_bar.progress(fraction, text=message)
                    progress_text.write(message)

                model, X_test, y_test, results = train_and_select(
                    validated_frame,
                    model_names=selected_models,
                    model_parameters=model_parameters,
                    test_size=test_size,
                    cv_folds=cv_folds,
                    random_state=int(random_state),
                    progress_callback=report_progress,
                )
                macro_f1, report = evaluate_model(
                    model,
                    X_test,
                    y_test,
                    output_dir=OUTPUTS_DIR,
                )
                training_status.update(
                    label="Training and evaluation complete",
                    state="complete",
                    expanded=False,
                )
            progress_bar.progress(1.0, text="Training complete.")
        except (ValueError, OSError) as exc:
            st.error(f"Training did not complete: {exc}")
            return

        result_rows = [
            {
                "Model": result["name"].replace("_", " ").title(),
                "Mean CV macro F1": result["mean_f1_macro"],
                "CV standard deviation": result["std_f1_macro"],
                "Selected": result["name"]
                == max(results, key=lambda item: item["mean_f1_macro"])["name"],
            }
            for result in results
        ]
        st.session_state["last_training_run"] = {
            "model": max(results, key=lambda item: item["mean_f1_macro"])["name"],
            "macro_f1": float(macro_f1),
            "report": report,
            "results": result_rows,
            "source": Path(source_name).name,
        }
        st.success(
            f"Saved the selected model to `{MODEL_PATH}` and its metadata to "
            f"`{MODEL_CARD}`."
        )

    previous_run = st.session_state.get("last_training_run")
    if previous_run:
        st.divider()
        st.subheader("Latest training results")
        score_col, model_col, data_col = st.columns(3)
        score_col.metric("Holdout macro F1", f"{previous_run['macro_f1']:.3f}")
        model_col.metric(
            "Selected model",
            previous_run["model"].replace("_", " ").title(),
        )
        data_col.metric("Training source", previous_run["source"])
        st.dataframe(previous_run["results"], width="stretch", hide_index=True)
        with st.expander("Detailed classification report"):
            st.code(previous_run["report"], language="text")
        confusion_matrix = OUTPUTS_DIR / "confusion_matrix.png"
        if confusion_matrix.is_file():
            st.image(str(confusion_matrix), caption="Latest holdout confusion matrix")
