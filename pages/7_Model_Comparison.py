
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Model Comparison | FraudLens AI",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Advanced Machine Learning Model Comparison")

st.write(
    "Compare machine-learning algorithms, handle class imbalance, "
    "tune Random Forest hyperparameters, and select a model "
    "using cross-validation."
)

st.info(
    "Model selection uses cross-validation on the training data. "
    "The held-out test set is reserved for final evaluation."
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "transactions.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


if not DATA_PATH.exists():
    st.error("Dataset not found: data/transactions.csv")
    st.stop()

try:
    df = load_data()

except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()


# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------

if "is_fraud" not in df.columns:
    st.error("Dataset must contain an 'is_fraud' column.")
    st.stop()

df["is_fraud"] = pd.to_numeric(
    df["is_fraud"],
    errors="coerce",
)

df = df.dropna(subset=["is_fraud"]).copy()

if not set(df["is_fraud"].unique()).issubset({0, 1}):
    st.error("The target column must contain only 0 and 1.")
    st.stop()

df["is_fraud"] = df["is_fraud"].astype(int)

if df["is_fraud"].nunique() != 2:
    st.error("Dataset must contain both fraud and legitimate examples.")
    st.stop()

X = df.drop(
    columns=["is_fraud", "transaction_id"],
    errors="ignore",
)

y = df["is_fraud"]

if X.empty:
    st.error("No model features found.")
    st.stop()

try:
    X = X.apply(pd.to_numeric, errors="raise")
except (ValueError, TypeError):
    st.error("All model features must be numeric.")
    st.stop()

if X.isnull().any().any():
    st.error("Missing values found. Clean the dataset before training.")
    st.stop()

if not np.isfinite(X.to_numpy(dtype=float)).all():
    st.error("Infinite values found in the model features.")
    st.stop()


# --------------------------------------------------
# CLASS DISTRIBUTION
# --------------------------------------------------

st.header("📊 Dataset Information")

total_count = len(df)
fraud_count = int((y == 1).sum())
legitimate_count = int((y == 0).sum())
fraud_percentage = fraud_count / total_count * 100

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Transactions", f"{total_count:,}")
c2.metric("Fraud Examples", f"{fraud_count:,}")
c3.metric("Legitimate Examples", f"{legitimate_count:,}")
c4.metric("Fraud Percentage", f"{fraud_percentage:.1f}%")

class_distribution = pd.DataFrame({
    "Transaction Type": ["Legitimate", "Fraud"],
    "Count": [legitimate_count, fraud_count],
})

fig_distribution = px.pie(
    class_distribution,
    names="Transaction Type",
    values="Count",
    title="Dataset Class Distribution",
    hole=0.45,
)

st.plotly_chart(
    fig_distribution,
    use_container_width=True,
)

st.caption(
    "Class imbalance can make accuracy misleading. "
    "Precision, recall, F1-score, and Average Precision "
    "are also considered during model evaluation."
)


# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

except ValueError as error:
    st.error(f"Unable to split the dataset: {error}")
    st.stop()


st.subheader("Training and Testing Split")

c1, c2, c3 = st.columns(3)

c1.metric("Training Samples", f"{len(X_train):,}")
c2.metric("Testing Samples", f"{len(X_test):,}")
c3.metric("Cross-Validation Folds", CV_FOLDS)

st.caption(
    "All models use the same stratified 80/20 split. "
    "Cross-validation is performed only on the training data."
)


# --------------------------------------------------
# CROSS-VALIDATION CONFIGURATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# --------------------------------------------------
# DEFINE BASELINE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
}


# --------------------------------------------------
# RANDOM FOREST HYPERPARAMETER SEARCH
# --------------------------------------------------

rf_base = RandomForestClassifier(
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_parameters = {
    "n_estimators": [100, 150, 200, 300],
    "max_depth": [None, 8, 12, 16],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}


# --------------------------------------------------
# TRAIN AND COMPARE
# --------------------------------------------------

if st.button(
    "🚀 Train, Tune and Compare Models",
    type="primary",
    use_container_width=True,
):

    results = []
    fitted_models = {}
    confusion_matrices = {}
    cv_details = {}

    progress = st.progress(0)
    status = st.empty()

    try:

        # ------------------------------------------
        # BASELINE MODEL CROSS-VALIDATION
        # ------------------------------------------

        for i, (name, model) in enumerate(models.items()):

            status.write(f"Cross-validating {name}...")

            scores = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring={
                    "f1": "f1",
                    "roc_auc": "roc_auc",
                    "average_precision": "average_precision",
                },
                n_jobs=-1,
                return_train_score=False,
            )

            cv_details[name] = {
                "CV F1 Mean": scores["test_f1"].mean(),
                "CV F1 Std": scores["test_f1"].std(),
                "CV ROC-AUC Mean": scores["test_roc_auc"].mean(),
                "CV Average Precision Mean": (
                    scores["test_average_precision"].mean()
                ),
            }

            # Fit using the complete training split
            model.fit(X_train, y_train)

            fitted_models[name] = model

            progress.progress((i + 1) / 4)

        # ------------------------------------------
        # RANDOM FOREST HYPERPARAMETER TUNING
        # ------------------------------------------

        status.write(
            "Tuning Random Forest hyperparameters. "
            "This may take a few minutes..."
        )

        search = RandomizedSearchCV(
            estimator=rf_base,
            param_distributions=rf_parameters,
            n_iter=8,
            scoring="f1",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )

        search.fit(X_train, y_train)

        tuned_rf = search.best_estimator_

        tuned_name = "Tuned Random Forest"

        fitted_models[tuned_name] = tuned_rf

        cv_details[tuned_name] = {
            "CV F1 Mean": search.best_score_,
            "CV F1 Std": search.cv_results_[
                "std_test_score"
            ][search.best_index_],
            "CV ROC-AUC Mean": np.nan,
            "CV Average Precision Mean": np.nan,
        }

        progress.progress(1.0)

        # ------------------------------------------
        # EVALUATE MODELS ON HELD-OUT TEST SET
        # ------------------------------------------

        status.write("Evaluating models on the held-out test set...")

        for name, model in fitted_models.items():

            predictions = model.predict(X_test)

            probabilities_all = model.predict_proba(X_test)

            fraud_index = list(model.classes_).index(1)

            fraud_probabilities = probabilities_all[:, fraud_index]

            result = {
                "Model": name,

                "CV F1 Mean": cv_details[name]["CV F1 Mean"],
                "CV F1 Std": cv_details[name]["CV F1 Std"],

                "Accuracy": accuracy_score(
                    y_test,
                    predictions,
                ),

                "Precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),

                "Recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),

                "F1 Score": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),

                "ROC-AUC": roc_auc_score(
                    y_test,
                    fraud_probabilities,
                ),

                "Average Precision": average_precision_score(
                    y_test,
                    fraud_probabilities,
                ),
            }

            results.append(result)

            confusion_matrices[name] = confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1],
            )

        # ------------------------------------------
        # SELECT MODEL USING CV F1
        # ------------------------------------------

        results_df = pd.DataFrame(results)

        best_name = max(
            cv_details,
            key=lambda name: cv_details[name]["CV F1 Mean"],
        )

        best_model = fitted_models[best_name]

        # ------------------------------------------
        # SAVE SELECTED MODEL
        # ------------------------------------------

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            best_model,
            MODEL_PATH,
        )

        metadata = {
            "model_name": best_name,
            "features": X.columns.tolist(),
            "target": "is_fraud",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
            "selection_metric": "cross_validation_f1",
            "cv_best_f1": float(
                cv_details[best_name]["CV F1 Mean"]
            ),
            "threshold": 0.50,
            "best_params": (
                search.best_params_
                if best_name == tuned_name
                else {}
            ),
        }

        with open(
            METADATA_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

        # ------------------------------------------
        # STORE RESULTS
        # ------------------------------------------

        st.session_state["comparison_results"] = results_df

        st.session_state["comparison_confusion"] = confusion_matrices

        st.session_state["comparison_cv_details"] = cv_details

        st.session_state["comparison_best_name"] = best_name

        st.session_state["comparison_best_params"] = (
            search.best_params_
        )

        st.session_state["comparison_metadata"] = metadata

        status.success("All models trained and evaluated successfully!")

        st.success(
            f"Selected model: {best_name}. "
            "Model and metadata saved successfully."
        )

    except Exception as error:

        status.empty()
        st.error(f"Model training failed: {error}")
        st.exception(error)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "comparison_results" in st.session_state:

    results_df = st.session_state["comparison_results"]

    cv_details = st.session_state["comparison_cv_details"]

    best_name = st.session_state["comparison_best_name"]

    best_params = st.session_state["comparison_best_params"]

    metadata = st.session_state["comparison_metadata"]

    st.divider()

    # ----------------------------------------------
    # CROSS-VALIDATION RESULTS
    # ----------------------------------------------

    st.header("🔄 Cross-Validation Results")

    cv_df = pd.DataFrame.from_dict(
        cv_details,
        orient="index",
    ).reset_index(names="Model")

    st.dataframe(
        cv_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The model is selected using mean F1-score across training "
        "cross-validation folds. Standard deviation indicates variation "
        "between folds."
    )

    # ----------------------------------------------
    # TEST SET COMPARISON
    # ----------------------------------------------

    st.header("🏆 Held-Out Test Set Comparison")

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Average Precision",
    ]

    display_df = results_df.copy()

    display_df[metric_columns] = (
        display_df[metric_columns] * 100
    ).round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Test-set results are reported after model selection. "
        "They are not used to select the winning model."
    )

    # ----------------------------------------------
    # PERFORMANCE CHART
    # ----------------------------------------------

    st.subheader("📈 Model Performance Visualization")

    chart_df = results_df.melt(
        id_vars="Model",
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Score",
    )

    chart_df["Score"] *= 100

    fig = px.bar(
        chart_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Model Performance on Held-Out Test Data",
        text_auto=".1f",
    )

    fig.update_layout(
        yaxis_title="Performance (%)",
        xaxis_title="Machine Learning Model",
        yaxis_range=[0, 100],
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ----------------------------------------------
    # SELECTED MODEL
    # ----------------------------------------------

    st.divider()

    st.header("🥇 Selected Model")

    selected_result = results_df[
        results_df["Model"] == best_name
    ].iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Selected Model",
        best_name,
    )

    c2.metric(
        "Cross-Validation F1",
        f"{metadata['cv_best_f1']:.2%}",
    )

    c3.metric(
        "Test F1",
        f"{selected_result['F1 Score']:.2%}",
    )

    st.success(
        f"{best_name} was selected using the highest mean "
        "cross-validation F1-score."
    )

    # ----------------------------------------------
    # RANDOM FOREST PARAMETERS
    # ----------------------------------------------

    if best_name == "Tuned Random Forest":

        st.subheader("⚙️ Best Random Forest Parameters")

        params_df = pd.DataFrame(
            list(best_params.items()),
            columns=["Parameter", "Selected Value"],
        )

        st.dataframe(
            params_df,
            use_container_width=True,
            hide_index=True,
        )

    # ----------------------------------------------
    # CONFUSION MATRIX
    # ----------------------------------------------

    st.divider()

    st.header("🔍 Confusion Matrix")

    selected_model = st.selectbox(
        "Choose a model to inspect",
        list(st.session_state["comparison_confusion"].keys()),
    )

    cm = st.session_state["comparison_confusion"][selected_model]

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Legitimate",
            "Actual Fraud",
        ],
        columns=[
            "Predicted Legitimate",
            "Predicted Fraud",
        ],
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"Confusion Matrix: {selected_model}",
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True,
    )

    tn, fp, fn, tp = cm.ravel()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("True Negatives", f"{tn:,}")
    c2.metric("False Positives", f"{fp:,}")
    c3.metric("False Negatives", f"{fn:,}")
    c4.metric("True Positives", f"{tp:,}")

    # ----------------------------------------------
    # DOWNLOAD REPORT
    # ----------------------------------------------

    st.divider()

    st.header("📥 Download Reports")

    st.download_button(
        "⬇️ Download Model Comparison CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="fraudlens_model_comparison.csv",
        mime="text/csv",
    )

    st.download_button(
        "⬇️ Download Cross-Validation CSV",
        data=cv_df.to_csv(index=False).encode("utf-8"),
        file_name="fraudlens_cross_validation.csv",
        mime="text/csv",
    )

    # ----------------------------------------------
    # SAVED MODEL INFORMATION
    # ----------------------------------------------

    st.divider()

    st.header("💾 Saved Model Information")

    st.success("Selected model saved successfully.")

    st.write(f"**Model:** {metadata['model_name']}")
    st.write(f"**Model file:** `{MODEL_PATH.name}`")
    st.write(f"**Metadata file:** `{METADATA_PATH.name}`")
    st.write(f"**Features:** {len(metadata['features'])}")
    st.write(f"**Selection metric:** {metadata['selection_metric']}")


else:

    st.info(
        "Click 'Train, Tune and Compare Models' to begin "
        "the advanced model training process."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "FraudLens AI | Advanced model comparison using synthetic transaction data."
)

st.caption(
    "Results are experimental. Cross-validation and held-out test "
    "performance do not establish real-world banking effectiveness."
)
