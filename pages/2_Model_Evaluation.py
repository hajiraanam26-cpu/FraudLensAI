
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Model Evaluation | FraudLens AI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Evaluation")

st.caption(
    "Comprehensive evaluation of fraud detection performance, "
    "classification quality, and decision thresholds."
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "transactions.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.joblib"
META_PATH = BASE_DIR / "models" / "metadata.json"

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    return {}


if not DATA_PATH.exists():
    st.error("Dataset not found: data/transactions.csv")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Trained model not found: models/fraud_model.joblib")
    st.stop()


try:
    df = load_data()
    model = load_model()
    metadata = load_metadata()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------

if "is_fraud" not in df.columns:
    st.error("Dataset must contain an 'is_fraud' column.")
    st.stop()

df["is_fraud"] = pd.to_numeric(
    df["is_fraud"],
    errors="coerce"
)

df = df.dropna(subset=["is_fraud"]).copy()
df["is_fraud"] = df["is_fraud"].astype(int)

if not set(df["is_fraud"].unique()).issubset({0, 1}):
    st.error("The target column must contain only 0 and 1.")
    st.stop()

if df["is_fraud"].nunique() != 2:
    st.error(
        "Evaluation requires both legitimate and fraud examples."
    )
    st.stop()


# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------

features = metadata.get("features")

if not features:
    features = [
        column for column in df.columns
        if column not in ["is_fraud", "transaction_id"]
    ]

missing_features = [
    feature for feature in features
    if feature not in df.columns
]

if missing_features:
    st.error(
        "Missing model features in dataset: "
        + ", ".join(missing_features)
    )
    st.stop()

X = df[features].copy()
y = df["is_fraud"].copy()

try:
    X = X.apply(pd.to_numeric, errors="raise")

except (ValueError, TypeError):
    st.error("Model features must be numeric.")
    st.stop()

if X.isnull().any().any():
    st.error(
        "Missing values found in model features. "
        "Clean the dataset first."
    )
    st.stop()


# --------------------------------------------------
# HELD-OUT TEST DATA
# --------------------------------------------------

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(metadata.get("test_size", 0.20)),
        random_state=int(metadata.get("random_state", 42)),
        stratify=y,
    )

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)

        if expected != list(X_test.columns):
            st.error(
                "The saved model's feature names do not match "
                "the dataset or metadata feature order."
            )
            st.stop()

    elif hasattr(model, "n_features_in_"):
        if model.n_features_in_ != X_test.shape[1]:
            st.error(
                "The saved model expects a different number "
                "of features."
            )
            st.stop()

    probabilities_all = model.predict_proba(X_test)

    model_classes = list(model.classes_)

    if 1 not in model_classes:
        st.error(
            "The trained model does not contain a fraud class (1)."
        )
        st.stop()

    fraud_index = model_classes.index(1)

    probabilities = probabilities_all[:, fraud_index]

except Exception as error:
    st.error(f"Evaluation failed: {error}")

    st.info(
        "Check that the saved model, metadata, and dataset "
        "come from the same training configuration."
    )

    st.stop()


# --------------------------------------------------
# THRESHOLD SELECTION
# --------------------------------------------------

st.divider()

st.header("🎚️ Fraud Classification Threshold")

st.write(
    "Adjust the probability threshold to control how sensitive "
    "the model is when identifying potentially fraudulent transactions."
)

threshold = st.slider(
    "Fraud probability threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
)

predictions = (
    probabilities >= threshold
).astype(int)


# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)

auc = roc_auc_score(
    y_test,
    probabilities,
)

avg_precision = average_precision_score(
    y_test,
    probabilities,
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1],
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0 else 0
)

fraud_detection_rate = recall


# --------------------------------------------------
# PERFORMANCE SUMMARY
# --------------------------------------------------

st.divider()

st.header("📈 Model Performance Summary")

st.caption(
    f"Evaluation threshold: {threshold:.2f} | "
    f"Test transactions: {len(y_test):,}"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{accuracy:.2%}")
c2.metric("Precision", f"{precision:.2%}")
c3.metric("Recall", f"{recall:.2%}")
c4.metric("F1-Score", f"{f1:.2%}")

c5, c6, c7, c8 = st.columns(4)

c5.metric("ROC-AUC", f"{auc:.3f}")
c6.metric("Average Precision", f"{avg_precision:.3f}")
c7.metric("Specificity", f"{specificity:.2%}")
c8.metric(
    "Fraud Detection Rate",
    f"{fraud_detection_rate:.2%}"
)

st.info(
    "Recall measures the proportion of actual fraud cases detected. "
    "Precision measures how many transactions flagged as fraud "
    "were actually fraud. The preferred threshold depends on the "
    "cost of missed fraud versus false alarms."
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.divider()

st.header("🔍 Confusion Matrix")

cm_df = pd.DataFrame(
    cm,
    index=["Actual Legitimate", "Actual Fraud"],
    columns=["Predicted Legitimate", "Predicted Fraud"],
)

fig_cm = px.imshow(
    cm_df,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Blues",
    labels={
        "x": "Predicted Class",
        "y": "Actual Class",
        "color": "Transactions",
    },
    title="Actual vs Predicted Transactions",
)

st.plotly_chart(
    fig_cm,
    use_container_width=True,
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("True Negatives", f"{tn:,}")
m2.metric("False Positives", f"{fp:,}")
m3.metric("False Negatives", f"{fn:,}")
m4.metric("True Positives", f"{tp:,}")

st.caption(
    "False negatives are missed fraud cases. False positives are "
    "legitimate transactions incorrectly flagged as fraud."
)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

st.divider()

st.header("📋 Detailed Classification Report")

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Legitimate", "Fraud"],
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(3),
    use_container_width=True,
)

st.download_button(
    "⬇️ Download Classification Report",
    data=report_df.to_csv().encode("utf-8"),
    file_name="fraudlens_classification_report.csv",
    mime="text/csv",
)


# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

st.divider()

st.header("📉 ROC Curve")

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    probabilities,
)

fig_roc = go.Figure()

fig_roc.add_trace(
    go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"FraudLens AI (AUC = {auc:.3f})",
    )
)

fig_roc.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(dash="dash"),
    )
)

fig_roc.update_layout(
    title="Receiver Operating Characteristic",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
)

st.plotly_chart(
    fig_roc,
    use_container_width=True,
)


# --------------------------------------------------
# PRECISION-RECALL CURVE
# --------------------------------------------------

st.divider()

st.header("🎯 Precision-Recall Curve")

pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
    y_test,
    probabilities,
)

fig_pr = go.Figure()

fig_pr.add_trace(
    go.Scatter(
        x=pr_recall,
        y=pr_precision,
        mode="lines",
        name="Precision-Recall",
    )
)

fig_pr.add_hline(
    y=float(y_test.mean()),
    line_dash="dash",
    annotation_text="Fraud prevalence baseline",
)

fig_pr.update_layout(
    title=f"Precision-Recall Curve (AP = {avg_precision:.3f})",
    xaxis_title="Recall",
    yaxis_title="Precision",
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
)

st.plotly_chart(
    fig_pr,
    use_container_width=True,
)

st.caption(
    "The precision-recall curve is particularly useful for fraud "
    "detection when the fraud class is rare."
)


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.divider()

st.header("🧠 Feature Importance")

importance_values = None
feature_names = X.columns

if hasattr(model, "feature_importances_"):

    importance_values = model.feature_importances_

elif hasattr(model, "named_steps"):

    final_estimator = list(model.named_steps.values())[-1]

    if hasattr(final_estimator, "feature_importances_"):
        importance_values = final_estimator.feature_importances_

    elif hasattr(final_estimator, "coef_"):
        importance_values = np.abs(
            final_estimator.coef_
        ).flatten()


if importance_values is not None:

    if len(importance_values) == len(feature_names):

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance_values,
        }).sort_values(
            "Importance",
            ascending=True,
        )

        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Global Model Feature Importance",
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True,
        )

        st.caption(
            "Global feature importance describes overall model behaviour. "
            "It does not explain an individual transaction. Use the SHAP "
            "page for transaction-level explanations."
        )

    else:
        st.info(
            "Feature importance dimensions do not match the input features."
        )

else:
    st.info(
        "Built-in feature importance is not available for this model."
    )


# --------------------------------------------------
# DOWNLOAD EVALUATION RESULTS
# --------------------------------------------------

st.divider()

st.header("📥 Export Evaluation Results")

evaluation_summary = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Average Precision",
        "Specificity",
        "Threshold",
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives",
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        auc,
        avg_precision,
        specificity,
        threshold,
        tn,
        fp,
        fn,
        tp,
    ],
})

st.dataframe(
    evaluation_summary,
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "⬇️ Download Evaluation Metrics",
    data=evaluation_summary.to_csv(index=False).encode("utf-8"),
    file_name="fraudlens_evaluation_metrics.csv",
    mime="text/csv",
)


# --------------------------------------------------
# SAVE REPORTS TO PROJECT
# --------------------------------------------------

st.divider()

st.header("💾 Save Evaluation Reports")

st.write(
    "Save the current threshold-based evaluation results "
    "to your local FraudLens AI reports folder."
)

if st.button(
    "💾 Save Current Evaluation Results",
    type="primary",
):

    saved_metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "average_precision": round(float(avg_precision), 4),
        "specificity": round(float(specificity), 4),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "test_rows": int(len(y_test)),
        "model_type": type(model).__name__,
    }

    try:

        with open(
            REPORT_DIR / "metrics.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                saved_metrics,
                file,
                indent=4,
            )

        saved_predictions = X_test.copy()

        saved_predictions.insert(
            0,
            "actual",
            y_test.to_numpy(),
        )

        saved_predictions["predicted"] = predictions

        saved_predictions["fraud_probability"] = probabilities

        saved_predictions.to_csv(
            REPORT_DIR / "test_predictions.csv",
            index=False,
        )

        st.success(
            "Evaluation reports saved successfully!"
        )

        st.write(
            "Saved files:"
        )

        st.code(
            "reports/metrics.json\n"
            "reports/test_predictions.csv"
        )

    except Exception as error:

        st.error(
            f"Unable to save evaluation reports: {error}"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "FraudLens AI | Model evaluation using held-out synthetic transaction data."
)

st.caption(
    "Important: This evaluation uses a reproducible random split, "
    "not an independent external test dataset. Synthetic results "
    "should not be interpreted as real-world banking performance."
)
