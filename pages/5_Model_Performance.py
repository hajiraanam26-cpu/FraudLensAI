
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import json

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Model Performance",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 Model Performance Dashboard")

st.write(
    "Evaluate FraudLens AI using held-out test-set predictions."
)

BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = (
    BASE_DIR / "reports" / "test_predictions.csv"
)

METRICS_PATH = (
    BASE_DIR / "reports" / "metrics.json"
)

if not PREDICTIONS_PATH.exists():

    st.warning(
        "Held-out test predictions were not found. "
        "Generate reports/test_predictions.csv from your "
        "model evaluation script before reporting metrics."
    )

    st.stop()

df = pd.read_csv(PREDICTIONS_PATH)

required = ["y_true", "y_pred", "y_prob"]

missing = [c for c in required if c not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

y_true = df["y_true"].astype(int)
y_pred = df["y_pred"].astype(int)
y_prob = df["y_prob"].astype(float)

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred, zero_division=0
)

recall = recall_score(
    y_true, y_pred, zero_division=0
)

f1 = f1_score(
    y_true, y_pred, zero_division=0
)

try:
    auc = roc_auc_score(y_true, y_prob)
except ValueError:
    auc = None

st.subheader("Evaluation Metrics")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Accuracy", f"{accuracy:.2%}")
c2.metric("Precision", f"{precision:.2%}")
c3.metric("Recall", f"{recall:.2%}")
c4.metric("F1 Score", f"{f1:.2%}")

c5.metric(
    "ROC-AUC",
    f"{auc:.4f}" if auc is not None else "N/A"
)

st.markdown("---")

left, right = st.columns(2)

with left:

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Predicted Legitimate", "Predicted Fraud"],
        y=["Actual Legitimate", "Actual Fraud"],
        colorscale="Blues",
        showscale=True
    )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    st.subheader("Classification Report")

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Legitimate", "Fraud"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

st.markdown("---")

if auc is not None:

    st.subheader("ROC Curve")

    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_prob
    )

    roc_data = pd.DataFrame({
        "False Positive Rate": fpr,
        "True Positive Rate": tpr
    })

    fig = px.line(
        roc_data,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"ROC Curve (AUC = {auc:.4f})"
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(dash="dash")
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

if METRICS_PATH.exists():

    with open(METRICS_PATH, "r") as file:
        metadata = json.load(file)

    with st.expander("Saved Model Evaluation Information"):
        st.json(metadata)

st.caption(
    "Metrics are calculated from held-out test predictions."
)
