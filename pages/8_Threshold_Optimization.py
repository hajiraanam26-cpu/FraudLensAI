
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.title("⚖️ Fraud Risk Threshold Optimization")

st.write(
    "Explore how changing the classification threshold "
    "affects FraudLens AI's fraud detection performance."
)

st.caption(
    "This dashboard uses saved predictions from the "
    "1,000-transaction evaluation dataset."
)

st.divider()


# ==================================================
# LOAD TEST PREDICTIONS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = (
    BASE_DIR / "reports" / "test_predictions.csv"
)


@st.cache_data
def load_predictions():

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "reports/test_predictions.csv was not found. "
            "Please generate the evaluation predictions first."
        )

    data = pd.read_csv(PREDICTIONS_PATH)

    required_columns = [
        "actual",
        "fraud_probability"
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data["actual"] = pd.to_numeric(
        data["actual"],
        errors="coerce"
    )

    data["fraud_probability"] = pd.to_numeric(
        data["fraud_probability"],
        errors="coerce"
    )

    data = data.dropna(
        subset=required_columns
    ).copy()

    if data.empty:
        raise ValueError(
            "No valid prediction records were found."
        )

    if not data["actual"].isin([0, 1]).all():
        raise ValueError(
            "Actual labels must contain only 0 and 1."
        )

    if not data["fraud_probability"].between(0, 1).all():
        raise ValueError(
            "Fraud probabilities must be between 0 and 1."
        )

    if data["actual"].nunique() != 2:
        raise ValueError(
            "The evaluation dataset must contain both classes."
        )

    return data.reset_index(drop=True)


try:
    df = load_predictions()

except Exception as e:
    st.error("Unable to load evaluation predictions.")
    st.exception(e)
    st.stop()


y_true = df["actual"].astype(int).to_numpy()
probabilities = df["fraud_probability"].to_numpy()


# ==================================================
# METRIC CALCULATION FUNCTION
# ==================================================

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# ==================================================
# 1. DATASET OVERVIEW
# ==================================================

st.header("📊 1. Evaluation Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Transactions",
        f"{len(df):,}"
    )

with col2:
    fraud_count = int((y_true == 1).sum())

    st.metric(
        "Actual Fraud Transactions",
        fraud_count
    )

with col3:
    legitimate_count = int((y_true == 0).sum())

    st.metric(
        "Actual Legitimate Transactions",
        legitimate_count
    )

st.divider()


# ==================================================
# 2. INTERACTIVE THRESHOLD SLIDER
# ==================================================

st.header("🎚️ 2. Adjust Fraud Detection Threshold")

st.write(
    "Choose a threshold to control how easily a transaction "
    "is classified as fraudulent."
)

threshold = st.slider(
    "Fraud Classification Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.01,
    help=(
        "Transactions with fraud probability greater than "
        "or equal to this value are classified as fraud."
    )
)

current = calculate_metrics(
    y_true,
    probabilities,
    threshold
)

st.info(
    f"Transactions with fraud probability ≥ "
    f"**{threshold:.2f}** are classified as fraudulent."
)

st.divider()


# ==================================================
# 3. CURRENT THRESHOLD PERFORMANCE
# ==================================================

st.header("📈 3. Performance at Selected Threshold")

metric_columns = st.columns(3)

display_metrics = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1 Score", "f1"),
    ("Specificity", "specificity")
]

for index, (label, key) in enumerate(display_metrics):

    with metric_columns[index % 3]:
        st.metric(
            label,
            f"{current[key] * 100:.2f}%"
        )

st.divider()


# ==================================================
# 4. DYNAMIC CONFUSION MATRIX
# ==================================================

st.header("🎯 4. Confusion Matrix at Selected Threshold")

st.write(
    "See how many legitimate and fraudulent transactions "
    "are correctly or incorrectly classified."
)

cm = [
    [current["tn"], current["fp"]],
    [current["fn"], current["tp"]]
]

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual Legitimate",
        "Actual Fraud"
    ],
    columns=[
        "Predicted Legitimate",
        "Predicted Fraud"
    ]
)

fig_cm = px.imshow(
    cm_df,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Blues",
    title=f"Confusion Matrix — Threshold {threshold:.2f}"
)

fig_cm.update_layout(
    height=400,
    xaxis_title="Predicted Class",
    yaxis_title="Actual Class"
)

st.plotly_chart(
    fig_cm,
    use_container_width=True
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "True Positives",
        current["tp"]
    )

    st.metric(
        "False Negatives",
        current["fn"]
    )

with col2:
    st.metric(
        "True Negatives",
        current["tn"]
    )

    st.metric(
        "False Positives",
        current["fp"]
    )

st.divider()


# ==================================================
# 5. THRESHOLD COMPARISON
# ==================================================

st.header("📊 5. Compare Different Thresholds")

st.write(
    "Compare how different classification thresholds "
    "affect precision, recall, F1-score and accuracy."
)

comparison_thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]

comparison_results = []

for value in comparison_thresholds:

    result = calculate_metrics(
        y_true,
        probabilities,
        value
    )

    comparison_results.append({
        "Threshold": value,
        "Accuracy": result["accuracy"],
        "Precision": result["precision"],
        "Recall": result["recall"],
        "F1 Score": result["f1"],
        "Specificity": result["specificity"],
        "True Positives": result["tp"],
        "False Positives": result["fp"],
        "False Negatives": result["fn"]
    })


comparison_df = pd.DataFrame(
    comparison_results
)

# Display percentages in the table
display_comparison = comparison_df.copy()

percentage_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "Specificity"
]

for column in percentage_columns:

    display_comparison[column] = (
        display_comparison[column] * 100
    ).round(2)

st.dataframe(
    display_comparison,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Performance values in the table are displayed as percentages."
)

st.divider()


# ==================================================
# 6. THRESHOLD PERFORMANCE GRAPH
# ==================================================

st.header("📉 6. Threshold vs Model Performance")

chart_df = comparison_df.melt(
    id_vars=["Threshold"],
    value_vars=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Specificity"
    ],
    var_name="Metric",
    value_name="Score"
)

fig_performance = px.line(
    chart_df,
    x="Threshold",
    y="Score",
    color="Metric",
    markers=True,
    title="Performance Metrics Across Thresholds"
)

fig_performance.update_layout(
    height=500,
    xaxis_title="Classification Threshold",
    yaxis_title="Metric Score",
    yaxis=dict(
        range=[0, 1]
    )
)

st.plotly_chart(
    fig_performance,
    use_container_width=True
)

st.divider()


# ==================================================
# 7. PRECISION VS RECALL
# ==================================================

st.header("⚖️ 7. Precision vs Recall Trade-off")

st.write(
    "A lower threshold may identify more fraudulent transactions "
    "but can also increase false alarms. A higher threshold may "
    "reduce false alarms but miss more fraud."
)

fig_tradeoff = go.Figure()

fig_tradeoff.add_trace(
    go.Scatter(
        x=comparison_df["Threshold"],
        y=comparison_df["Precision"],
        mode="lines+markers",
        name="Precision"
    )
)

fig_tradeoff.add_trace(
    go.Scatter(
        x=comparison_df["Threshold"],
        y=comparison_df["Recall"],
        mode="lines+markers",
        name="Recall"
    )
)

fig_tradeoff.add_vline(
    x=threshold,
    line_dash="dash",
    annotation_text=f"Selected: {threshold:.2f}"
)

fig_tradeoff.update_layout(
    title="Precision and Recall at Different Thresholds",
    xaxis_title="Classification Threshold",
    yaxis_title="Score",
    yaxis=dict(range=[0, 1]),
    height=450
)

st.plotly_chart(
    fig_tradeoff,
    use_container_width=True
)

st.divider()


# ==================================================
# 8. BEST F1 THRESHOLD IN COMPARISON GRID
# ==================================================

st.header("🏆 8. Best F1 Threshold in Comparison")

best_row = comparison_df.loc[
    comparison_df["F1 Score"].idxmax()
]

st.metric(
    "Highest F1 Score in Tested Thresholds",
    f"{best_row['F1 Score'] * 100:.2f}%",
    help="Highest F1 among the nine thresholds in the comparison table."
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Corresponding Threshold",
        f"{best_row['Threshold']:.2f}"
    )

with col2:
    st.metric(
        "Fraud Transactions Detected",
        int(best_row["True Positives"])
    )

st.caption(
    "This is an exploratory result on the saved evaluation dataset, "
    "not an independently validated optimal deployment threshold."
)

st.divider()


# ==================================================
# 9. DOWNLOAD THRESHOLD REPORT
# ==================================================

st.header("📥 9. Download Threshold Comparison Report")

csv_data = comparison_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Threshold Comparison CSV",
    data=csv_data,
    file_name="FraudLens_AI_Threshold_Comparison.csv",
    mime="text/csv",
    use_container_width=True
)


# ==================================================
# 10. INTERPRETATION
# ==================================================

st.header("💡 10. How to Interpret Thresholds")

st.markdown("""
**Lower threshold (for example, 0.30):**
- More transactions may be flagged as fraud.
- Fraud recall may increase.
- False positives may also increase.

**Default threshold (0.50):**
- Uses the conventional 50% probability cutoff.
- Provides a useful baseline for comparison.

**Higher threshold (for example, 0.70):**
- Fewer transactions may be flagged as fraud.
- False positives may decrease.
- More fraudulent transactions may be missed.

The best operating threshold depends on the relative cost
of missed fraud and unnecessary transaction reviews.
""")

st.warning(
    "Threshold comparisons use the saved evaluation predictions. "
    "For a final unbiased performance estimate, select the threshold "
    "using a separate validation dataset and evaluate it once on an "
    "untouched test dataset."
)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "FraudLens AI | Fraud Risk Threshold Optimization"
)

st.caption(
    "Changing the threshold changes classification decisions, "
    "not the trained model parameters."
)
