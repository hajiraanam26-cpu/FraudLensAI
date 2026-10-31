
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.title("💰 Cost-Based Fraud Analysis")

st.write(
    "Estimate the financial impact of fraud detection "
    "mistakes and compare different classification thresholds."
)

st.caption(
    "This dashboard uses the saved, untouched test predictions. "
    "All financial costs are illustrative assumptions."
)

st.divider()


# ==================================================
# 2. LOAD TEST PREDICTIONS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = (
    BASE_DIR / "reports" / "test_predictions.csv"
)


@st.cache_data
def load_predictions():

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Test predictions not found. "
            "Run python src/train_model.py first."
        )

    data = pd.read_csv(PREDICTIONS_PATH)

    required = [
        "actual",
        "fraud_probability",
        "amount",
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    for col in required:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna(
        subset=required
    ).copy()

    if data.empty:
        raise ValueError(
            "No valid test predictions found."
        )

    if not data["actual"].isin([0, 1]).all():
        raise ValueError(
            "Actual labels must be 0 or 1."
        )

    if not data["fraud_probability"].between(0, 1).all():
        raise ValueError(
            "Fraud probabilities must be between 0 and 1."
        )

    if (data["amount"] < 0).any():
        raise ValueError(
            "Transaction amounts cannot be negative."
        )

    if data["actual"].nunique() != 2:
        raise ValueError(
            "Both legitimate and fraud transactions are required."
        )

    return data.reset_index(drop=True)


try:
    df = load_predictions()

except Exception as e:
    st.error("Unable to load evaluation predictions.")
    st.exception(e)
    st.stop()


# ==================================================
# 3. COST ASSUMPTIONS
# ==================================================

st.header("⚙️ 1. Configure Financial Assumptions")

st.write(
    "Adjust the assumptions below to explore how the "
    "business impact changes."
)

col1, col2 = st.columns(2)

with col1:

    loss_rate = st.slider(
        "Estimated loss from undetected fraud (%)",
        min_value=10,
        max_value=100,
        value=100,
        step=5,
        help=(
            "Percentage of a missed fraudulent transaction "
            "amount treated as estimated financial loss."
        )
    )

with col2:

    review_cost = st.number_input(
        "Cost of reviewing one flagged transaction",
        min_value=0.0,
        max_value=100000.0,
        value=100.0,
        step=50.0,
        help=(
            "Illustrative manual investigation cost per "
            "flagged transaction, in the same units as "
            "the transaction amount."
        )
    )

loss_rate = loss_rate / 100

st.info(
    "These are adjustable assumptions, not measured banking "
    "losses. Transaction amounts and review costs must use "
    "consistent currency units."
)

st.divider()


# ==================================================
# 4. COST CALCULATION FUNCTION
# ==================================================

def calculate_costs(data, threshold, loss_rate, review_cost):

    actual = data["actual"].to_numpy().astype(int)

    probabilities = (
        data["fraud_probability"].to_numpy()
    )

    amounts = data["amount"].to_numpy()

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predictions,
        labels=[0, 1]
    ).ravel()

    missed_fraud_amount = float(
        amounts[
            (actual == 1) & (predictions == 0)
        ].sum()
    )

    flagged_count = int(
        predictions.sum()
    )

    missed_fraud_loss = (
        missed_fraud_amount * loss_rate
    )

    investigation_cost = (
        flagged_count * review_cost
    )

    total_cost = (
        missed_fraud_loss + investigation_cost
    )

    return {
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "missed_fraud_amount": missed_fraud_amount,
        "missed_fraud_loss": missed_fraud_loss,
        "flagged_count": flagged_count,
        "investigation_cost": investigation_cost,
        "total_cost": total_cost,
        "precision": precision_score(
            actual,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            actual,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            actual,
            predictions,
            zero_division=0
        ),
    }


# ==================================================
# 5. SELECTED THRESHOLD
# ==================================================

st.header("🎚️ 2. Select Fraud Detection Threshold")

threshold = st.slider(
    "Classification threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.01
)

result = calculate_costs(
    df,
    threshold,
    loss_rate,
    review_cost
)

st.divider()


# ==================================================
# 6. BUSINESS IMPACT OVERVIEW
# ==================================================

st.header("📊 3. Estimated Business Impact")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Estimated Missed Fraud Loss",
        f"{result['missed_fraud_loss']:,.2f}"
    )

with col2:
    st.metric(
        "Investigation Cost",
        f"{result['investigation_cost']:,.2f}"
    )

with col3:
    st.metric(
        "Total Estimated Cost",
        f"{result['total_cost']:,.2f}"
    )

st.caption(
    "Total estimated cost = estimated missed-fraud loss "
    "+ investigation cost for all flagged transactions."
)

st.divider()


# ==================================================
# 7. FRAUD DETECTION DETAILS
# ==================================================

st.header("🚨 4. Fraud Detection Breakdown")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "True Positives",
        result["tp"]
    )

with col2:
    st.metric(
        "False Positives",
        result["fp"]
    )

with col3:
    st.metric(
        "True Negatives",
        result["tn"]
    )

with col4:
    st.metric(
        "False Negatives",
        result["fn"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Precision",
        f"{result['precision'] * 100:.2f}%"
    )

with col2:
    st.metric(
        "Recall",
        f"{result['recall'] * 100:.2f}%"
    )

with col3:
    st.metric(
        "F1 Score",
        f"{result['f1'] * 100:.2f}%"
    )

st.divider()


# ==================================================
# 8. COST COMPONENTS CHART
# ==================================================

st.header("📉 5. Cost Distribution")

cost_df = pd.DataFrame({
    "Cost Type": [
        "Missed Fraud Loss",
        "Investigation Cost"
    ],
    "Estimated Cost": [
        result["missed_fraud_loss"],
        result["investigation_cost"]
    ]
})

fig_cost = px.bar(
    cost_df,
    x="Cost Type",
    y="Estimated Cost",
    text_auto=".2f",
    title="Estimated Cost by Category",
)

fig_cost.update_layout(
    height=400,
    yaxis_title="Estimated Cost Units"
)

st.plotly_chart(
    fig_cost,
    use_container_width=True
)

st.divider()


# ==================================================
# 9. COMPARE THRESHOLDS
# ==================================================

st.header("⚖️ 6. Compare Business Costs Across Thresholds")

thresholds = np.round(
    np.arange(0.30, 0.71, 0.05),
    2
)

comparison = []

for value in thresholds:

    metrics = calculate_costs(
        df,
        float(value),
        loss_rate,
        review_cost
    )

    comparison.append({
        "Threshold": value,
        "Missed Fraud Loss": metrics["missed_fraud_loss"],
        "Investigation Cost": metrics["investigation_cost"],
        "Total Estimated Cost": metrics["total_cost"],
        "False Positives": metrics["fp"],
        "False Negatives": metrics["fn"],
        "Precision (%)": metrics["precision"] * 100,
        "Recall (%)": metrics["recall"] * 100,
    })


comparison_df = pd.DataFrame(comparison)

display_df = comparison_df.copy()

cost_columns = [
    "Missed Fraud Loss",
    "Investigation Cost",
    "Total Estimated Cost"
]

for column in cost_columns:
    display_df[column] = display_df[column].round(2)

for column in [
    "Precision (%)",
    "Recall (%)"
]:
    display_df[column] = display_df[column].round(2)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.divider()


# ==================================================
# 10. COST VS THRESHOLD GRAPH
# ==================================================

st.header("📈 7. Threshold vs Estimated Cost")

fig_threshold = px.line(
    comparison_df,
    x="Threshold",
    y=[
        "Missed Fraud Loss",
        "Investigation Cost",
        "Total Estimated Cost"
    ],
    markers=True,
    title="Estimated Business Cost at Different Thresholds"
)

fig_threshold.update_layout(
    height=500,
    xaxis_title="Classification Threshold",
    yaxis_title="Estimated Cost Units"
)

st.plotly_chart(
    fig_threshold,
    use_container_width=True
)

st.divider()


# ==================================================
# 11. LOWEST COST THRESHOLD
# ==================================================

st.header("🏆 8. Lowest-Cost Threshold in Comparison")

best_row = comparison_df.loc[
    comparison_df["Total Estimated Cost"].idxmin()
]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Lowest Estimated Cost",
        f"{best_row['Total Estimated Cost']:,.2f}"
    )

with col2:
    st.metric(
        "Corresponding Threshold",
        f"{best_row['Threshold']:.2f}"
    )

st.caption(
    "This is the lowest estimated cost among the tested "
    "thresholds under your selected assumptions. It is not "
    "automatically a validated deployment threshold."
)

st.divider()


# ==================================================
# 12. DOWNLOAD REPORT
# ==================================================

st.header("📥 9. Download Cost Analysis")

csv_data = comparison_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "Download Cost Analysis CSV",
    data=csv_data,
    file_name="FraudLens_AI_Cost_Analysis.csv",
    mime="text/csv",
    use_container_width=True
)


# ==================================================
# 13. INTERPRETATION
# ==================================================

st.header("💡 10. Business Interpretation")

st.markdown("""
- **False negatives:** Fraudulent transactions that the model
  fails to identify. Their transaction amounts contribute to
  estimated missed-fraud loss.

- **False positives:** Legitimate transactions incorrectly
  flagged by the model. They contribute to review workload.

- **Investigation cost:** Applied to every flagged transaction,
  including both true positives and false positives.

- **Threshold selection:** A lower threshold may reduce missed
  fraud but increase investigation workload. A higher threshold
  may reduce investigations but allow more fraud to pass.

The lowest-cost threshold depends on the financial assumptions,
the dataset and the organization's risk tolerance.
""")

st.warning(
    "This is an educational cost model using synthetic data. "
    "It does not account for fraud recovery, chargebacks, "
    "customer churn, operational capacity or other real-world costs."
)

st.divider()

st.caption(
    "FraudLens AI | Cost-Based Fraud Analysis"
)
