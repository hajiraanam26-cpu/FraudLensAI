
import streamlit as st
import plotly.express as px

from utils.helpers import (
    load_data,
    load_model,
    add_predictions
)

st.set_page_config(
    page_title="FraudLens AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 FraudLens AI")
st.caption(
    "AI-Powered Explainable Fraud Detection & Risk Intelligence"
)

try:
    df = load_data()
    model = load_model()
    results = add_predictions(df, model)

except Exception as e:
    st.error(f"Unable to load application: {e}")
    st.stop()

# Sidebar
st.sidebar.title("🧠 FraudLens AI")
st.sidebar.success("Model Loaded Successfully")
st.sidebar.caption("Random Forest + SHAP")

st.sidebar.markdown("---")
st.sidebar.info(
    "Use the navigation menu to explore analytics, "
    "fraud alerts, SHAP explanations and model performance."
)

# KPI calculations
total = len(results)

actual_fraud = int(results["is_fraud"].sum())

legitimate = total - actual_fraud

fraud_rate = (
    actual_fraud / total * 100 if total else 0
)

predicted_fraud = int(
    results["prediction"].sum()
)

high_risk = int(
    (results["risk_level"] == "High Risk").sum()
)

st.subheader("📊 Executive Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Transactions", f"{total:,}")
c2.metric("Actual Fraud", f"{actual_fraud:,}")
c3.metric("Legitimate", f"{legitimate:,}")
c4.metric("Actual Fraud Rate", f"{fraud_rate:.2f}%")

st.markdown("---")

c5, c6, c7 = st.columns(3)

c5.metric("Model Fraud Predictions", f"{predicted_fraud:,}")
c6.metric("High Risk Transactions", f"{high_risk:,}")
c7.metric(
    "Average Fraud Probability",
    f"{results['fraud_probability'].mean() * 100:.2f}%"
)

st.markdown("---")

left, right = st.columns(2)

with left:
    st.subheader("Fraud vs Legitimate")

    pie_data = results["is_fraud"].value_counts()

    fig = px.pie(
        names=["Legitimate" if i == 0 else "Fraud"
               for i in pie_data.index],
        values=pie_data.values,
        hole=0.55,
        title="Actual Transaction Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Risk Distribution")

    risk_counts = (
        results["risk_level"]
        .value_counts()
        .reindex(
            ["Low Risk", "Medium Risk", "High Risk"],
            fill_value=0
        )
        .reset_index()
    )

    risk_counts.columns = ["Risk Level", "Transactions"]

    fig = px.bar(
        risk_counts,
        x="Risk Level",
        y="Transactions",
        color="Risk Level",
        title="Model-Assigned Risk Categories"
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

left, right = st.columns(2)

with left:
    st.subheader("Fraud Activity by Hour")

    hourly = results.groupby("hour").agg(
        total=("is_fraud", "size"),
        fraud=("is_fraud", "sum")
    ).reset_index()

    hourly["fraud_rate"] = (
        hourly["fraud"] / hourly["total"] * 100
    )

    fig = px.line(
        hourly,
        x="hour",
        y="fraud_rate",
        markers=True,
        title="Actual Fraud Rate by Hour"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Transaction Amount Analysis")

    fig = px.box(
        results,
        x="is_fraud",
        y="amount",
        color="is_fraud",
        title="Amount Distribution by Actual Class"
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("🚨 Recent High-Risk Transactions")

display_columns = [
    "transaction_id",
    "amount",
    "hour",
    "fraud_probability",
    "risk_level"
]

if "transaction_id" in results.columns:
    recent = (
        results[results["risk_level"] == "High Risk"]
        .sort_values("fraud_probability", ascending=False)
        [display_columns]
        .head(10)
    )

    st.dataframe(recent, use_container_width=True)

else:
    st.warning("Transaction ID column not found.")

st.caption(
    "FraudLens AI | Explainable Fraud Detection System"
)
