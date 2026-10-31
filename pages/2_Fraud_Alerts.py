
import streamlit as st
import pandas as pd

from utils.helpers import (
    load_model,
    load_data,
    get_fraud_probability,
    classify_risk
)

st.set_page_config(
    page_title="Fraud Alerts",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Fraud Detection Alerts")
st.write("Identify and investigate suspicious transactions.")

model = load_model()
df = load_data()

threshold = st.sidebar.slider(
    "Fraud Alert Threshold",
    0.10,
    0.95,
    0.50,
    0.05
)

probabilities = get_fraud_probability(model, df)

df = df.copy()

df["fraud_probability"] = probabilities

df["prediction"] = (
    df["fraud_probability"] >= threshold
).astype(int)

df["risk_level"] = df["fraud_probability"].apply(
    lambda p: (
        "Low Risk" if p < 0.30
        else "Medium Risk" if p < 0.70
        else "High Risk"
    )
)

alerts = df[df["prediction"] == 1].copy()

high = (df["risk_level"] == "High Risk").sum()
medium = (df["risk_level"] == "Medium Risk").sum()
low = (df["risk_level"] == "Low Risk").sum()

c1, c2, c3 = st.columns(3)

c1.metric("High Risk", int(high))
c2.metric("Medium Risk", int(medium))
c3.metric("Low Risk", int(low))

st.markdown("---")

st.subheader("🚨 Suspicious Transactions")

if len(alerts) > 0:

    alerts = alerts.sort_values(
        "fraud_probability",
        ascending=False
    )

    st.dataframe(
        alerts,
        use_container_width=True
    )

    csv = alerts.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Fraud Alerts",
        csv,
        "fraud_alerts.csv",
        "text/csv"
    )

else:
    st.success("No transactions crossed the selected threshold.")

st.markdown("---")

st.subheader("🔍 Test an Individual Transaction")

with st.form("transaction_form"):

    c1, c2 = st.columns(2)

    with c1:
        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0
        )

        hour = st.slider(
            "Transaction Hour",
            0,
            23,
            12
        )

        distance = st.number_input(
            "Distance from Usual Location (km)",
            min_value=0.0,
            value=5.0
        )

    with c2:
        failed = st.number_input(
            "Failed Attempts",
            min_value=0,
            value=0,
            step=1
        )

        international = st.selectbox(
            "International Transaction",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    submitted = st.form_submit_button(
        "Analyze Transaction",
        use_container_width=True
    )

if submitted:

    transaction = pd.DataFrame([{
        "amount": amount,
        "hour": hour,
        "distance_from_usual_km": distance,
        "failed_attempts": failed,
        "international": international
    }])

    probability = get_fraud_probability(
        model,
        transaction
    )[0]

    risk = classify_risk(probability)

    prediction = int(probability >= threshold)

    st.subheader("Risk Assessment")

    st.metric(
        "Fraud Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    if risk == "High Risk":
        st.error(f"🚨 {risk}")

    elif risk == "Medium Risk":
        st.warning(f"⚠️ {risk}")

    else:
        st.success(f"✅ {risk}")

    st.write(
        "Prediction:",
        "Potential Fraud" if prediction else "Likely Legitimate"
    )

    st.caption(
        "Risk category and fraud classification use separate "
        "thresholds. This is a model assessment, not a confirmed fraud."
    )
