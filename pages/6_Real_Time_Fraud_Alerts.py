
from pathlib import Path
from datetime import datetime
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "fraud_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"


# --------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Real-Time Fraud Alerts | FraudLens AI",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Fraud Monitoring & Alert Center")

st.caption(
    "Interactive transaction monitoring using the trained "
    "FraudLens AI machine-learning model."
)


# --------------------------------------------------
# 3. LOAD MODEL AND METADATA
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as file:
            return json.load(file)

    return {}


try:
    model = load_model()
    metadata = load_metadata()

except Exception as e:
    st.error(f"Unable to load model or metadata: {e}")
    st.stop()


if not hasattr(model, "predict_proba"):
    st.error("The loaded model does not support probability predictions.")
    st.stop()


# --------------------------------------------------
# 4. VALIDATE MODEL FEATURES
# --------------------------------------------------

input_features = [
    "amount",
    "hour",
    "distance_from_usual_km",
    "failed_attempts",
    "international"
]

expected_features = metadata.get("features", input_features)

missing_features = [
    feature for feature in expected_features
    if feature not in input_features
]

if missing_features:
    st.error(
        "The trained model expects additional features that are not "
        f"available in this form: {missing_features}"
    )
    st.info(
        "The transaction form must provide every feature used "
        "during model training."
    )
    st.stop()


if len(expected_features) != len(input_features):
    st.error(
        "The model feature configuration does not match this "
        "transaction form. Please check metadata.json."
    )
    st.stop()


# --------------------------------------------------
# 5. IDENTIFY FRAUD CLASS
# --------------------------------------------------

classes = list(
    getattr(model, "classes_", [])
)

if 1 not in classes:
    st.error("Fraud class 1 was not found in the trained model.")
    st.stop()

fraud_index = classes.index(1)


# --------------------------------------------------
# 6. SESSION STATE
# --------------------------------------------------

if "alert_history" not in st.session_state:
    st.session_state.alert_history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# --------------------------------------------------
# 7. ALERT CONFIGURATION
# --------------------------------------------------

st.divider()
st.subheader("⚙️ Risk Alert Configuration")

col1, col2 = st.columns(2)

with col1:
    medium_threshold = st.slider(
        "Medium Risk Threshold (%)",
        min_value=10,
        max_value=80,
        value=30,
        step=5
    )

with col2:
    high_threshold = st.slider(
        "High Risk Threshold (%)",
        min_value=20,
        max_value=95,
        value=60,
        step=5
    )

if medium_threshold >= high_threshold:
    st.warning(
        "High Risk Threshold must be greater than "
        "Medium Risk Threshold."
    )
    st.stop()


st.markdown(
    f"""
    - 🟢 **Low Risk:** Below {medium_threshold}%
    - 🟡 **Medium Risk:** {medium_threshold}% to below {high_threshold}%
    - 🔴 **High Risk:** {high_threshold}% and above
    """
)


# --------------------------------------------------
# 8. TRANSACTION INPUT FORM
# --------------------------------------------------

st.divider()
st.subheader("📝 Analyze a New Transaction")

with st.form("fraud_transaction_form"):

    col1, col2 = st.columns(2)

    with col1:

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.01,
            value=500.0,
            step=100.0
        )

        hour = st.slider(
            "Transaction Hour",
            min_value=0,
            max_value=23,
            value=12
        )

        distance = st.number_input(
            "Distance from Usual Location (km)",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

    with col2:

        failed_attempts = st.number_input(
            "Failed Login / Payment Attempts",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )

        international_choice = st.selectbox(
            "International Transaction",
            ["No", "Yes"]
        )

    submitted = st.form_submit_button(
        "🔍 Analyze Transaction",
        use_container_width=True
    )


# --------------------------------------------------
# 9. ANALYZE TRANSACTION
# --------------------------------------------------

if submitted:

    international = (
        1 if international_choice == "Yes" else 0
    )

    input_values = {
        "amount": amount,
        "hour": hour,
        "distance_from_usual_km": distance,
        "failed_attempts": failed_attempts,
        "international": international
    }

    input_data = pd.DataFrame(
        [[input_values[feature] for feature in expected_features]],
        columns=expected_features
    )

    try:

        probabilities = model.predict_proba(input_data)[0]

        fraud_probability = float(
            probabilities[fraud_index]
        )

        prediction = model.predict(input_data)[0]

        # Risk classification

        if fraud_probability >= high_threshold / 100:
            risk_status = "HIGH"
            risk_message = "HIGH RISK - ALERT"

        elif fraud_probability >= medium_threshold / 100:
            risk_status = "MEDIUM"
            risk_message = "MEDIUM RISK - REVIEW"

        else:
            risk_status = "LOW"
            risk_message = "LOW RISK"

        transaction_id = (
            "FL-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        )

        result = {
            "Transaction ID": transaction_id,
            "Date & Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Amount": amount,
            "Hour": hour,
            "Distance (km)": distance,
            "Failed Attempts": failed_attempts,
            "International": international_choice,
            "Fraud Probability (%)": round(
                fraud_probability * 100, 2
            ),
            "Model Prediction": (
                "Fraud" if prediction == 1 else "Legitimate"
            ),
            "Risk Level": risk_status,
            "Risk Status": risk_message
        }

        st.session_state.latest_result = result

        st.session_state.alert_history.append(result)

    except Exception as e:

        st.error(f"Transaction analysis failed: {e}")


# --------------------------------------------------
# 10. DISPLAY LATEST RESULT
# --------------------------------------------------

if st.session_state.latest_result is not None:

    result = st.session_state.latest_result

    st.divider()
    st.subheader("📊 Transaction Risk Assessment")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Fraud Probability",
        f"{result['Fraud Probability (%)']:.2f}%"
    )

    col2.metric(
        "Risk Level",
        result["Risk Level"]
    )

    col3.metric(
        "Model Prediction",
        result["Model Prediction"]
    )

    st.progress(
        result["Fraud Probability (%)"] / 100
    )

    if result["Risk Level"] == "HIGH":

        st.error(
            "🚨 HIGH-RISK TRANSACTION DETECTED! "
            "The model estimates elevated fraud risk. "
            "Further review is recommended."
        )

    elif result["Risk Level"] == "MEDIUM":

        st.warning(
            "⚠️ MEDIUM-RISK TRANSACTION. "
            "Consider reviewing this transaction."
        )

    else:

        st.success(
            "✅ LOW-RISK TRANSACTION. "
            "The model estimates a lower fraud probability."
        )

    st.write("**Transaction ID:**", result["Transaction ID"])

    st.write("**Alert Status:**", result["Risk Status"])


# --------------------------------------------------
# 11. ALERT HISTORY DASHBOARD
# --------------------------------------------------

st.divider()
st.subheader("📋 Transaction Monitoring History")

history = pd.DataFrame(
    st.session_state.alert_history
)

if not history.empty:

    high_count = (
        history["Risk Level"] == "HIGH"
    ).sum()

    medium_count = (
        history["Risk Level"] == "MEDIUM"
    ).sum()

    low_count = (
        history["Risk Level"] == "LOW"
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Analyzed Transactions",
        len(history)
    )

    col2.metric(
        "🔴 High Risk",
        int(high_count)
    )

    col3.metric(
        "🟡 Medium Risk",
        int(medium_count)
    )

    col4.metric(
        "🟢 Low Risk",
        int(low_count)
    )

    # Filter history

    st.subheader("Filter Alert History")

    selected_risk = st.selectbox(
        "Choose Risk Level",
        ["All", "HIGH", "MEDIUM", "LOW"]
    )

    filtered_history = history.copy()

    if selected_risk != "All":

        filtered_history = filtered_history[
            filtered_history["Risk Level"] == selected_risk
        ]

    st.dataframe(
        filtered_history,
        use_container_width=True,
        hide_index=True
    )

    # Download history

    csv_data = history.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Complete Alert History",
        data=csv_data,
        file_name="fraudlens_alert_history.csv",
        mime="text/csv"
    )

    # Clear history

    if st.button("🗑️ Clear Alert History"):

        st.session_state.alert_history = []
        st.session_state.latest_result = None

        st.rerun()

else:

    st.info(
        "No transactions analyzed yet. "
        "Enter a transaction above to start monitoring."
    )


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "FraudLens AI | Interactive Fraud Monitoring Prototype | "
    "Machine-learning predictions using synthetic data. "
    "Alerts are estimates, not confirmed fraud."
)
