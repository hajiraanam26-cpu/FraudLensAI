
import streamlit as st
import pandas as pd

from utils.helpers import (
    load_data,
    load_model,
    add_predictions
)

st.set_page_config(
    page_title="Transaction History",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Transaction History")

df = add_predictions(load_data(), load_model())

st.write(
    "Search, filter and investigate previously processed transactions."
)

search = st.text_input(
    "Search Transaction ID"
)

risk_options = [
    "All",
    "Low Risk",
    "Medium Risk",
    "High Risk"
]

risk_filter = st.selectbox(
    "Filter by Risk Level",
    risk_options
)

class_filter = st.selectbox(
    "Filter by Actual Class",
    ["All", "Fraud", "Legitimate"]
)

filtered = df.copy()

if search and "transaction_id" in filtered.columns:
    filtered = filtered[
        filtered["transaction_id"].astype(str).str.contains(
            search,
            case=False,
            na=False
        )
    ]

if risk_filter != "All":
    filtered = filtered[
        filtered["risk_level"] == risk_filter
    ]

if class_filter == "Fraud":
    filtered = filtered[filtered["is_fraud"] == 1]

elif class_filter == "Legitimate":
    filtered = filtered[filtered["is_fraud"] == 0]

st.metric("Matching Transactions", len(filtered))

columns = [
    "transaction_id",
    "amount",
    "hour",
    "distance_from_usual_km",
    "failed_attempts",
    "international",
    "fraud_probability",
    "prediction",
    "risk_level"
]

columns = [
    c for c in columns if c in filtered.columns
]

st.dataframe(
    filtered[columns],
    use_container_width=True,
    hide_index=True
)

csv = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Export Transaction History",
    csv,
    "transaction_history.csv",
    "text/csv"
)

st.markdown("---")

st.subheader("Transaction Details")

if len(filtered) > 0:

    selected = st.selectbox(
        "Select a Transaction",
        filtered.index.tolist(),
        format_func=lambda i: (
            f"Row {i} | "
            f"{filtered.loc[i, 'risk_level']} | "
            f"{filtered.loc[i, 'fraud_probability']:.2%}"
        )
    )

    st.json(
        filtered.loc[selected].to_dict()
    )

else:
    st.warning("No transactions match your filters.")
