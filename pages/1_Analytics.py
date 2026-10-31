
import streamlit as st
import plotly.express as px

from utils.helpers import load_data, load_model, add_predictions

st.set_page_config(
    page_title="Fraud Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Fraud Analytics")
st.write("Explore fraud patterns and transaction behaviour.")

df = add_predictions(load_data(), load_model())

# Filters
st.sidebar.header("Analytics Filters")

min_amount = float(df["amount"].min())
max_amount = float(df["amount"].max())

amount_range = st.sidebar.slider(
    "Transaction Amount",
    min_amount,
    max_amount,
    (min_amount, max_amount)
)

hours = sorted(df["hour"].unique())

selected_hours = st.sidebar.multiselect(
    "Select Hours",
    hours,
    default=hours
)

fraud_filter = st.sidebar.selectbox(
    "Actual Transaction Type",
    ["All", "Fraud", "Legitimate"]
)

filtered = df[
    df["amount"].between(*amount_range)
    & df["hour"].isin(selected_hours)
].copy()

if fraud_filter == "Fraud":
    filtered = filtered[filtered["is_fraud"] == 1]

elif fraud_filter == "Legitimate":
    filtered = filtered[filtered["is_fraud"] == 0]

st.metric("Filtered Transactions", len(filtered))

c1, c2 = st.columns(2)

with c1:
    st.subheader("Amount vs Fraud")

    fig = px.histogram(
        filtered,
        x="amount",
        color="is_fraud",
        nbins=30,
        barmode="overlay"
    )

    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Fraud by Hour")

    hourly = filtered.groupby("hour").agg(
        transactions=("is_fraud", "size"),
        fraud=("is_fraud", "sum")
    ).reset_index()

    hourly["fraud_rate"] = (
        hourly["fraud"] / hourly["transactions"] * 100
    )

    fig = px.bar(
        hourly,
        x="hour",
        y="fraud_rate",
        title="Fraud Rate by Hour"
    )

    st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)

with c1:
    st.subheader("Failed Attempts Analysis")

    fig = px.histogram(
        filtered,
        x="failed_attempts",
        color="is_fraud",
        barmode="group"
    )

    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Distance from Usual Location")

    fig = px.box(
        filtered,
        x="is_fraud",
        y="distance_from_usual_km",
        color="is_fraud"
    )

    st.plotly_chart(fig, use_container_width=True)

st.subheader("International Transaction Analysis")

international = filtered.groupby(
    ["international", "is_fraud"]
).size().reset_index(name="count")

fig = px.bar(
    international,
    x="international",
    y="count",
    color="is_fraud",
    barmode="group"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Filtered Transaction Data")

st.dataframe(filtered, use_container_width=True)

csv = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download Filtered Transactions",
    csv,
    "fraud_analytics.csv",
    "text/csv"
)
