
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "transactions.csv"

st.set_page_config(
    page_title="Fraud Analytics | FraudLens AI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Fraud Analytics Dashboard")
st.caption(
    "Monitor transaction patterns, identify fraud indicators "
    "and explore synthetic transaction data."
)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load transaction data: {e}")
    st.stop()


# --------------------------------------------------
# 3. DATA VALIDATION
# --------------------------------------------------

required_columns = {
    "is_fraud",
    "amount",
    "hour",
    "distance_from_usual_km",
    "failed_attempts",
    "international",
}

missing = required_columns - set(df.columns)

if missing:
    st.error(
        f"Missing required columns: {', '.join(sorted(missing))}"
    )
    st.stop()

if df.empty:
    st.warning("The transaction dataset is empty.")
    st.stop()

numeric_columns = list(required_columns)

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(subset=numeric_columns).copy()

if df.empty:
    st.error("No valid transaction records are available.")
    st.stop()

df["is_fraud"] = df["is_fraud"].astype(int)
df["international"] = df["international"].astype(int)

if not df["is_fraud"].isin([0, 1]).all():
    st.error("The fraud label must contain only 0 and 1.")
    st.stop()

df["Transaction Type"] = df["is_fraud"].map({
    0: "Legitimate",
    1: "Fraud",
})

df["Transaction Category"] = df["international"].map({
    0: "Domestic",
    1: "International",
})

df["Hour"] = df["hour"].astype(int)


# --------------------------------------------------
# 4. KEY METRICS
# --------------------------------------------------

total_transactions = len(df)

fraud_df = df[df["is_fraud"] == 1]
legitimate_df = df[df["is_fraud"] == 0]

fraud_count = len(fraud_df)
legitimate_count = len(legitimate_df)

fraud_rate = fraud_count / total_transactions * 100

average_amount = df["amount"].mean()

average_fraud_amount = (
    fraud_df["amount"].mean()
    if not fraud_df.empty else 0
)

total_fraud_amount = fraud_df["amount"].sum()

st.divider()
st.subheader("📊 Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

c2.metric(
    "Fraud Examples",
    f"{fraud_count:,}"
)

c3.metric(
    "Synthetic Fraud Rate",
    f"{fraud_rate:.1f}%"
)

c4.metric(
    "Average Transaction",
    f"{average_amount:,.2f}"
)

c5, c6, c7 = st.columns(3)

c5.metric(
    "Legitimate Transactions",
    f"{legitimate_count:,}"
)

c6.metric(
    "Average Fraud Amount",
    f"{average_fraud_amount:,.2f}"
)

c7.metric(
    "Total Fraud-Labelled Amount",
    f"{total_fraud_amount:,.2f}"
)

st.info(
    "These statistics describe synthetic demonstration data. "
    "They do not represent actual financial fraud."
)


# --------------------------------------------------
# 5. INTERACTIVE FILTERS
# --------------------------------------------------

st.divider()
st.subheader("🔎 Explore Transactions")

filter1, filter2, filter3 = st.columns(3)

with filter1:
    transaction_type = st.selectbox(
        "Transaction Type",
        ["All", "Fraud", "Legitimate"]
    )

with filter2:
    category = st.selectbox(
        "Transaction Category",
        ["All", "Domestic", "International"]
    )

with filter3:
    hour_range = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=(0, 23)
    )

min_amount = float(df["amount"].min())
max_amount = float(df["amount"].max())

if min_amount < max_amount:
    amount_range = st.slider(
        "Transaction Amount Range",
        min_value=min_amount,
        max_value=max_amount,
        value=(min_amount, max_amount)
    )
else:
    amount_range = (min_amount, max_amount)


filtered_df = df.copy()

if transaction_type != "All":
    filtered_df = filtered_df[
        filtered_df["Transaction Type"] == transaction_type
    ]

if category != "All":
    filtered_df = filtered_df[
        filtered_df["Transaction Category"] == category
    ]

filtered_df = filtered_df[
    filtered_df["Hour"].between(
        hour_range[0],
        hour_range[1]
    )
]

filtered_df = filtered_df[
    filtered_df["amount"].between(
        amount_range[0],
        amount_range[1]
    )
]

st.write(
    f"**Transactions matching filters: {len(filtered_df):,}**"
)


# --------------------------------------------------
# 6. FRAUD VS LEGITIMATE DISTRIBUTION
# --------------------------------------------------

st.divider()
st.subheader("🥧 Transaction Distribution")

distribution = (
    df["Transaction Type"]
    .value_counts()
    .reset_index()
)

distribution.columns = [
    "Transaction Type",
    "Count"
]

fig_pie = px.pie(
    distribution,
    names="Transaction Type",
    values="Count",
    hole=0.55,
    title="Fraud vs Legitimate Transactions",
)

fig_pie.update_traces(
    textinfo="percent+label"
)

st.plotly_chart(
    fig_pie,
    use_container_width=True
)


# --------------------------------------------------
# 7. HOURLY FRAUD ANALYSIS
# --------------------------------------------------

st.divider()
st.subheader("🕒 Fraud Patterns by Hour")

hourly = (
    df.groupby("Hour")
    .agg(
        Transactions=("is_fraud", "size"),
        Fraud_Count=("is_fraud", "sum"),
        Fraud_Rate=("is_fraud", "mean")
    )
    .reset_index()
)

hourly["Fraud Rate (%)"] = (
    hourly["Fraud_Rate"] * 100
)

fig_hour = px.line(
    hourly,
    x="Hour",
    y="Fraud Rate (%)",
    markers=True,
    hover_data=[
        "Transactions",
        "Fraud_Count"
    ],
    title="Synthetic Fraud Rate Throughout the Day"
)

fig_hour.update_xaxes(
    dtick=1,
    title="Hour of Day"
)

fig_hour.update_yaxes(
    title="Fraud Rate (%)"
)

st.plotly_chart(
    fig_hour,
    use_container_width=True
)


# --------------------------------------------------
# 8. TRANSACTION AMOUNT ANALYSIS
# --------------------------------------------------

st.divider()
st.subheader("💰 Transaction Amount Analysis")

fig_amount = px.histogram(
    df,
    x="amount",
    color="Transaction Type",
    nbins=40,
    barmode="overlay",
    opacity=0.7,
    title="Transaction Amount Distribution"
)

fig_amount.update_layout(
    xaxis_title="Transaction Amount",
    yaxis_title="Number of Transactions"
)

st.plotly_chart(
    fig_amount,
    use_container_width=True
)


# --------------------------------------------------
# 9. FAILED ATTEMPT ANALYSIS
# --------------------------------------------------

st.divider()
st.subheader("⚠️ Failed Attempts vs Fraud")

failed_summary = (
    df.groupby("failed_attempts")
    .agg(
        Transactions=("is_fraud", "size"),
        Fraud_Rate=("is_fraud", "mean")
    )
    .reset_index()
)

failed_summary["Fraud Rate (%)"] = (
    failed_summary["Fraud_Rate"] * 100
)

fig_failed = px.bar(
    failed_summary,
    x="failed_attempts",
    y="Fraud Rate (%)",
    text="Fraud Rate (%)",
    hover_data=["Transactions"],
    title="Synthetic Fraud Rate by Failed Attempts"
)

fig_failed.update_layout(
    xaxis_title="Number of Failed Attempts",
    yaxis_title="Fraud Rate (%)"
)

st.plotly_chart(
    fig_failed,
    use_container_width=True
)


# --------------------------------------------------
# 10. DOMESTIC VS INTERNATIONAL ANALYSIS
# --------------------------------------------------

st.divider()
st.subheader("🌍 Domestic vs International Transactions")

international_summary = (
    df.groupby("Transaction Category")
    .agg(
        Transactions=("is_fraud", "size"),
        Fraud_Count=("is_fraud", "sum"),
        Fraud_Rate=("is_fraud", "mean")
    )
    .reset_index()
)

international_summary["Fraud Rate (%)"] = (
    international_summary["Fraud_Rate"] * 100
)

fig_international = px.bar(
    international_summary,
    x="Transaction Category",
    y="Fraud Rate (%)",
    text="Fraud Rate (%)",
    hover_data=[
        "Transactions",
        "Fraud_Count"
    ],
    title="Synthetic Fraud Rate by Transaction Category"
)

st.plotly_chart(
    fig_international,
    use_container_width=True
)


# --------------------------------------------------
# 11. HIGH-VALUE FRAUD ANALYSIS
# --------------------------------------------------

st.divider()
st.subheader("🚨 High-Value Fraud Examples")

if fraud_df.empty:

    st.info("No fraud-labelled transactions were found.")

else:

    high_value_fraud = fraud_df.nlargest(
        10,
        "amount"
    )

    fig_high = px.bar(
        high_value_fraud,
        x="amount",
        y="transaction_id" if "transaction_id" in df.columns
        else high_value_fraud.index.astype(str),
        orientation="h",
        title="Top 10 Highest-Amount Fraud Examples",
        labels={
            "amount": "Transaction Amount",
            "transaction_id": "Transaction ID"
        }
    )

    st.plotly_chart(
        fig_high,
        use_container_width=True
    )


# --------------------------------------------------
# 12. FILTERED TRANSACTION TABLE
# --------------------------------------------------

st.divider()
st.subheader("📋 Transaction Records")

display_columns = [
    column for column in [
        "transaction_id",
        "amount",
        "hour",
        "distance_from_usual_km",
        "failed_attempts",
        "international",
        "is_fraud"
    ]
    if column in filtered_df.columns
]

st.dataframe(
    filtered_df[display_columns].head(100),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 13. DOWNLOAD FILTERED DATA
# --------------------------------------------------

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="📥 Download Filtered Transactions (CSV)",
    data=csv_data,
    file_name="fraudlens_filtered_transactions.csv",
    mime="text/csv"
)


# --------------------------------------------------
# 14. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "FraudLens AI | Interactive Fraud Analytics | "
    "Educational prototype using synthetic transaction data."
)
