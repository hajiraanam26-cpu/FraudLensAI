
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.helpers import (
    load_data,
    load_model,
    add_predictions,
    FEATURES,
    get_shap_values,
    get_global_shap_importance
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.title("🧠 Explainable AI Dashboard")

st.write(
    "Understand why FraudLens AI classifies transactions "
    "as potentially fraudulent or legitimate using SHAP."
)

st.divider()


# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

try:
    df = load_data()
    model = load_model()

    # Reset index for consistent transaction selection
    df = df.reset_index(drop=True)

    # Generate model predictions
    predicted_df = add_predictions(df, model)

except Exception as e:
    st.error("Unable to load the dataset or model.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# TRANSACTION SELECTION
# --------------------------------------------------

st.subheader("🔍 Select a Transaction")

selected_index = st.selectbox(
    "Choose Transaction ID",
    options=list(range(len(predicted_df))),
    format_func=lambda x: f"Transaction {x + 1}"
)

# Get selected transaction
row = df.iloc[[selected_index]][FEATURES]

selected_result = predicted_df.iloc[selected_index]

fraud_probability = float(selected_result["fraud_probability"])
prediction = int(selected_result["prediction"])
risk_level = selected_result["risk_level"]


# --------------------------------------------------
# DISPLAY PREDICTION RESULTS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Fraud Probability",
        f"{fraud_probability * 100:.2f}%"
    )

with col2:
    st.metric(
        "Prediction",
        "Likely Fraudulent" if prediction == 1
        else "Likely Legitimate"
    )

with col3:
    st.metric(
        "Risk Category",
        risk_level
    )

if prediction == 1:
    st.error("⚠️ Model predicts this transaction is likely fraudulent.")
else:
    st.success("✅ Model predicts this transaction is likely legitimate.")

st.divider()


# --------------------------------------------------
# INDIVIDUAL SHAP EXPLANATION
# --------------------------------------------------

st.subheader("🧠 Individual Transaction Explanation")

try:
    explainer, shap_values = get_shap_values(
        model,
        row,
        df[FEATURES]
    )

    shap_values = np.asarray(shap_values).reshape(-1)

    # Create SHAP contribution dataframe
    contribution_df = pd.DataFrame({
        "Feature": FEATURES,
        "Feature Value": row.iloc[0].values,
        "SHAP Value": shap_values
    })

    contribution_df["Influence"] = contribution_df[
        "SHAP Value"
    ].apply(
        lambda x: "Towards fraud" if x > 0
        else "Away from fraud" if x < 0
        else "No influence"
    )

    # Individual SHAP bar chart
    individual_df = contribution_df.copy()
    individual_df["Absolute SHAP"] = individual_df[
        "SHAP Value"
    ].abs()

    individual_df = individual_df.sort_values(
        "Absolute SHAP",
        ascending=True
    )

    fig_individual = px.bar(
        individual_df,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        color="SHAP Value",
        color_continuous_scale="RdBu",
        title="Individual Feature Contributions"
    )

    fig_individual.update_layout(
        height=400,
        xaxis_title="SHAP Value (Log-Odds)",
        yaxis_title="Feature",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig_individual,
        use_container_width=True
    )

    st.caption(
        "Positive SHAP values push the model output towards fraud. "
        "Negative SHAP values push it away from fraud. "
        "Values are in log-odds, not percentage points."
    )

    st.divider()


    # --------------------------------------------------
    # TOP 3 INFLUENCING FEATURES
    # --------------------------------------------------

    st.subheader("🏆 Top 3 Influencing Features")

    top_features = contribution_df.copy()

    top_features["Absolute SHAP"] = top_features[
        "SHAP Value"
    ].abs()

    top_features = top_features.sort_values(
        "Absolute SHAP",
        ascending=False
    ).head(3)

    for _, feature in top_features.iterrows():

        direction = feature["Influence"]

        st.write(
            f"**{feature['Feature']}** — {direction} "
            f"(SHAP: {feature['SHAP Value']:.4f})"
        )

    st.divider()


    # --------------------------------------------------
    # SHAP WATERFALL PLOT
    # --------------------------------------------------

    st.subheader("🌊 SHAP Waterfall Plot")

    st.write(
        "This visualization shows how each feature moves "
        "the selected transaction's model output from the "
        "baseline towards its final prediction."
    )

    # Get the baseline expected value
    base_values = np.asarray(
        explainer.expected_value
    ).reshape(-1)

    estimator = model.named_steps["model"]
    classes = list(estimator.classes_)

    if 1 in classes and len(base_values) == len(classes):
        fraud_index = classes.index(1)
        base_value = float(base_values[fraud_index])
    else:
        base_value = float(base_values[0])

    # Calculate final model output
    final_value = float(
        base_value + shap_values.sum()
    )

    # Waterfall chart data
    waterfall_labels = (
        ["Baseline"]
        + FEATURES
        + ["Final Output"]
    )

    waterfall_values = (
        [base_value]
        + shap_values.tolist()
        + [final_value]
    )

    waterfall_measure = (
        ["absolute"]
        + ["relative"] * len(FEATURES)
        + ["total"]
    )

    fig_waterfall = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=waterfall_measure,
            x=waterfall_labels,
            y=waterfall_values,
            text=[
                f"{value:+.3f}"
                for value in waterfall_values
            ],
            textposition="outside",
            connector={
                "line": {
                    "width": 1
                }
            },
            increasing={
                "marker": {
                    "color": "#e74c3c"
                }
            },
            decreasing={
                "marker": {
                    "color": "#2ecc71"
                }
            },
            totals={
                "marker": {
                    "color": "#3498db"
                }
            }
        )
    )

    fig_waterfall.update_layout(
        title="How Each Feature Influences This Transaction",
        xaxis_title="Features",
        yaxis_title="Model Output (Log-Odds)",
        showlegend=False,
        height=550,
        margin=dict(
            t=70,
            b=110
        )
    )

    st.plotly_chart(
        fig_waterfall,
        use_container_width=True
    )

    st.caption(
        "The baseline represents the model's expected output. "
        "Each feature adds or subtracts its SHAP contribution. "
        "The final output is the fraud-class log-odds."
    )

    st.divider()


    # --------------------------------------------------
    # FEATURE CONTRIBUTION DETAILS
    # --------------------------------------------------

    st.subheader("📋 Feature Contribution Details")

    display_df = contribution_df.copy()

    display_df["SHAP Value"] = display_df[
        "SHAP Value"
    ].round(4)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

except Exception as e:
    st.error("Unable to generate the individual SHAP explanation.")
    st.exception(e)


# --------------------------------------------------
# GLOBAL SHAP FEATURE IMPORTANCE
# --------------------------------------------------

st.divider()

st.subheader("🌍 Global Feature Importance")

st.write(
    "Understand which features have the greatest overall "
    "influence on the model's predictions across multiple transactions."
)

try:
    global_importance = get_global_shap_importance(
        model,
        df
    )

    fig_global = px.bar(
        global_importance.sort_values(
            "Importance",
            ascending=True
        ),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title="Mean Absolute SHAP Feature Importance"
    )

    fig_global.update_layout(
        height=450,
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Feature",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig_global,
        use_container_width=True
    )

    st.caption(
        "Global importance is calculated using a sample of "
        "100 transactions. Higher values indicate greater "
        "average influence on the model's fraud-class output."
    )

    st.divider()

    st.subheader("📊 Global Feature Importance Details")

    global_display = global_importance.copy()

    global_display["Importance"] = global_display[
        "Importance"
    ].round(4)

    st.dataframe(
        global_display,
        use_container_width=True,
        hide_index=True
    )

except Exception as e:
    st.error("Unable to calculate global SHAP importance.")
    st.exception(e)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "FraudLens AI | Explainable Logistic Regression + SHAP"
)

st.caption(
    "SHAP explains model behaviour, not necessarily real-world causes."
)
