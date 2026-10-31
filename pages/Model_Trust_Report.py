
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from utils.helpers import (
    load_data,
    load_model,
    get_global_shap_importance,
    get_shap_values,
    FEATURES
)


# ==================================================
# PATHS AND PAGE TITLE
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"

st.title("🛡️ Model Explainability & Trust Report")

st.write(
    "A comprehensive assessment of FraudLens AI's "
    "performance, explainability, reliability and limitations."
)

st.divider()


# ==================================================
# LOAD METRICS, DATA AND MODEL
# ==================================================

@st.cache_data
def load_metrics():

    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "reports/metrics.json was not found. "
            "Please run model evaluation and save the reports."
        )

    with open(METRICS_PATH, "r") as file:
        return json.load(file)


try:
    metrics = load_metrics()
    df = load_data().reset_index(drop=True)
    model = load_model()

except Exception as e:
    st.error("Unable to load project data or model.")
    st.exception(e)
    st.stop()


# ==================================================
# 1. MODEL PERFORMANCE OVERVIEW
# ==================================================

st.header("📊 1. Model Performance Overview")

st.write(
    "The following metrics summarize the model's "
    "performance on the saved evaluation dataset."
)

metric_items = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1 Score", "f1_score"),
    ("ROC-AUC", "roc_auc"),
    ("Average Precision", "average_precision")
]

columns = st.columns(3)

for index, (label, key) in enumerate(metric_items):

    with columns[index % 3]:
        st.metric(
            label,
            f"{metrics[key] * 100:.2f}%"
        )

st.divider()


# ==================================================
# 2. FRAUD DETECTION ANALYSIS
# ==================================================

st.header("🎯 2. Fraud Detection Analysis")

st.write(
    "The confusion matrix compares actual transaction "
    "labels with the model's predicted labels."
)

cm = metrics["confusion_matrix"]

tn, fp = cm[0]
fn, tp = cm[1]

cm_df = pd.DataFrame(
    cm,
    index=["Actual Legitimate", "Actual Fraud"],
    columns=["Predicted Legitimate", "Predicted Fraud"]
)

fig_cm = px.imshow(
    cm_df,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Blues",
    title="Fraud Detection Confusion Matrix"
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
    st.metric("True Positives", tp)
    st.metric("True Negatives", tn)

with col2:
    st.metric("False Positives", fp)
    st.metric("False Negatives", fn)

st.subheader("What These Results Mean")

st.markdown(f"""
- **True Positives ({tp}):** Fraudulent transactions correctly detected.
- **True Negatives ({tn}):** Legitimate transactions correctly identified.
- **False Positives ({fp}):** Legitimate transactions incorrectly flagged as fraud.
- **False Negatives ({fn}):** Fraudulent transactions missed by the model.
""")

st.subheader("Additional Detection Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Specificity",
        f"{metrics['specificity'] * 100:.2f}%"
    )

with col2:
    st.metric(
        "Fraud Detection Rate",
        f"{metrics['recall'] * 100:.2f}%"
    )

with col3:
    st.metric(
        "Classification Threshold",
        f"{metrics.get('threshold', 0.5):.2f}"
    )

st.caption(
    "Fraud detection rate represents recall for the fraud class."
)

st.divider()


# ==================================================
# 3. GLOBAL SHAP EXPLAINABILITY
# ==================================================

st.header("🧠 3. Explainable AI Analysis")

st.write(
    "SHAP (SHapley Additive exPlanations) helps explain "
    "which features influence the model's predictions."
)

global_importance = None
ranked_features = None

try:

    global_importance = get_global_shap_importance(
        model,
        df
    )

    fig_shap = px.bar(
        global_importance.sort_values(
            "Importance",
            ascending=True
        ),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title="Global Mean Absolute SHAP Importance"
    )

    fig_shap.update_layout(
        height=450,
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Feature",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig_shap,
        use_container_width=True
    )

    st.subheader("Feature Importance Ranking")

    ranked_features = global_importance.sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True).copy()

    ranked_features.insert(
        0,
        "Rank",
        range(1, len(ranked_features) + 1)
    )

    ranked_features["Importance"] = (
        ranked_features["Importance"].round(4)
    )

    st.dataframe(
        ranked_features,
        use_container_width=True,
        hide_index=True
    )

    most_important = ranked_features.iloc[0]["Feature"]

    st.info(
        f"**Most influential feature: {most_important}.** "
        "It has the highest average absolute SHAP contribution "
        "in the analyzed sample."
    )

except Exception as e:

    st.warning(
        "Global SHAP analysis could not be generated."
    )
    st.exception(e)

st.caption(
    "Mean absolute SHAP measures average contribution magnitude. "
    "It does not show direction or prove causation."
)

st.divider()


# ==================================================
# 4. FRAUD DETECTION INSIGHTS
# ==================================================

st.header("🔎 4. Fraud Detection Insights")

st.write(
    "Explore how transaction features influence FraudLens AI's "
    "fraud predictions using SHAP explanations."
)

direction_df = None

if global_importance is not None:

    try:

        # ------------------------------------------
        # GLOBAL SHAP DIRECTION ANALYSIS
        # ------------------------------------------

        st.subheader(
            "📊 How Features Influence Fraud Predictions"
        )

        st.write(
            "Average signed SHAP contributions are calculated "
            "across a reproducible sample of transactions."
        )

        sample_size = min(50, len(df))

        shap_sample = df.sample(
            n=sample_size,
            random_state=42
        )

        all_shap_values = []

        progress = st.progress(0)

        for index, (_, sample_row) in enumerate(
            shap_sample.iterrows()
        ):

            transaction = sample_row[FEATURES].to_frame().T

            _, transaction_shap = get_shap_values(
                model,
                transaction,
                df[FEATURES]
            )

            all_shap_values.append(
                np.asarray(transaction_shap).reshape(-1)
            )

            progress.progress(
                (index + 1) / sample_size
            )

        progress.empty()

        shap_matrix = np.vstack(all_shap_values)

        # Average signed contribution
        mean_signed_shap = np.mean(
            shap_matrix,
            axis=0
        )

        # Average absolute contribution
        mean_absolute_shap = np.mean(
            np.abs(shap_matrix),
            axis=0
        )

        direction_df = pd.DataFrame({
            "Feature": FEATURES,
            "Average SHAP": mean_signed_shap,
            "Mean Absolute SHAP": mean_absolute_shap
        })

        direction_df["Direction"] = (
            direction_df["Average SHAP"].apply(
                lambda value:
                "Towards fraud" if value > 0
                else "Away from fraud" if value < 0
                else "Neutral on average"
            )
        )

        direction_df = direction_df.sort_values(
            "Mean Absolute SHAP",
            ascending=False
        )

        # ------------------------------------------
        # AVERAGE SHAP DIRECTION CHART
        # ------------------------------------------

        fig_direction = px.bar(
            direction_df.sort_values(
                "Average SHAP",
                ascending=True
            ),
            x="Average SHAP",
            y="Feature",
            orientation="h",
            color="Direction",
            color_discrete_map={
                "Towards fraud": "#e74c3c",
                "Away from fraud": "#2ecc71",
                "Neutral on average": "#95a5a6"
            },
            title="Average SHAP Contribution by Feature"
        )

        fig_direction.update_layout(
            height=450,
            xaxis_title="Average SHAP Value (Log-Odds)",
            yaxis_title="Feature"
        )

        st.plotly_chart(
            fig_direction,
            use_container_width=True
        )

        st.caption(
            "Positive average SHAP values indicate a general "
            "push towards fraud. Negative values indicate a "
            "general push away from fraud. Individual transactions "
            "may behave differently."
        )

        # ------------------------------------------
        # FEATURE LEVEL INSIGHTS TABLE
        # ------------------------------------------

        st.subheader("📋 Feature-Level Insights")

        display_direction = direction_df.copy()

        display_direction["Average SHAP"] = (
            display_direction["Average SHAP"].round(4)
        )

        display_direction["Mean Absolute SHAP"] = (
            display_direction["Mean Absolute SHAP"].round(4)
        )

        st.dataframe(
            display_direction,
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------
        # FEATURE INTERPRETATIONS
        # ------------------------------------------

        st.subheader("💡 What These Features Represent")

        feature_descriptions = {
            "amount": (
                "Transaction amount. The model may use transaction "
                "amount patterns when estimating fraud risk."
            ),

            "hour": (
                "Hour of the transaction. The model can learn "
                "patterns associated with transaction timing."
            ),

            "distance_from_usual_km": (
                "Distance from the customer's usual transaction "
                "location. Distance may influence the predicted risk."
            ),

            "failed_attempts": (
                "Number of failed transaction attempts. Repeated "
                "failures may contribute to a higher predicted risk."
            ),

            "international": (
                "Whether a transaction is international. The model "
                "can learn different patterns for domestic and "
                "international transactions."
            )
        }

        for _, feature in direction_df.iterrows():

            feature_name = feature["Feature"]

            with st.expander(
                f"{feature_name} — {feature['Direction']}"
            ):

                st.write(
                    feature_descriptions.get(
                        feature_name,
                        "This feature contributes to the model prediction."
                    )
                )

                st.write(
                    f"Average SHAP contribution: "
                    f"**{feature['Average SHAP']:.4f}**"
                )

                st.write(
                    f"Mean absolute SHAP importance: "
                    f"**{feature['Mean Absolute SHAP']:.4f}**"
                )

        st.divider()

        # ------------------------------------------
        # INDIVIDUAL TRANSACTION EXPLANATION
        # ------------------------------------------

        st.subheader(
            "🔍 Explore an Individual Transaction"
        )

        selected_index = st.selectbox(
            "Select a transaction for explanation",
            options=list(range(len(df))),
            format_func=lambda x: f"Transaction {x + 1}",
            key="trust_report_transaction"
        )

        selected_row = df.iloc[
            [selected_index]
        ][FEATURES]

        probabilities = model.predict_proba(
            selected_row
        )[0]

        model_classes = list(model.classes_)
        fraud_index = model_classes.index(1)

        selected_probability = float(
            probabilities[fraud_index]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Fraud Probability",
                f"{selected_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Model Prediction",
                "Likely Fraudulent"
                if selected_probability >= metrics.get("threshold", 0.5)
                else "Likely Legitimate"
            )

        _, selected_shap = get_shap_values(
            model,
            selected_row,
            df[FEATURES]
        )

        selected_shap = np.asarray(
            selected_shap
        ).reshape(-1)

        individual_insights = pd.DataFrame({
            "Feature": FEATURES,
            "Transaction Value": selected_row.iloc[0].values,
            "SHAP Contribution": selected_shap
        })

        individual_insights["Influence"] = (
            individual_insights["SHAP Contribution"].apply(
                lambda value:
                "Towards fraud" if value > 0
                else "Away from fraud" if value < 0
                else "Neutral"
            )
        )

        individual_insights["Absolute Contribution"] = (
            individual_insights["SHAP Contribution"].abs()
        )

        individual_insights = individual_insights.sort_values(
            "Absolute Contribution",
            ascending=False
        )

        st.write("**Top three contributing features:**")

        for _, feature in individual_insights.head(3).iterrows():

            st.write(
                f"- **{feature['Feature']}** "
                f"(Value: {feature['Transaction Value']}) — "
                f"{feature['Influence']} "
                f"(SHAP: {feature['SHAP Contribution']:.4f})"
            )

        st.dataframe(
            individual_insights[
                [
                    "Feature",
                    "Transaction Value",
                    "SHAP Contribution",
                    "Influence"
                ]
            ].round(4),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Individual SHAP contributions explain this selected "
            "transaction's model output in log-odds."
        )

    except Exception as e:

        st.error(
            "Unable to generate Fraud Detection Insights."
        )
        st.exception(e)

else:

    st.warning(
        "Fraud Detection Insights are unavailable because "
        "global SHAP analysis could not be calculated."
    )

st.divider()


# ==================================================
# 5. MODEL TRUST ASSESSMENT
# ==================================================

st.header("🔐 5. Model Trust Assessment")

st.write(
    "This section summarizes the strengths and limitations "
    "of the current FraudLens AI model."
)

st.subheader("✅ Model Strengths")

st.markdown("""
- Uses a Logistic Regression classification model.
- Applies StandardScaler through an sklearn Pipeline.
- Uses class weighting to help address class imbalance.
- Provides fraud probability scores.
- Supports individual transaction explanations using SHAP.
- Provides global feature importance analysis.
- Evaluates performance using multiple classification metrics.
""")

st.subheader("⚠️ Model Limitations")

st.markdown("""
- The current dataset is synthetic and may not represent real banking transactions.
- Model performance on real-world financial data is not established.
- False positives may incorrectly flag legitimate transactions.
- False negatives mean some fraudulent transactions may go undetected.
- SHAP explains model behavior, not the actual cause of fraud.
- Performance can change when transaction patterns change.
- The classification threshold affects the balance between precision and recall.
""")

st.subheader("📋 Evaluation Details")

details = {
    "Evaluation Samples": metrics.get("test_rows", "Not available"),
    "Model Type": metrics.get("model_type", "Not available"),
    "Classification Threshold": metrics.get("threshold", 0.5),
    "Evaluation Date": "See your saved evaluation run"
}

st.table(
    pd.DataFrame(
        list(details.items()),
        columns=["Parameter", "Value"]
    )
)

st.divider()


# ==================================================
# 6. OVERALL ASSESSMENT
# ==================================================

st.header("📌 6. Overall Assessment")

st.write(
    f"""
    FraudLens AI achieved an F1-score of
    **{metrics['f1_score'] * 100:.2f}%** and ROC-AUC of
    **{metrics['roc_auc'] * 100:.2f}%** on its saved evaluation set.

    The model detected **{tp} out of {tp + fn}** fraudulent
    transactions in that evaluation set.

    Its SHAP explanations provide visibility into how the
    model uses transaction features when producing predictions.
    """
)

st.warning(
    "This is an experimental academic project using synthetic data. "
    "It should not be used to make real financial decisions without "
    "validation on representative real-world data, security testing "
    "and appropriate human oversight."
)

st.divider()


# ==================================================
# 7. DOWNLOADABLE TRUST REPORT
# ==================================================

st.header("📥 7. Download Model Trust Report")

st.write(
    "Download a summary of the model's evaluation results, "
    "fraud detection analysis and trust considerations."
)

report = f"""
# FraudLens AI
## Model Explainability & Trust Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Model Information

Model Type: {metrics.get("model_type", "Not available")}
Evaluation Samples: {metrics.get("test_rows", "Not available")}
Classification Threshold: {metrics.get("threshold", 0.5)}

## 2. Model Performance

Accuracy: {metrics["accuracy"] * 100:.2f}%
Precision: {metrics["precision"] * 100:.2f}%
Recall: {metrics["recall"] * 100:.2f}%
F1 Score: {metrics["f1_score"] * 100:.2f}%
ROC-AUC: {metrics["roc_auc"] * 100:.2f}%
Average Precision: {metrics["average_precision"] * 100:.2f}%
Specificity: {metrics["specificity"] * 100:.2f}%

## 3. Confusion Matrix

True Negatives: {tn}
False Positives: {fp}
False Negatives: {fn}
True Positives: {tp}

## 4. Global SHAP Feature Importance

"""

if ranked_features is not None:

    for _, item in ranked_features.iterrows():

        report += (
            f"- {item['Feature']}: "
            f"{item['Importance']:.4f}\n"
        )

if direction_df is not None:

    report += "\n## 5. Average SHAP Direction Analysis\n\n"

    for _, item in direction_df.iterrows():

        report += (
            f"- {item['Feature']}: "
            f"{item['Direction']} "
            f"(Average SHAP: {item['Average SHAP']:.4f})\n"
        )

report += """
## 6. Model Strengths

- Logistic Regression classification model.
- StandardScaler preprocessing pipeline.
- Class weighting.
- Individual and global SHAP explanations.
- Multiple model evaluation metrics.

## 7. Model Limitations

- Synthetic data may not represent real banking behavior.
- Real-world generalization has not been established.
- False positives and false negatives remain possible.
- SHAP explains model behavior, not causation.
- Performance may change with new transaction patterns.

## 8. Responsible Use

This is an academic experimental project.
It must not be used for real financial decisions without
appropriate real-world validation, security testing and
human oversight.

---
FraudLens AI | Model Explainability & Trust Report
"""

st.download_button(
    label="📄 Download Trust Report",
    data=report,
    file_name="FraudLens_AI_Trust_Report.md",
    mime="text/markdown",
    use_container_width=True
)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "FraudLens AI | AI-Powered Fraud Detection & Explainability"
)

st.caption(
    "SHAP explains model behavior, not necessarily real-world causes."
)

