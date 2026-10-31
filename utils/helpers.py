
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "transactions.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.joblib"

FEATURES = [
    "amount",
    "hour",
    "distance_from_usual_km",
    "failed_attempts",
    "international"
]


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    required = FEATURES + ["is_fraud"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).copy()

    return df


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def get_fraud_probability(model, data):
    probabilities = model.predict_proba(data[FEATURES])

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Fraud class 1 not found in model.")

    fraud_index = classes.index(1)

    return probabilities[:, fraud_index]


def classify_risk(probability):
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.70:
        return "Medium Risk"
    return "High Risk"


def get_prediction(probability):
    return int(probability >= 0.50)


def add_predictions(df, model):
    result = df.copy()

    result["fraud_probability"] = get_fraud_probability(
        model, result
    )

    result["prediction"] = (
        result["fraud_probability"] >= 0.50
    ).astype(int)

    result["risk_level"] = (
        result["fraud_probability"].apply(classify_risk)
    )

    return result


def _get_pipeline_parts(model):
    """Extract scaler and estimator from the trained Pipeline."""

    if not hasattr(model, "named_steps"):
        raise TypeError("Expected a fitted sklearn Pipeline.")

    steps = model.named_steps

    scaler = steps.get("scaler")
    estimator = steps.get("model")

    if scaler is None or estimator is None:
        raise ValueError(
            "Pipeline must contain 'scaler' and 'model' steps."
        )

    return scaler, estimator


@st.cache_resource
def _get_linear_explainer(_model, background_data):
    """Create a SHAP LinearExplainer for Logistic Regression."""

    import shap

    scaler, estimator = _get_pipeline_parts(_model)

    background_scaled = scaler.transform(
        background_data[FEATURES]
    )

    explainer = shap.LinearExplainer(
        estimator,
        background_scaled
    )

    return explainer


def get_shap_values(model, row, background_data=None):
    """Calculate SHAP values for the fraud class."""

    scaler, estimator = _get_pipeline_parts(model)

    if background_data is None:
        background_data = load_data()[FEATURES]

    background_data = background_data[FEATURES].sample(
        n=min(100, len(background_data)),
        random_state=42
    )

    explainer = _get_linear_explainer(
        model,
        background_data
    )

    scaled_row = scaler.transform(row[FEATURES])

    shap_result = explainer(scaled_row)

    classes = list(estimator.classes_)

    if 1 not in classes:
        raise ValueError("Fraud class 1 not found in estimator.")

    fraud_index = classes.index(1)

    values = np.asarray(shap_result.values)

    if values.ndim == 3:
        values = values[0, :, fraud_index]
    elif values.ndim == 2:
        values = values[0]

    return explainer, values


def get_global_shap_importance(model, data):
    """Calculate global mean absolute SHAP importance."""

    scaler, estimator = _get_pipeline_parts(model)

    data = data[FEATURES].sample(
        n=min(100, len(data)),
        random_state=42
    )

    explainer = _get_linear_explainer(
        model,
        data
    )

    scaled_data = scaler.transform(data[FEATURES])

    shap_result = explainer(scaled_data)

    classes = list(estimator.classes_)

    if 1 not in classes:
        raise ValueError("Fraud class 1 not found in estimator.")

    fraud_index = classes.index(1)

    values = np.asarray(shap_result.values)

    if values.ndim == 3:
        values = values[:, :, fraud_index]

    importance = np.mean(
        np.abs(values),
        axis=0
    )

    return pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importance
    }).sort_values(
        "Importance",
        ascending=False
    )
