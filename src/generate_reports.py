
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "transactions.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.joblib"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Features
feature_columns = [
    "amount",
    "hour",
    "distance_from_usual_km",
    "failed_attempts",
    "international",
]

# Load dataset
print("Loading transaction dataset...")
df = pd.read_csv(DATA_PATH)

X = df[feature_columns]
y = df["is_fraud"]

# Recreate the held-out split from train_model.py
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Load existing trained model (do not retrain)
print("Loading existing fraud detection model...")
model = joblib.load(MODEL_PATH)

# Generate predictions
print("Generating test predictions...")

y_pred = model.predict(X_test)

classes = list(model.classes_)
fraud_index = classes.index(1)

y_prob = model.predict_proba(X_test)[:, fraud_index]

# Calculate metrics
metrics = {
    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    "precision": round(
        float(precision_score(y_test, y_pred, zero_division=0)), 4
    ),
    "recall": round(
        float(recall_score(y_test, y_pred, zero_division=0)), 4
    ),
    "f1_score": round(
        float(f1_score(y_test, y_pred, zero_division=0)), 4
    ),
    "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    "confusion_matrix": confusion_matrix(
        y_test, y_pred
    ).tolist(),
    "test_rows": int(len(X_test)),
    "model_type": type(model).__name__,
}

# Save metrics
metrics_path = REPORT_DIR / "metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

# Save individual predictions
predictions = X_test.copy()

predictions.insert(0, "actual", y_test.to_numpy())
predictions["predicted"] = y_pred
predictions["fraud_probability"] = y_prob

predictions.to_csv(
    REPORT_DIR / "test_predictions.csv",
    index=False,
)

# Display results
print("\nMODEL EVALUATION RESULTS")
print("------------------------")

for name, value in metrics.items():
    print(f"{name}: {value}")

print("\nReports successfully generated!")
print("Saved:", metrics_path)
print("Saved:", REPORT_DIR / "test_predictions.csv")
