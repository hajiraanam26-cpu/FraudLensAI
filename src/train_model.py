
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "transactions.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "fraud_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


# ==================================================
# 2. CONFIGURATION
# ==================================================

RANDOM_STATE = 42

FEATURES = [
    "amount",
    "hour",
    "distance_from_usual_km",
    "failed_attempts",
    "international",
]

TARGET = "is_fraud"


# ==================================================
# 3. LOAD AND VALIDATE DATA
# ==================================================

print("Loading transaction dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

df = df[required_columns].copy()

df = df.apply(
    pd.to_numeric,
    errors="coerce"
).dropna()

if not df[TARGET].isin([0, 1]).all():
    raise ValueError(
        "Target column must contain only 0 and 1."
    )

if df[TARGET].nunique() != 2:
    raise ValueError(
        "Dataset must contain both legitimate and fraud classes."
    )

X = df[FEATURES]
y = df[TARGET].astype(int)

print(f"Dataset rows: {len(df)}")
print("\nClass distribution:")
print(y.value_counts())


# ==================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ==================================================

print("\nCreating stratified data splits...")

# First reserve 15% for final testing.
X_remaining, X_test, y_remaining, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=y,
)

# From the remaining 85%, reserve 15% of the
# original dataset for validation.
validation_fraction = 0.15 / 0.85

X_train, X_validation, y_train, y_validation = train_test_split(
    X_remaining,
    y_remaining,
    test_size=validation_fraction,
    random_state=RANDOM_STATE,
    stratify=y_remaining,
)

print(f"Training rows: {len(X_train)}")
print(f"Validation rows: {len(X_validation)}")
print(f"Testing rows: {len(X_test)}")


# ==================================================
# 5. BUILD LOGISTIC REGRESSION PIPELINE
# ==================================================

model = Pipeline([
    ("scaler", StandardScaler()),

    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )),
])


# ==================================================
# 6. CROSS-VALIDATION ON TRAINING DATA
# ==================================================

print("\nRunning 5-fold cross-validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
)

print("\nCross-validation F1 scores:")

for fold, score in enumerate(cv_scores, start=1):
    print(f"Fold {fold}: {score:.4f}")

cv_mean = float(np.mean(cv_scores))
cv_std = float(np.std(cv_scores))

print(f"\nMean CV F1: {cv_mean:.4f}")
print(f"CV Standard Deviation: {cv_std:.4f}")


# ==================================================
# 7. TRAIN FINAL MODEL ON TRAINING DATA
# ==================================================

print("\nTraining Logistic Regression pipeline...")

model.fit(
    X_train,
    y_train
)


# ==================================================
# 8. SELECT THRESHOLD USING VALIDATION DATA
# ==================================================

print("\nSelecting threshold using validation data...")

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]

threshold_results = []

threshold_candidates = np.arange(
    0.10,
    0.91,
    0.01
)

for threshold in threshold_candidates:

    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    threshold_results.append({
        "threshold": float(threshold),
        "f1_score": float(score),
    })

best_threshold_result = max(
    threshold_results,
    key=lambda item: item["f1_score"]
)

best_threshold = best_threshold_result["threshold"]

print(f"Selected threshold: {best_threshold:.2f}")
print(
    "Validation F1: "
    f"{best_threshold_result['f1_score']:.4f}"
)


# ==================================================
# 9. FINAL EVALUATION ON UNTOUCHED TEST DATA
# ==================================================

print("\nEvaluating on untouched test data...")

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

average_precision = average_precision_score(
    y_test,
    test_probabilities
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

print("\nFinal Classification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    )
)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")
print(f"Specificity: {specificity:.4f}")


# ==================================================
# 10. SAVE TRAINED MODEL
# ==================================================

joblib.dump(
    model,
    MODEL_PATH
)

print(f"\nModel saved: {MODEL_PATH}")


# ==================================================
# 11. SAVE MODEL METADATA
# ==================================================

metadata = {
    "model_name": "Logistic Regression",
    "model_type": "Pipeline",
    "features": FEATURES,
    "target": TARGET,
    "random_state": RANDOM_STATE,
    "training_rows": int(len(X_train)),
    "validation_rows": int(len(X_validation)),
    "testing_rows": int(len(X_test)),
    "cross_validation_folds": 5,
    "cv_f1_mean": round(cv_mean, 6),
    "cv_f1_std": round(cv_std, 6),
    "selected_threshold": round(best_threshold, 4),
    "validation_f1": round(
        best_threshold_result["f1_score"], 6
    ),
    "test_accuracy": round(float(accuracy), 6),
    "test_precision": round(float(precision), 6),
    "test_recall": round(float(recall), 6),
    "test_f1": round(float(f1), 6),
    "test_roc_auc": round(float(roc_auc), 6),
    "test_average_precision": round(
        float(average_precision), 6
    ),
    "test_specificity": round(float(specificity), 6),
    "confusion_matrix": [
        [int(tn), int(fp)],
        [int(fn), int(tp)],
    ],
    "data_type": "synthetic demonstration data",
}

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)

print(f"Metadata saved: {METADATA_PATH}")


# ==================================================
# 12. SAVE FINAL TEST PREDICTIONS
# ==================================================

test_results = X_test.copy()

test_results["actual"] = y_test.to_numpy()

test_results["fraud_probability"] = test_probabilities

test_results["prediction"] = test_predictions

test_results["threshold"] = best_threshold

predictions_path = REPORT_DIR / "test_predictions.csv"

test_results.to_csv(
    predictions_path,
    index=False
)

print(f"Test predictions saved: {predictions_path}")


# ==================================================
# 13. SAVE FINAL TEST METRICS
# ==================================================

metrics = {
    "accuracy": round(float(accuracy), 4),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1_score": round(float(f1), 4),
    "roc_auc": round(float(roc_auc), 4),
    "average_precision": round(float(average_precision), 4),
    "specificity": round(float(specificity), 4),
    "threshold": round(float(best_threshold), 4),
    "confusion_matrix": [
        [int(tn), int(fp)],
        [int(fn), int(tp)],
    ],
    "test_rows": int(len(X_test)),
    "model_type": "Pipeline",
    "model_name": "Logistic Regression",
    "cv_f1_mean": round(cv_mean, 4),
    "cv_f1_std": round(cv_std, 4),
}

metrics_path = REPORT_DIR / "metrics.json"

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)

print(f"Test metrics saved: {metrics_path}")

print("\n======================================")
print("FRAUDLENS AI MODEL TRAINING COMPLETED")
print("======================================")
