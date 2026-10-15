
import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# PROJECT PATHS AND IMPORTS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from models.logistic_model import create_logistic_model

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "heart_failure_clinical_records_dataset.csv"
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ============================================================
# STEP 8.9 — LOGISTIC REGRESSION 5-FOLD CROSS-VALIDATION
# ============================================================

SEED = 42

# Load dataset
df = pd.read_csv(DATA_PATH)

# Prepare features and target
X = df.drop(columns=["DEATH_EVENT", "time"])
y = df["DEATH_EVENT"]

# Create results folder
os.makedirs(RESULTS_DIR, exist_ok=True)

# Use the same outer folds as the ANN
outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

fold_results = []

print("=" * 65)
print("LOGISTIC REGRESSION — 5-FOLD CROSS-VALIDATION")
print("=" * 65)

for fold, (outer_train_idx, test_idx) in enumerate(
    outer_cv.split(X, y), start=1
):
    print(f"\n{'=' * 25} FOLD {fold} {'=' * 25}")

    # Separate outer training and test data
    X_outer_train = X.iloc[outer_train_idx]
    y_outer_train = y.iloc[outer_train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    # Use the same internal split as the ANN
    inner_train_idx, inner_val_idx = train_test_split(
        np.arange(len(X_outer_train)),
        test_size=0.20,
        random_state=SEED + fold,
        stratify=y_outer_train
    )

    X_train = X_outer_train.iloc[inner_train_idx]
    y_train = y_outer_train.iloc[inner_train_idx]

    # Create the reusable pipeline.
    # StandardScaler is included inside this pipeline.
    model = create_logistic_model()

    # Train using only the training split
    model.fit(X_train, y_train)

    # Generate predictions and probabilities
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    y_true = y_test.to_numpy()

    # Calculate metrics
    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true, predictions, zero_division=0
    )

    sensitivity = recall_score(
        y_true, predictions, zero_division=0
    )

    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else 0.0
    )

    auroc = roc_auc_score(y_true, probabilities)

    # Store fold results
    fold_results.append({
        "Fold": fold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1_Score": f1,
        "AUROC": auroc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(
        f"Confusion matrix: TN={tn}, FP={fp}, "
        f"FN={fn}, TP={tp}"
    )


# ============================================================
# STEP 8.10 — SAVE AND SUMMARIZE RESULTS
# ============================================================

results_df = pd.DataFrame(fold_results)

results_path = os.path.join(
    RESULTS_DIR,
    "logistic_regression_5fold_results.csv"
)

results_df.to_csv(results_path, index=False)

metrics = [
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1_Score",
    "AUROC"
]

summary_df = pd.DataFrame({
    "Metric": metrics,
    "Mean": [
        results_df[m].mean() for m in metrics
    ],
    "Standard_Deviation": [
        results_df[m].std(ddof=1) for m in metrics
    ]
})

summary_path = os.path.join(
    RESULTS_DIR,
    "logistic_regression_5fold_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 65)
print("LOGISTIC REGRESSION 5-FOLD SUMMARY")
print("=" * 65)

print(
    summary_df.to_string(
        index=False,
        float_format="%.4f"
    )
)

print("\nFold results saved to:")
print(results_path)

print("\nSummary saved to:")
print(summary_path)



