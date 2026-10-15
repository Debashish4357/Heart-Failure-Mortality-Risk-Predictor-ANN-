
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# ============================================================
# PROJECT PATHS AND REUSABLE MODEL
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
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
N_SPLITS = 5

# ============================================================
# STEP 1 — LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

# Exclude target and leakage-prone 'time' feature
X = df.drop(columns=["DEATH_EVENT", "time"])
y = df["DEATH_EVENT"]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# Use reproducible stratified folds
skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

# ============================================================
# STEP 2 — LOGISTIC REGRESSION 5-FOLD CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION — 5-FOLD CROSS-VALIDATION")
print("=" * 60)

fold_results = []

for fold, (train_index, val_index) in enumerate(
    skf.split(X, y), start=1
):
    X_train = X.iloc[train_index]
    X_val = X.iloc[val_index]

    y_train = y.iloc[train_index]
    y_val = y.iloc[val_index]

    # Create a fresh scaler + Logistic Regression pipeline
    model = create_logistic_model()

    # Fit preprocessing and model on training data only
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    auroc = roc_auc_score(y_val, y_prob)

    precision = precision_score(
        y_val, y_pred, zero_division=0
    )

    sensitivity = recall_score(
        y_val, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_val, y_pred, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else 0.0
    )

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

    print(f"\nFold {fold}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# Summarize fold results
fold_results_df = pd.DataFrame(fold_results)

metrics = [
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1_Score",
    "AUROC"
]

print("\n===== Logistic Regression Results =====")

for metric in metrics:
    print(
        f"Mean {metric}: "
        f"{fold_results_df[metric].mean():.4f}"
    )

fold_results_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "train_logistic_5fold_results.csv"
    ),
    index=False
)

# ============================================================
# STEP 3 — OUT-OF-FOLD (OOF) PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("OUT-OF-FOLD PREDICTIONS")
print("=" * 60)

# Each patient receives a probability from a model that
# was not trained on that patient's fold.
oof_probabilities = np.full(len(X), np.nan)

for fold, (train_index, val_index) in enumerate(
    skf.split(X, y), start=1
):
    print(f"\nProcessing Fold {fold}...")

    X_train = X.iloc[train_index]
    X_val = X.iloc[val_index]
    y_train = y.iloc[train_index]

    model = create_logistic_model()
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_val)[:, 1]

    # Store probabilities at original patient positions
    oof_probabilities[val_index] = y_prob

    print(f"Stored {len(y_prob)} validation predictions")

oof_probabilities = pd.Series(oof_probabilities)

print("\nOOF prediction count:", len(oof_probabilities))
print(
    "Missing OOF predictions:",
    oof_probabilities.isna().sum()
)
print("Minimum probability:", oof_probabilities.min())
print("Maximum probability:", oof_probabilities.max())

print("\nFirst 10 OOF probabilities:")
print(oof_probabilities.head(10))

# Save OOF predictions
oof_probabilities_array = oof_probabilities.to_numpy()

oof_predictions_df = pd.DataFrame({
    "Actual_DEATH_EVENT": y.to_numpy(),
    "OOF_Probability": oof_probabilities_array
})

oof_predictions_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "logistic_oof_predictions.csv"
    ),
    index=False
)

# ============================================================
# STEP 4 — THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

thresholds = [
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70
]

print("\nThreshold Results:")
print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Sensitivity':<15}"
    f"{'Specificity':<15}"
)

for threshold in thresholds:
    y_pred_threshold = (
        oof_probabilities_array >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, y_pred_threshold, labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y, y_pred_threshold)

    sensitivity = (
        tp / (tp + fn) if (tp + fn) > 0 else 0.0
    )

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else 0.0
    )

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{sensitivity:<15.4f}"
        f"{specificity:<15.4f}"
    )

# ============================================================
# STEP 5 — VALIDATION-DERIVED THRESHOLD
# Youden's J Statistic
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION-DERIVED THRESHOLD")
print("=" * 60)

threshold_results = []

for threshold in thresholds:
    y_pred_threshold = (
        oof_probabilities_array >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, y_pred_threshold, labels=[0, 1]
    ).ravel()

    sensitivity = (
        tp / (tp + fn) if (tp + fn) > 0 else 0.0
    )

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else 0.0
    )

    youden_j = sensitivity + specificity - 1

    threshold_results.append({
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youden_j": youden_j
    })

print("\nThreshold Evaluation:")

for result in threshold_results:
    print(
        f"Threshold: {result['threshold']:.2f} | "
        f"Sensitivity: {result['sensitivity']:.4f} | "
        f"Specificity: {result['specificity']:.4f} | "
        f"Youden J: {result['youden_j']:.4f}"
    )

best_threshold_result = max(
    threshold_results,
    key=lambda result: result["youden_j"]
)

final_threshold = best_threshold_result["threshold"]

print("\nSelected threshold:", f"{final_threshold:.2f}")
print(
    "Sensitivity:",
    f"{best_threshold_result['sensitivity']:.4f}"
)
print(
    "Specificity:",
    f"{best_threshold_result['specificity']:.4f}"
)
print(
    "Youden J:",
    f"{best_threshold_result['youden_j']:.4f}"
)

# Save threshold analysis
pd.DataFrame(threshold_results).to_csv(
    os.path.join(
        RESULTS_DIR,
        "logistic_threshold_analysis.csv"
    ),
    index=False
)

# ============================================================
# STEP 6 — CALIBRATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CALIBRATION ANALYSIS")
print("=" * 60)

brier_score = brier_score_loss(
    y, oof_probabilities_array
)

oof_auroc = roc_auc_score(
    y, oof_probabilities_array
)

prob_true, prob_pred = calibration_curve(
    y,
    oof_probabilities_array,
    n_bins=5,
    strategy="uniform"
)

calibration_df = pd.DataFrame({
    "Mean_Predicted_Probability": prob_pred,
    "Observed_Positive_Fraction": prob_true
})

print("\nCalibration values:")
print(calibration_df.to_string(index=False))

calibration_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "train_logistic_calibration_curve.csv"
    ),
    index=False
)

# Calibration plot
plt.figure(figsize=(7, 6))

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label="Logistic Regression"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Frequency")
plt.title("Calibration Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "train_logistic_calibration_curve.png"
    ),
    dpi=150
)
plt.close()

print("\nBrier Score:", f"{brier_score:.4f}")
print("OOF AUROC:", f"{oof_auroc:.4f}")

# ============================================================
# STEP 7 — FINAL OOF EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL LOGISTIC REGRESSION OOF EVALUATION")
print("=" * 60)

oof_predictions = (
    oof_probabilities_array >= final_threshold
).astype(int)

oof_accuracy = accuracy_score(y, oof_predictions)

tn, fp, fn, tp = confusion_matrix(
    y, oof_predictions, labels=[0, 1]
).ravel()

oof_sensitivity = (
    tp / (tp + fn) if (tp + fn) > 0 else 0.0
)

oof_specificity = (
    tn / (tn + fp) if (tn + fp) > 0 else 0.0
)

oof_precision = precision_score(
    y, oof_predictions, zero_division=0
)

oof_f1 = f1_score(
    y, oof_predictions, zero_division=0
)

print("\nSelected Threshold:", f"{final_threshold:.2f}")

print("\nConfusion Matrix:")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

print("\nFinal OOF Metrics:")
print(f"Accuracy:    {oof_accuracy:.4f}")
print(f"AUROC:       {oof_auroc:.4f}")
print(f"Sensitivity: {oof_sensitivity:.4f}")
print(f"Specificity: {oof_specificity:.4f}")
print(f"Precision:   {oof_precision:.4f}")
print(f"F1 Score:    {oof_f1:.4f}")
print(f"Brier Score: {brier_score:.4f}")

# ============================================================
# STEP 8 — ROC AND PRECISION-RECALL CURVES
# ============================================================

print("\n" + "=" * 60)
print("ROC AND PRECISION-RECALL CURVES")
print("=" * 60)

# ROC curve
fpr, tpr, roc_thresholds = roc_curve(
    y, oof_probabilities_array
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUROC = {oof_auroc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "logistic_roc_curve.png"),
    dpi=150
)
plt.close()

# Precision-recall curve
precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(
        y, oof_probabilities_array
    )
)

average_precision = average_precision_score(
    y, oof_probabilities_array
)

plt.figure(figsize=(7, 6))

plt.plot(
    recall_values,
    precision_values,
    label=f"Logistic Regression (AP = {average_precision:.3f})"
)

plt.axhline(
    y=y.mean(),
    linestyle="--",
    label="Positive-class prevalence"
)

plt.xlabel("Recall (Sensitivity)")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Logistic Regression")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "logistic_precision_recall_curve.png"
    ),
    dpi=150
)
plt.close()

print(f"\nOOF AUROC: {oof_auroc:.4f}")
print(f"Average Precision: {average_precision:.4f}")
print("Evaluation curves saved successfully.")

# ============================================================
# STEP 9 — SAVE FINAL EVALUATION METRICS
# ============================================================

print("\n" + "=" * 60)
print("SAVING LOGISTIC REGRESSION RESULTS")
print("=" * 60)

results = {
    "Model": "Logistic Regression",
    "Evaluation": "Out-of-Fold",
    "Threshold": final_threshold,
    "Accuracy": oof_accuracy,
    "AUROC": oof_auroc,
    "Sensitivity": oof_sensitivity,
    "Specificity": oof_specificity,
    "Precision": oof_precision,
    "F1_Score": oof_f1,
    "Brier_Score": brier_score,
    "Average_Precision": average_precision
}

results_df = pd.DataFrame([results])

results_path = os.path.join(
    RESULTS_DIR,
    "logistic_regression_results.csv"
)

results_df.to_csv(results_path, index=False)

print("\nBaseline results saved successfully.")
print("File:", results_path)
print("\nSaved metrics:")
print(results_df.to_string(index=False))

print("\nAll training and evaluation steps completed.")


