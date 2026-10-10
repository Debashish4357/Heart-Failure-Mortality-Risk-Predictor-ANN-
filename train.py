import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)


# Load dataset
df = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")


# Separate features and target
X = df.drop(columns=["DEATH_EVENT", "time"])
y = df["DEATH_EVENT"]


# Stratified 5-Fold Cross Validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nLogistic Regression - 5-Fold Cross Validation")


# Store results
auroc_scores = []
accuracy_scores = []
sensitivity_scores = []
specificity_scores = []


for fold, (train_index, val_index) in enumerate(skf.split(X, y), start=1):

    X_train = X.iloc[train_index]
    X_val = X.iloc[val_index]

    y_train = y.iloc[train_index]
    y_val = y.iloc[val_index]


    # Scale features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)


    # Create Logistic Regression model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )


    # Train model
    model.fit(X_train_scaled, y_train)


    # Predictions
    y_pred = model.predict(X_val_scaled)
    y_prob = model.predict_proba(X_val_scaled)[:, 1]


    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    auroc = roc_auc_score(y_val, y_prob)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        y_pred
    ).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)


    # Store results
    accuracy_scores.append(accuracy)
    auroc_scores.append(auroc)
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)


    print(f"\nFold {fold}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")


# Average results
print("\n===== Logistic Regression Results =====")

print(f"Mean Accuracy: {sum(accuracy_scores) / len(accuracy_scores):.4f}")
print(f"Mean AUROC: {sum(auroc_scores) / len(auroc_scores):.4f}")
print(f"Mean Sensitivity: {sum(sensitivity_scores) / len(sensitivity_scores):.4f}")
print(f"Mean Specificity: {sum(specificity_scores) / len(specificity_scores):.4f}")



# ============================================================
# OUT-OF-FOLD PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("OUT-OF-FOLD PREDICTIONS")
print("=" * 60)


# Store one probability prediction for every patient
oof_probabilities = [None] * len(X)


# 5-Fold Cross Validation
for fold, (train_index, val_index) in enumerate(
    skf.split(X, y),
    start=1
):

    print(f"\nProcessing Fold {fold}...")

    # Split data
    X_train = X.iloc[train_index]
    X_val = X.iloc[val_index]

    y_train = y.iloc[train_index]

    # Scale using training data only
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Create Logistic Regression model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    # Train model
    model.fit(
        X_train_scaled,
        y_train
    )

    # Generate probability predictions
    y_prob = model.predict_proba(
        X_val_scaled
    )[:, 1]

    # Store predictions at their original patient positions
    for index, probability in zip(
        val_index,
        y_prob
    ):
        oof_probabilities[index] = probability

    print(
        f"Stored {len(y_prob)} validation predictions"
    )


# Convert to pandas Series
oof_probabilities = pd.Series(
    oof_probabilities
)


# Verify results
print("\nOOF prediction count:", len(oof_probabilities))
print(
    "Missing OOF predictions:",
    oof_probabilities.isna().sum()
)

print(
    "Minimum probability:",
    oof_probabilities.min()
)

print(
    "Maximum probability:",
    oof_probabilities.max()
)

print("\nFirst 10 OOF probabilities:")
print(oof_probabilities.head(10))



# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)


# Thresholds to evaluate
thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


print("\nThreshold Results:")
print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Sensitivity':<15}"
    f"{'Specificity':<15}"
)


# Evaluate each threshold
for threshold in thresholds:

    # Convert probabilities into class predictions
    y_pred_threshold = (
        oof_probabilities >= threshold
    ).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y,
        y_pred_threshold
    ).ravel()

    # Calculate metrics
    accuracy = accuracy_score(
        y,
        y_pred_threshold
    )

    sensitivity = tp / (tp + fn)

    specificity = tn / (tn + fp)

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{sensitivity:<15.4f}"
        f"{specificity:<15.4f}"
    )


    

    # ============================================================
# VALIDATION-DERIVED THRESHOLD
# Using Youden's J Statistic
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION-DERIVED THRESHOLD")
print("=" * 60)


threshold_results = []


for threshold in thresholds:

    # Convert probabilities into predictions
    y_pred_threshold = (
        oof_probabilities >= threshold
    ).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y,
        y_pred_threshold
    ).ravel()

    # Metrics
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Youden's J
    youden_j = sensitivity + specificity - 1

    threshold_results.append({
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youden_j": youden_j
    })


# Find threshold with highest Youden's J
best_threshold_result = max(
    threshold_results,
    key=lambda x: x["youden_j"]
)


print("\nThreshold Evaluation:")

for result in threshold_results:

    print(
        f"Threshold: {result['threshold']:.2f} | "
        f"Sensitivity: {result['sensitivity']:.4f} | "
        f"Specificity: {result['specificity']:.4f} | "
        f"Youden J: {result['youden_j']:.4f}"
    )


print("\nSelected validation-derived threshold:")

print(
    f"Threshold: "
    f"{best_threshold_result['threshold']:.2f}"
)

print(
    f"Sensitivity: "
    f"{best_threshold_result['sensitivity']:.4f}"
)

print(
    f"Specificity: "
    f"{best_threshold_result['specificity']:.4f}"
)

print(
    f"Youden J: "
    f"{best_threshold_result['youden_j']:.4f}"
)




# ============================================================
# CALIBRATION
# ============================================================

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt


print("\n" + "=" * 60)
print("CALIBRATION ANALYSIS")
print("=" * 60)


# ============================================================
# Prepare OOF Probabilities
# ============================================================

# Convert OOF predictions to NumPy array
oof_probabilities_array = oof_probabilities.to_numpy()

print("\nOOF probabilities prepared.")
print("Number of predictions:", len(oof_probabilities_array))


# ============================================================
# Calibration Curve
# ============================================================

prob_true, prob_pred = calibration_curve(
    y,
    oof_probabilities_array,
    n_bins=5,
    strategy="uniform"
)


print("\nCalibration values:")

for predicted, actual in zip(
    prob_pred,
    prob_true
):
    print(
        f"Predicted probability: {predicted:.4f} | "
        f"Observed frequency: {actual:.4f}"
    )


# Plot calibration curve
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
plt.grid()

plt.tight_layout()
plt.show()



# ============================================================
#  BRIER SCORE
# ============================================================

brier_score = brier_score_loss(
    y,
    oof_probabilities_array
)

print("\nBrier Score:", round(brier_score, 4))





# ============================================================
# STEP 6 — FINAL ML EVALUATION
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


print("\n" + "=" * 60)
print("FINAL LOGISTIC REGRESSION OOF EVALUATION")
print("=" * 60)


# ============================================================
# STEP 6.1 — OOF Class Predictions
# ============================================================

# Use the validation-derived threshold selected earlier
final_threshold = best_threshold_result["threshold"]

oof_predictions = (
    oof_probabilities_array >= final_threshold
).astype(int)


# ============================================================
# STEP 6.2 — Accuracy
# ============================================================

oof_accuracy = accuracy_score(
    y,
    oof_predictions
)


# ============================================================
# STEP 6.3 — AUROC
# ============================================================

oof_auroc = roc_auc_score(
    y,
    oof_probabilities_array
)


# ============================================================
# STEP 6.4 — Confusion Matrix
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y,
    oof_predictions
).ravel()


# ============================================================
# STEP 6.5 — Sensitivity and Specificity
# ============================================================

oof_sensitivity = tp / (tp + fn)

oof_specificity = tn / (tn + fp)


# ============================================================
# STEP 6.6 — Precision
# ============================================================

oof_precision = precision_score(
    y,
    oof_predictions,
    zero_division=0
)


# ============================================================
# STEP 6.7 — F1 Score
# ============================================================

oof_f1 = f1_score(
    y,
    oof_predictions,
    zero_division=0
)


# ============================================================
# Final Results
# ============================================================

print("\nSelected Threshold:", round(final_threshold, 4))

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
# STEP 6.9 — ROC AND PRECISION-RECALL CURVES
# ============================================================

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


print("\n" + "=" * 60)
print("ROC AND PRECISION-RECALL CURVES")
print("=" * 60)


# ------------------------------------------------------------
# 6.9.1 — ROC Curve
# ------------------------------------------------------------

fpr, tpr, roc_thresholds = roc_curve(
    y,
    oof_probabilities_array
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
plt.grid()
plt.tight_layout()
plt.show()



# ------------------------------------------------------------
# 6.9.2 — Precision-Recall Curve
# ------------------------------------------------------------

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(
        y,
        oof_probabilities_array
    )
)

average_precision = average_precision_score(
    y,
    oof_probabilities_array
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
plt.grid()
plt.tight_layout()
plt.show()


print(f"\nOOF AUROC: {oof_auroc:.4f}")
print(f"Average Precision: {average_precision:.4f}")

print("\nEvaluation curves generated successfully.")





