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
# STEP 3 — OUT-OF-FOLD PREDICTIONS
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
# STEP 4 — THRESHOLD ANALYSIS
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
# STEP 4.6 — VALIDATION-DERIVED THRESHOLD
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