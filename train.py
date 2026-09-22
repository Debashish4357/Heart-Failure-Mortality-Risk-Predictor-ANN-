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