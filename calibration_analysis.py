
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

# ============================================================
# STEP 8.12 — LOGISTIC REGRESSION CALIBRATION ANALYSIS
# ============================================================

SEED = 42
N_BINS = 5

# 1. Load dataset
df = pd.read_csv(
    "data/heart_failure_clinical_records_dataset.csv"
)

# Exclude target and leakage-prone 'time'
X = df.drop(columns=["DEATH_EVENT", "time"])
y = df["DEATH_EVENT"]

# 2. Generate out-of-fold probabilities
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

oof_probabilities = np.zeros(len(X))

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y), start=1
):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=SEED
        )
    )

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]

    model.fit(X_train, y_train)

    oof_probabilities[test_idx] = (
        model.predict_proba(X_test)[:, 1]
    )

    print(f"Fold {fold}: predictions generated.")

# 3. Calculate Brier score and AUROC
brier = brier_score_loss(y, oof_probabilities)
auroc = roc_auc_score(y, oof_probabilities)

# 4. Calculate calibration curve points
fraction_positive, mean_predicted = calibration_curve(
    y,
    oof_probabilities,
    n_bins=N_BINS,
    strategy="uniform"
)

# 5. Save calibration data
os.makedirs("results", exist_ok=True)

calibration_df = pd.DataFrame({
    "Mean_Predicted_Probability": mean_predicted,
    "Observed_Positive_Fraction": fraction_positive
})

calibration_df.to_csv(
    "results/logistic_calibration_curve.csv",
    index=False
)

# Save each patient's out-of-fold probability
prediction_df = pd.DataFrame({
    "Actual_DEATH_EVENT": y.to_numpy(),
    "OOF_Probability": oof_probabilities
})

prediction_df.to_csv(
    "results/logistic_oof_predictions.csv",
    index=False
)

# 6. Plot calibration curve
plt.figure(figsize=(7, 6))

plt.plot(
    [0, 1], [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    label="Logistic Regression"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Fraction of Positive Cases")
plt.title("Logistic Regression Calibration Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/logistic_calibration_curve.png",
    dpi=150
)

plt.close()

# 7. Print results
print("\n" + "=" * 55)
print("CALIBRATION RESULTS")
print("=" * 55)

print(f"Brier Score: {brier:.4f}")
print(f"OOF AUROC:   {auroc:.4f}")

print("\nCalibration curve data:")
print(calibration_df.to_string(index=False))

print("\nSaved files:")
print("results/logistic_calibration_curve.csv")
print("results/logistic_calibration_curve.png")
print("results/logistic_oof_predictions.csv")







