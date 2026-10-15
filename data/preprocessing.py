
import os
import sys
import pandas as pd

from sklearn.model_selection import StratifiedKFold

# ============================================================
# PROJECT PATHS
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

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop(columns=["DEATH_EVENT", "time"])
y = df["DEATH_EVENT"]

# ============================================================
# STRATIFIED 5-FOLD CROSS-VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nTesting preprocessing pipeline...")

for fold, (train_index, val_index) in enumerate(
    skf.split(X, y), start=1
):
    X_train = X.iloc[train_index]
    X_val = X.iloc[val_index]

    # Create a fresh pipeline for this fold.
    # Only the scaler is used here to test preprocessing.
    model = create_logistic_model()
    scaler = model.named_steps["scaler"]

    # Fit preprocessing only on training data
    X_train_processed = scaler.fit_transform(X_train)

    # Transform validation data using the fitted scaler
    X_val_processed = scaler.transform(X_val)

    print(f"\nFold {fold}")
    print("Train shape:", X_train_processed.shape)
    print("Validation shape:", X_val_processed.shape)

print("\nPreprocessing pipeline completed successfully.")


