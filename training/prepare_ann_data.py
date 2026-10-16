
import os
import sys
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# STEP 8.1 — PREPARE ANN DATA
# ============================================================

# Load dataset
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "heart_failure_clinical_records_dataset.csv"
)
df = pd.read_csv(DATA_PATH)

# Separate features and target
# Exclude 'time' because it is leakage-prone
X = df.drop(columns=["DEATH_EVENT", "time"])
y = df["DEATH_EVENT"]


# Create Stratified 5-Fold CV
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Use the first fold for this preparation test
train_index, val_index = next(skf.split(X, y))

X_train = X.iloc[train_index]
X_val = X.iloc[val_index]

y_train = y.iloc[train_index]
y_val = y.iloc[val_index]


# Fit scaler only on training data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# Convert features to PyTorch tensors
X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val_scaled,
    dtype=torch.float32
)


# Convert targets to PyTorch tensors
y_train_tensor = torch.tensor(
    y_train.to_numpy(),
    dtype=torch.float32
).reshape(-1, 1)

y_val_tensor = torch.tensor(
    y_val.to_numpy(),
    dtype=torch.float32
).reshape(-1, 1)


# Verify tensor shapes
print("=" * 50)
print("ANN DATA PREPARATION TEST")
print("=" * 50)

print("Training features:", X_train_tensor.shape)
print("Training targets:", y_train_tensor.shape)

print("Validation features:", X_val_tensor.shape)
print("Validation targets:", y_val_tensor.shape)

print("\nFeature tensor dtype:", X_train_tensor.dtype)
print("Target tensor dtype:", y_train_tensor.dtype)

print("\nANN data preparation completed successfully.")


















