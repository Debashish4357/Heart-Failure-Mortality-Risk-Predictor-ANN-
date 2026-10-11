
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ============================================================
# STEP 8.1 — PREPARE ANN DATA
# ============================================================

# Load dataset
df = pd.read_csv(
    "data/heart_failure_clinical_records_dataset.csv"
)

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






