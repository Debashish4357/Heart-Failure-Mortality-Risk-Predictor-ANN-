import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Load dataset
df = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")


# Features and target
X = df.drop(columns=["DEATH_EVENT", "time"])
y = df["DEATH_EVENT"]


# Leakage-safe preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ("scaler", StandardScaler())
])


# Stratified 5-Fold Cross Validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)