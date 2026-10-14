
import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Load the dataset
df = pd.read_csv(
    "data/heart_failure_clinical_records_dataset.csv"
)

# 2. Prepare features and target
X = df.drop(columns=["DEATH_EVENT", "time"])
y = df["DEATH_EVENT"]

# 3. Create the complete pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

# 4. Train on all available data for the MVP
model.fit(X, y)

# 5. Save model and feature configuration
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/logistic_regression_pipeline.joblib"
)

joblib.dump(
    list(X.columns),
    "models/feature_names.joblib"
)

print("Model trained and saved successfully.")
print("Training samples:", len(X))
print("Input features:", len(X.columns))
print("Model: models/logistic_regression_pipeline.joblib")
print("Features: models/feature_names.joblib")
