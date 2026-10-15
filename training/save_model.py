
import os
import sys
import joblib
import pandas as pd

# 1. Get the project root directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# 2. Add project root to Python's import path
sys.path.insert(0, PROJECT_ROOT)

from models.logistic_model import create_logistic_model


# 3. Load the dataset
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "heart_failure_clinical_records_dataset.csv"
)

df = pd.read_csv(DATA_PATH)

# 4. Prepare features and target
X = df.drop(columns=["DEATH_EVENT", "time"])
y = df["DEATH_EVENT"]

# 5. Create and train the Logistic Regression pipeline
model = create_logistic_model()
model.fit(X, y)

# 6. Define paths for saved files
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "logistic_regression_pipeline.joblib"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "feature_names.joblib"
)

# 7. Save the trained model and feature names
joblib.dump(model, MODEL_PATH)
joblib.dump(list(X.columns), FEATURES_PATH)

# 8. Display confirmation
print("Model trained and saved successfully.")
print("Training samples:", len(X))
print("Input features:", len(X.columns))
print("Model:", MODEL_PATH)
print("Features:", FEATURES_PATH)
