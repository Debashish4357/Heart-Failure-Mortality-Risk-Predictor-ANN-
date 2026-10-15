import os
import sys
import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

# Load the saved model and feature names
model = joblib.load(
    os.path.join(
        PROJECT_ROOT,
        "models",
        "logistic_regression_pipeline.joblib"
    )
)

feature_names = joblib.load(
    os.path.join(PROJECT_ROOT, "models", "feature_names.joblib")
)

# Load one sample from the dataset
df = pd.read_csv(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "heart_failure_clinical_records_dataset.csv"
    )
)

sample = df[feature_names].iloc[[0]]

# Generate prediction
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("Model loaded successfully.")
print("Number of features:", len(feature_names))
print("Predicted class:", prediction)
print("Predicted probability of death event:", round(probability, 4))