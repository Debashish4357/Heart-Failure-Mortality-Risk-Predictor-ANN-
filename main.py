
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_pipeline.joblib"
FEATURES_PATH = BASE_DIR / "models" / "feature_names.joblib"

# Load the trained model and expected feature names
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

app = FastAPI(
    title="Heart Failure Mortality Predictor",
    description="Educational machine-learning prototype, not a clinical tool.",
    version="1.0.0",
)


class PatientData(BaseModel):
    age: float
    anaemia: int
    creatinine_phosphokinase: float
    diabetes: int
    ejection_fraction: float
    high_blood_pressure: int
    platelets: float
    serum_creatinine: float
    serum_sodium: float
    sex: int
    smoking: int


@app.get("/")
def home():
    return {"message": "Heart Failure Mortality Predictor API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_count": len(feature_names),
    }


@app.post("/predict")
def predict(patient: PatientData):
    # Convert input into the exact feature order used during training
    patient_data = patient.model_dump()
    input_data = pd.DataFrame(
        [[patient_data[name] for name in feature_names]],
        columns=feature_names,
    )

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    return {
        "predicted_class": prediction,
        "predicted_label": (
            "Death event predicted" if prediction == 1
            else "No death event predicted"
        ),
        "predicted_probability": round(probability, 4),
        "notice": "Educational prototype only; not for medical decisions.",
    }


