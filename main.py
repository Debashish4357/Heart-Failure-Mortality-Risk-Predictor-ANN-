
import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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

# Configure CORS to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    try:
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
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again.",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

