
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

