
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

