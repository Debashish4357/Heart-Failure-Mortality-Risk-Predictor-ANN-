# Heart Failure Mortality Risk Predictor (ANN)

A full-stack, machine-learning-driven clinical decision-support prototype that estimates mortality risk for heart failure patients at the point of clinical assessment.

## Overview

This project develops an end-to-end machine learning pipeline and web application for predicting the probability of a mortality event in heart failure patients.

The project follows a structured Advanced Data Science workflow:

**Data Quality → EDA → Leakage Assessment → Feature Preparation → Stratified Cross-Validation → Scaling → Logistic Regression Baseline → ANN → Evaluation → Calibration → API → Frontend**

The primary machine learning task is **binary classification**, where:

- `0` = No death event
- `1` = Death event

The system is designed as a **clinical decision-support prototype** and is not intended to replace professional clinical judgment.

---

## Key Features

### 🔒 Strict Data Leakage Prevention

The dataset contains a `time` feature representing follow-up duration.

Because this information is retrospective for the defined prediction timepoint, it is excluded from the model feature set.

The model currently uses **11 clinical input features**.

### 📊 Data Quality & EDA

The dataset preparation process includes:

- Dataset shape and column inspection
- Missing-value checking
- Duplicate checking
- Data-type inspection
- Binary-value validation
- Numerical range inspection
- Potential outlier detection using the IQR method
- Target-distribution analysis
- Numerical feature-distribution analysis
- Feature-vs-target exploratory analysis

No potential outliers are automatically removed solely because they were detected.

### ⚖️ Feature and Target Preparation

The dataset is separated into:

```text
X → 11 clinical input features
y → DEATH_EVENT
```

The `time` column is excluded from `X` to prevent leakage.

The binary clinical variables are already represented as `0/1`, so additional categorical encoding is not required for the current dataset.

### 🔄 Robust Validation

The project uses **Stratified 5-Fold Cross-Validation**.

Each fold maintains a similar proportion of the two target classes.

Feature preprocessing is performed within each fold to prevent validation information from leaking into training.

### 📏 Feature Scaling

Numerical features are standardized using `StandardScaler`.

The scaler is:

1. Fitted only on the training portion of each fold.
2. Applied to the corresponding validation portion using the training parameters.

This creates a leakage-safe preprocessing process.

### 📈 Logistic Regression Baseline

Before developing the ANN, Logistic Regression is used as the mandatory baseline model.

Current 5-fold baseline results:

| Metric | Mean Score |
|---|---:|
| Accuracy | **75.60%** |
| AUROC | **77.17%** |
| Sensitivity | **48.11%** |
| Specificity | **88.67%** |

The baseline provides a reference point for evaluating the eventual ANN.

### 🧠 Deep Artificial Neural Network

The main model will be developed using **PyTorch**.

The ANN will be trained and evaluated using the same validation methodology as the Logistic Regression baseline.

The architecture will be selected through experimentation while controlling overfitting using techniques such as:

- Dropout
- Weight decay
- Early stopping

### 🎯 Probability-First Prediction

The primary model output is a **mortality probability** rather than an arbitrary risk category.

For example:

```text
Predicted mortality probability: 0.78
```

Arbitrary labels such as LOW/MEDIUM/HIGH are not used unless thresholds are derived and justified using validation data.

### 📐 Probability Calibration

The final ML pipeline will evaluate whether predicted probabilities correspond appropriately to observed outcomes.

Calibration will use:

- Out-of-fold predictions
- Calibration/reliability curves
- Brier score

Calibration will be performed without evaluating the model on the same data used to fit the calibration procedure.

---

# Dataset

The project uses the **Heart Failure Clinical Records Dataset**.

Dataset characteristics:

- **299 patients**
- **13 original columns**
- Binary mortality target: `DEATH_EVENT`
- 12 original predictive/time-related variables before excluding `time`

After removing `time` and the target column, the model uses **11 input features**:

```text
age
anaemia
creatinine_phosphokinase
diabetes
ejection_fraction
high_blood_pressure
platelets
serum_creatinine
serum_sodium
sex
smoking
```

Target:

```text
DEATH_EVENT
```

---

# Machine Learning Pipeline

```text
Heart Failure Dataset
        │
        ▼
Data Quality Checks
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Leakage Assessment
        │
        ├── Remove `time`
        │
        ▼
Feature / Target Preparation
        │
        ▼
Stratified 5-Fold Cross-Validation
        │
        ▼
Leakage-Safe Scaling
        │
        ▼
Logistic Regression Baseline
        │
        ▼
PyTorch ANN
        │
        ▼
Model Evaluation
        │
        ├── AUROC
        ├── Sensitivity
        ├── Specificity
        ├── Accuracy
        ├── Precision
        ├── Recall
        └── F1
        │
        ▼
Out-of-Fold Predictions
        │
        ▼
Calibration
        │
        ├── Calibration Curve
        └── Brier Score
        │
        ▼
Final Model Artifact
        │
        ▼
FastAPI Prediction Service
        │
        ▼
React Clinical Dashboard
```

---

# Evaluation Metrics

The project primarily focuses on:

### AUROC

Measures the model's ability to distinguish between patients with and without a death event across classification thresholds.

### Sensitivity

Measures the proportion of actual death events correctly identified by the model.

```text
Sensitivity = TP / (TP + FN)
```

### Specificity

Measures the proportion of patients without a death event correctly identified.

```text
Specificity = TN / (TN + FP)
```

Supporting metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# Current Development Status

### Completed

- [x] Repository setup
- [x] Dataset loading
- [x] Data quality checks
- [x] Missing-value checks
- [x] Duplicate checks
- [x] Binary-value validation
- [x] Numerical range inspection
- [x] IQR-based outlier inspection
- [x] Target distribution analysis
- [x] Numerical feature EDA
- [x] Feature/target separation
- [x] `time` leakage exclusion
- [x] Stratified 5-Fold Cross-Validation
- [x] StandardScaler
- [x] Leakage-safe scaling
- [x] Logistic Regression baseline

### In Progress / Upcoming

- [ ] Final reusable preprocessing pipeline
- [ ] Complete feature-vs-target EDA
- [ ] PyTorch ANN implementation
- [ ] ANN architecture experimentation
- [ ] ANN regularization
- [ ] Early stopping
- [ ] ANN 5-Fold evaluation
- [ ] Baseline vs ANN comparison
- [ ] Out-of-fold predictions
- [ ] Calibration curve
- [ ] Brier score
- [ ] Model serialization
- [ ] FastAPI prediction service
- [ ] PostgreSQL integration
- [ ] JWT authentication
- [ ] Prediction history
- [ ] React clinical dashboard
- [ ] Docker/docker-compose deployment

---

# Tech Stack

## Machine Learning

- Python
- pandas
- NumPy
- scikit-learn
- PyTorch
- Matplotlib

## Backend

- FastAPI
- PostgreSQL
- SQLAlchemy
- Alembic
- JWT authentication

## Frontend

- React
- Tailwind CSS

## Deployment

- Docker
- docker-compose

---

# Project Structure

The machine learning portion is being developed incrementally.

Current structure:

```text
Heart-Failure-Mortality-Risk-Predictor-ANN/
│
├── data/
│   └── heart_failure_clinical_records_dataset.csv
│
├── check_data.py
├── train.py
├── preprocessing.py
│
├── venv/
│
└── README.md
```

The backend, frontend, model artifacts, and deployment structure will be added as the project progresses.

---

# Baseline Results

The current Logistic Regression baseline was evaluated using Stratified 5-Fold Cross-Validation.

```text
Mean Accuracy     : 0.7560
Mean AUROC        : 0.7717
Mean Sensitivity  : 0.4811
Mean Specificity  : 0.8867
```

These results serve as the reference baseline for the PyTorch ANN.

The ANN will be evaluated using the same validation methodology and metrics to ensure a fair comparison.

---

# Future Application Architecture

The planned production-style architecture is:

```text
React + Tailwind
       │
       ▼
    FastAPI
       │
       ├── Authentication
       ├── Prediction
       ├── History
       │
       ▼
  ML Preprocessing
       │
       ▼
    ANN Model
       │
       ▼
Mortality Probability
       │
       ▼
   PostgreSQL
```

The frontend will present the model's probability output in a probability-first format rather than relying on arbitrary categorical risk labels.

---

# Local Development

## Machine Learning Environment

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install the required ML packages:

```bash
pip install pandas numpy scikit-learn matplotlib torch
```

Run the data checks:

```bash
python check_data.py
```

Run the preprocessing pipeline:

```bash
python preprocessing.py
```

Run the Logistic Regression baseline:

```bash
python train.py
```

---

# Planned Full-Stack Setup

Once the ML pipeline is complete, the full application will use Docker and docker-compose to run:

- React frontend
- FastAPI backend
- PostgreSQL database

The exact Docker configuration will be added when the backend and frontend implementation begins.

---

# Clinical Safety Disclaimer

This system is a **machine-learning-based mortality-risk estimation prototype** intended for research and clinical decision-support purposes.

It is **not a diagnosis** and should not replace professional clinical judgment.

The model should not be used as the sole basis for patient discharge, treatment, or other clinical decisions.

Final clinical decisions must remain with qualified healthcare professionals.