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


print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nTesting preprocessing pipeline...")


for fold, (train_index, val_index) in enumerate(
    skf.split(X, y), start=1
):

    X_train = X.iloc[train_index]
    X_val = X.iloc[val_index]

    # Fit preprocessing only on training data
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)

    # Transform validation data
    X_val_processed = preprocessing_pipeline.transform(X_val)

    print(f"\nFold {fold}")
    print("Train shape:", X_train_processed.shape)
    print("Validation shape:", X_val_processed.shape)


print("\nPreprocessing pipeline completed successfully.")