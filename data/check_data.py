
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "heart_failure_clinical_records_dataset.csv"
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ============================================================
# BINARY VALUE CHECKS
# ============================================================

binary_cols = [
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
    "DEATH_EVENT"
]

print("\nBinary values:")

for col in binary_cols:
    print(f"{col}: {df[col].unique()}")

# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")
print(df["DEATH_EVENT"].value_counts().sort_index())

# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["DEATH_EVENT", "time"])
y = df["DEATH_EVENT"]

print("\nModel features:", X.columns.tolist())
print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# ============================================================
# OUTLIER INSPECTION USING IQR
# ============================================================

numerical_cols = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium"
]

print("\nPotential outliers using IQR:")

for col in numerical_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_count = (
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ).sum()

    print(f"{col}: {outlier_count}")

# ============================================================
# TARGET DISTRIBUTION PLOT
# ============================================================

target_counts = df["DEATH_EVENT"].value_counts().sort_index()

ax = target_counts.plot(kind="bar")

ax.set_title("Target Distribution")
ax.set_xlabel("DEATH_EVENT")
ax.set_ylabel("Number of Patients")
ax.set_xticks([0, 1], ["Survived", "Died"], rotation=0)

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "target_distribution.png"),
    dpi=150
)
plt.close()

# ============================================================
# NUMERICAL FEATURE DISTRIBUTIONS
# ============================================================

df[numerical_cols].hist(figsize=(12, 8), bins=20)

plt.suptitle("Numerical Feature Distributions")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "numerical_distributions.png"),
    dpi=150
)
plt.close()

# ============================================================
# FEATURE VS TARGET ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE VS TARGET ANALYSIS")
print("=" * 60)

for col in numerical_cols:
    print(f"\n{col}")

    summary = df.groupby("DEATH_EVENT")[col].agg(
        ["mean", "median", "min", "max"]
    )

    print(summary)

    df.boxplot(
        column=col,
        by="DEATH_EVENT"
    )

    plt.title(f"{col} vs DEATH_EVENT")
    plt.suptitle("")
    plt.xlabel("DEATH_EVENT (0 = No Death, 1 = Death)")
    plt.ylabel(col)
    plt.tight_layout()

    safe_name = col.replace("/", "_")

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{safe_name}_vs_death_event.png"
        ),
        dpi=150
    )
    plt.close()

# ============================================================
# BINARY FEATURES VS TARGET
# ============================================================

binary_feature_cols = [
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking"
]

print("\nBinary Features — Mean by Target Class:")

for col in binary_feature_cols:
    summary = df.groupby("DEATH_EVENT")[col].mean()

    print(f"\n{col}")
    print(summary)

# ============================================================
# FEATURE CORRELATION WITH TARGET
# ============================================================

print("\n" + "=" * 60)
print("FEATURE CORRELATION WITH TARGET")
print("=" * 60)

correlation = (
    df.drop(columns=["time"])
    .corr()["DEATH_EVENT"]
    .sort_values(ascending=False)
)

print("\nCorrelation of features with DEATH_EVENT:")
print(correlation)

correlation.to_csv(
    os.path.join(RESULTS_DIR, "feature_target_correlations.csv"),
    header=["Correlation"]
)

print("\nEDA completed successfully.")
print("Plots and correlation results saved to:")
print(RESULTS_DIR)


