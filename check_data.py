import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")

# Basic dataset checks
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# Check binary columns
binary_cols = [
    "anaemia", "diabetes", "high_blood_pressure",
    "sex", "smoking", "DEATH_EVENT"
]

print("\nBinary values:")
for col in binary_cols:
    print(col, ":", df[col].unique())

# Target distribution
print("\nTarget distribution:")
print(df["DEATH_EVENT"].value_counts())

# Separate features and target
X = df.drop(columns=["DEATH_EVENT", "time"])
y = df["DEATH_EVENT"]

print("\nModel features:", X.columns.tolist())
print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# Outlier check using IQR
numerical_cols = [
    "age", "creatinine_phosphokinase",
    "ejection_fraction", "platelets",
    "serum_creatinine", "serum_sodium"
]

print("\nPotential outliers:")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) |
                  (df[col] > Q3 + 1.5 * IQR)]
    print(col, ":", len(outliers))

# Target distribution plot
df["DEATH_EVENT"].value_counts().sort_index().plot(kind="bar")
plt.title("Target Distribution")
plt.xlabel("DEATH_EVENT")
plt.ylabel("Number of Patients")
plt.xticks([0, 1], ["Survived", "Died"], rotation=0)
plt.tight_layout()
plt.show()

# Numerical feature distributions
df[numerical_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle("Numerical Feature Distributions")
plt.tight_layout()
plt.show()




# Feature vs Target EDA

numerical_cols = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium"
]

for col in numerical_cols:
    df.boxplot(column=col, by="DEATH_EVENT")

    plt.title(f"{col} vs DEATH_EVENT")
    plt.suptitle("")
    plt.xlabel("DEATH_EVENT (0 = No Death, 1 = Death)")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()









