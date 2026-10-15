
import os
import pandas as pd

# Load the saved 5-fold summary files
ann = pd.read_csv("results/ann_5fold_summary.csv")
logistic = pd.read_csv(
    "results/logistic_regression_5fold_summary.csv"
)

# Keep the metric names and their mean scores
ann_comparison = ann[["Metric", "Mean"]].rename(
    columns={"Mean": "ANN_Mean"}
)

logistic_comparison = logistic[["Metric", "Mean"]].rename(
    columns={"Mean": "Logistic_Regression_Mean"}
)

# Combine both models using the metric name
comparison = pd.merge(
    logistic_comparison,
    ann_comparison,
    on="Metric",
    how="inner"
)

# Save the comparison
os.makedirs("results", exist_ok=True)

output_path = "results/model_comparison.csv"
comparison.to_csv(output_path, index=False)

print("=" * 55)
print("MODEL COMPARISON")
print("=" * 55)
print(comparison.to_string(index=False, float_format="%.4f"))
print("\nSaved to:", output_path)
