
import os
import sys
import pandas as pd

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ============================================================
# LOAD 5-FOLD SUMMARY FILES
# ============================================================

ann_path = os.path.join(
    RESULTS_DIR,
    "ann_5fold_summary.csv"
)

logistic_path = os.path.join(
    RESULTS_DIR,
    "logistic_regression_5fold_summary.csv"
)

ann = pd.read_csv(ann_path)
logistic = pd.read_csv(logistic_path)

# Keep metric names and mean scores
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

# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)

output_path = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)

comparison.to_csv(output_path, index=False)

print("=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

print(
    comparison.to_string(
        index=False,
        float_format="%.4f"
    )
)

print("\nSaved to:", output_path)
