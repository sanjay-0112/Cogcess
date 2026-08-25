import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT = "data/processed/fusion_test_predictions.csv"
OUTPUT = "results"

os.makedirs(OUTPUT, exist_ok=True)

df = pd.read_csv(INPUT)


# ============================================================
# 1. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["actual_grade"],
    df["predicted_grade"],
    alpha=0.25
)

minimum = min(
    df["actual_grade"].min(),
    df["predicted_grade"].min()
)

maximum = max(
    df["actual_grade"].max(),
    df["predicted_grade"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.title("Cogcess Fusion Model: Actual vs Predicted")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT}/actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# 2. ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    df["absolute_error"],
    bins=40
)

plt.xlabel("Absolute Prediction Error")
plt.ylabel("Number of Test Samples")
plt.title("Cogcess Fusion Model: Prediction Error Distribution")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT}/prediction_error_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. METRIC COMPARISON
# ============================================================

metrics = pd.DataFrame({
    "Model": [
        "Baseline Readability",
        "Cogcess Fusion"
    ],
    "MAE": [
        2.7832,
        0.3837
    ]
})

plt.figure(figsize=(8, 6))

plt.bar(
    metrics["Model"],
    metrics["MAE"]
)

plt.ylabel("MAE")
plt.title("Baseline vs Cogcess Fusion")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT}/baseline_vs_fusion_mae.png",
    dpi=300
)

plt.close()


print("=" * 60)
print("RESULT GRAPHS CREATED")
print("=" * 60)

print(f"{OUTPUT}/actual_vs_predicted.png")
print(f"{OUTPUT}/prediction_error_distribution.png")
print(f"{OUTPUT}/baseline_vs_fusion_mae.png")