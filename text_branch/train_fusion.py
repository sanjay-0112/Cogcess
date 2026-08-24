import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# SETTINGS
# ============================================================

INPUT_PATH = "data/processed/fusion_features_sample.csv"
OUTPUT_PATH = "models/cogcess_fusion_model.joblib"

TARGET = "target_grade"


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("LOADING FUSION DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "predicted_grade",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
    "smog_index",

    "cefr_a1",
    "cefr_a2",
    "cefr_b1",
    "cefr_b2",
    "cefr_c1",
    "cefr_c2",

    "lexical_mean",
    "lexical_median",
    "lexical_p90",
    "lexical_max",
    "lexical_difficult_ratio",
]

X = df[FEATURE_COLUMNS].values
y = df[TARGET].values


print("\nFeatures:", len(FEATURE_COLUMNS))
print("Samples:", len(X))


# ============================================================
# SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n" + "=" * 60)
print("FUSION SPLIT")
print("=" * 60)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))


# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 60)
print("COGCESS FUSION MODEL")
print("=" * 60)

model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

print(model)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

model.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION
# ============================================================

predictions = model.predict(
    X_val
)

mae = mean_absolute_error(
    y_val,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_val,
        predictions
    )
)

r2 = r2_score(
    y_val,
    predictions
)


print("\n" + "=" * 60)
print("FUSION VALIDATION RESULTS")
print("=" * 60)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# ============================================================
# BASELINE COMPARISON
# ============================================================

baseline = mean_absolute_error(
    y_val,
    X_val[:, 0]
)

print("\n" + "=" * 60)
print("BASELINE COMPARISON")
print("=" * 60)

print(
    f"Readability model MAE: {baseline:.4f}"
)

print(
    f"Fusion model MAE:       {mae:.4f}"
)

if mae < baseline:

    improvement = (
        (baseline - mae)
        / baseline
        * 100
    )

    print(
        f"Improvement: {improvement:.2f}%"
    )

else:

    difference = (
        (mae - baseline)
        / baseline
        * 100
    )

    print(
        f"Fusion is {difference:.2f}% worse"
    )


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for actual, predicted in zip(
    y_val[:20],
    predictions[:20]
):

    print(
        f"Actual: {actual:7.2f} | "
        f"Predicted: {predicted:7.2f}"
    )


# # ============================================================
# # FEATURE IMPORTANCE
# # ============================================================

# print("\n" + "=" * 60)
# print("FEATURE IMPORTANCE")
# print("=" * 60)

# importance = pd.DataFrame({
#     "feature": FEATURE_COLUMNS,
#     "importance": model.feature_importances_
# }).sort_values(
#     "importance",
#     ascending=False
# )

# print(
#     importance.to_string(index=False)
# )


# ============================================================
# SAVE
# ============================================================

joblib.dump(
    {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "baseline_mae": baseline,
        "validation_mae": mae,
        "validation_rmse": rmse,
        "validation_r2": r2,
    },
    OUTPUT_PATH
)

print(
    f"\nFusion model saved to: {OUTPUT_PATH}"
)