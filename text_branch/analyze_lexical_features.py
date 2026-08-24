import pandas as pd
import numpy as np

DATA_PATH = "data/processed/mendeley_lexical.csv"

print("Loading Mendeley lexical dataset...")
df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------
# BASIC INFO
# ------------------------------------------------------------

print("\n===== DATASET =====")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# ------------------------------------------------------------
# NUMERIC FEATURE CORRELATIONS
# ------------------------------------------------------------

numeric_features = [
    "fre",
    "log_frequency",
    "len",
    "difficult_ug",
    "lexical_difficulty"
]

print("\n===== NUMERIC CORRELATIONS =====")

correlation = df[numeric_features].corr()

print(
    correlation["lexical_difficulty"]
    .sort_values(ascending=False)
    .to_string()
)

# ------------------------------------------------------------
# CORRELATION WITH HUMAN DIFFICULTY
# ------------------------------------------------------------

print("\n===== CORRELATION WITH LEXICAL DIFFICULTY =====")

for feature in [
    "fre",
    "log_frequency",
    "len"
]:

    corr = df[feature].corr(
        df["lexical_difficulty"]
    )

    print(
        f"{feature:20s}: {corr:.4f}"
    )

# ------------------------------------------------------------
# DIFFICULTY BY WORD LENGTH
# ------------------------------------------------------------

print("\n===== DIFFICULTY BY WORD LENGTH =====")

length_stats = (
    df.groupby("len")["lexical_difficulty"]
    .agg(["count", "mean", "median"])
)

print(length_stats.to_string())

# ------------------------------------------------------------
# DIFFICULTY BY POS
# ------------------------------------------------------------

print("\n===== DIFFICULTY BY POS =====")

pos_stats = (
    df.groupby("pos")["lexical_difficulty"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)

print(pos_stats.to_string())

# ------------------------------------------------------------
# DIFFICULTY BY FREQUENCY
# ------------------------------------------------------------

print("\n===== DIFFICULTY BY FREQUENCY =====")

df["frequency_group"] = pd.cut(
    df["fre"],
    bins=[
        0,
        1,
        2,
        5,
        10,
        50,
        100,
        np.inf
    ],
    labels=[
        "1",
        "2",
        "3-5",
        "6-10",
        "11-50",
        "51-100",
        "100+"
    ]
)

frequency_stats = (
    df.groupby(
        "frequency_group",
        observed=True
    )["lexical_difficulty"]
    .agg(["count", "mean", "median"])
)

print(
    frequency_stats.to_string()
)

# ------------------------------------------------------------
# DIFFICULTY DISTRIBUTION
# ------------------------------------------------------------

print("\n===== DIFFICULTY DISTRIBUTION =====")

print(
    df["lexical_difficulty"]
    .describe()
    .to_string()
)

# ------------------------------------------------------------
# EASIEST / HARDEST WORDS
# ------------------------------------------------------------

print("\n===== HARDEST WORDS =====")

hardest = (
    df.sort_values(
        "lexical_difficulty",
        ascending=False
    )
    [["word", "fre", "len", "pos", "difficult_ug", "lexical_difficulty"]]
    .head(20)
)

print(
    hardest.to_string(index=False)
)

print("\n===== EASIEST NON-ZERO WORDS =====")

easiest_nonzero = (
    df[df["lexical_difficulty"] > 0]
    .sort_values(
        "lexical_difficulty"
    )
    [["word", "fre", "len", "pos", "difficult_ug", "lexical_difficulty"]]
    .head(20)
)

print(
    easiest_nonzero.to_string(index=False)
)

# ------------------------------------------------------------
# DIFFICULT WORD RATE
# ------------------------------------------------------------

print("\n===== DIFFICULT WORD RATE =====")

difficult_rate = (
    df["difficult_ug"] > 0
).mean()

print(
    f"Words marked difficult: "
    f"{difficult_rate * 100:.2f}%"
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n========================================")
print("LEXICAL FEATURE ANALYSIS COMPLETE")
print("========================================")