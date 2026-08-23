import pandas as pd
import numpy as np


DATA_PATH = "data/raw/dataset_english.csv"

df = pd.read_csv(DATA_PATH)


# ==========================================
# CLEAN DIFFICULTY COLUMNS
# ==========================================

df["difficult_ug"] = pd.to_numeric(
    df["difficult_ug"],
    errors="coerce"
)

df["difficult_pg"] = pd.to_numeric(
    df["difficult_pg"],
    errors="coerce"
)


print("===== MENDELEY DATASET =====")

print("Rows:", len(df))

print("\n===== MISSING / NOT OBSERVED =====")

print(
    "UG not observed:",
    df["difficult_ug"].isna().sum()
)

print(
    "PG not observed:",
    df["difficult_pg"].isna().sum()
)


# ==========================================
# UG ANALYSIS
# ==========================================

print("\n===== UNDERGRADUATE DIFFICULTY =====")

ug_observed = df["difficult_ug"].dropna()

print(ug_observed.describe())

print(
    "\nWords marked difficult:",
    (ug_observed > 0).sum()
)

print(
    "Words not marked difficult:",
    (ug_observed == 0).sum()
)


# ==========================================
# PG ANALYSIS
# ==========================================

print("\n===== POSTGRADUATE DIFFICULTY =====")

pg_observed = df["difficult_pg"].dropna()

print(pg_observed.describe())

print(
    "\nWords marked difficult:",
    (pg_observed > 0).sum()
)

print(
    "Words not marked difficult:",
    (pg_observed == 0).sum()
)


# ==========================================
# FREQUENCY
# ==========================================

print("\n===== WORD FREQUENCY =====")

print(df["fre"].describe())

df["log_frequency"] = np.log1p(df["fre"])

print("\nLog-frequency statistics:")

print(df["log_frequency"].describe())


# ==========================================
# WORD LENGTH
# ==========================================

print("\n===== WORD LENGTH =====")

print(df["len"].describe())


# ==========================================
# EXAMPLES OF DIFFICULT WORDS
# ==========================================

print("\n===== EXAMPLES: UG DIFFICULT WORDS =====")

print(
    df[
        df["difficult_ug"].notna()
        & (df["difficult_ug"] > 0)
    ][
        ["word", "fre", "len", "difficult_ug"]
    ]
    .sort_values(
        "difficult_ug",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)


print("\n===== EXAMPLES: PG DIFFICULT WORDS =====")

print(
    df[
        df["difficult_pg"].notna()
        & (df["difficult_pg"] > 0)
    ][
        ["word", "fre", "len", "difficult_pg"]
    ]
    .sort_values(
        "difficult_pg",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)