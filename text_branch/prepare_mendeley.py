import pandas as pd
import numpy as np


INPUT_PATH = "data/raw/dataset_english.csv"
OUTPUT_PATH = "data/processed/mendeley_lexical.csv"


df = pd.read_csv(INPUT_PATH)


# ==========================================
# BASIC CLEANING
# ==========================================

df = df.drop(columns=["Unnamed: 0"])

df["difficult_ug"] = pd.to_numeric(
    df["difficult_ug"],
    errors="coerce"
)

df["difficult_pg"] = pd.to_numeric(
    df["difficult_pg"],
    errors="coerce"
)

df["fre"] = pd.to_numeric(
    df["fre"],
    errors="coerce"
)

df["len"] = pd.to_numeric(
    df["len"],
    errors="coerce"
)


# ==========================================
# DUPLICATE ANALYSIS
# ==========================================

print("===== MENDELEY PREPARATION =====")

print("Original rows:", len(df))

print("Unique words:", df["word"].nunique())

print(
    "Duplicate rows:",
    df["word"].duplicated().sum()
)


# ==========================================
# POS EXTRACTION
# ==========================================

def extract_pos(value):

    value = str(value)

    if "'" in value:
        parts = value.split("'")

        if len(parts) >= 4:
            return parts[3]

    return "UNKNOWN"


df["pos"] = df["ps"].apply(extract_pos)


# ==========================================
# LOG FREQUENCY
# ==========================================

df["log_frequency"] = np.log1p(df["fre"])


# ==========================================
# NORMALIZED UG DIFFICULTY
# ==========================================

MAX_UG_DIFFICULTY = 25

df["lexical_difficulty"] = (
    df["difficult_ug"] / MAX_UG_DIFFICULTY
) * 100


# ==========================================
# REMOVE ROWS WITHOUT UG ANNOTATION
# ==========================================

model_df = df[
    df["difficult_ug"].notna()
].copy()


# ==========================================
# SELECT USEFUL COLUMNS
# ==========================================

model_df = model_df[
    [
        "word",
        "fre",
        "log_frequency",
        "len",
        "pos",
        "difficult_ug",
        "lexical_difficulty"
    ]
]


# ==========================================
# DISPLAY INFORMATION
# ==========================================

print("\nRows available for UG model:", len(model_df))

print("\nLexical difficulty statistics:")

print(
    model_df["lexical_difficulty"].describe()
)

print("\nPOS distribution:")

print(
    model_df["pos"].value_counts().head(20)
)

print("\nFirst 10 rows:")

print(
    model_df.head(10).to_string(index=False)
)


# ==========================================
# SAVE
# ==========================================

model_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved to: {OUTPUT_PATH}"
)