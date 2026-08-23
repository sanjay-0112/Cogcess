import pandas as pd


# ==========================================
# LOAD PROCESSED DATA
# ==========================================

agentlans = pd.read_csv(
    "data/processed/agentlans_features_sample.csv"
)

readme = pd.read_csv(
    "data/processed/readme_features.csv"
)


# ==========================================
# AGENTLANS ANALYSIS
# ==========================================

print("===== AGENTLANS =====")

print("Number of samples:", len(agentlans))

print("\nGrade statistics:")
print(agentlans["grade"].describe())


# ==========================================
# README++ ANALYSIS
# ==========================================

print("\n===== README++ =====")

print("Number of samples:", len(readme))

print("\nCEFR numeric statistics:")
print(readme["cefr_numeric"].describe())


# ==========================================
# READABILITY BY CEFR LEVEL
# ==========================================

print("\n===== README++ READABILITY BY CEFR =====")

cefr_analysis = (
    readme
    .groupby("cefr_level")
    [
        [
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "gunning_fog",
            "smog_index"
        ]
    ]
    .mean()
)

print(cefr_analysis)


# ==========================================
# WORD COMPLEXITY BY CEFR
# ==========================================

print("\n===== WORD FEATURES BY CEFR =====")

word_analysis = (
    readme
    .groupby("cefr_level")
    [
        [
            "word_count",
            "average_word_length",
            "average_sentence_length",
            "difficult_words"
        ]
    ]
    .mean()
)

print(word_analysis)