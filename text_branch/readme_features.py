import pandas as pd
import textstat
import re
from datasets import load_dataset


CEFR_MAPPING = {
    "A1": 0,
    "A2": 1,
    "B1": 2,
    "B2": 3,
    "C1": 4,
    "C2": 5
}


def extract_features(text):

    text = str(text).replace("\n", " ").strip()

    words = re.findall(r"\b[a-zA-Z]+\b", text)

    word_count = len(words)

    sentence_count = max(textstat.sentence_count(text), 1)

    average_word_length = (
        sum(len(word) for word in words) / word_count
        if word_count > 0 else 0
    )

    average_sentence_length = (
        word_count / sentence_count
        if sentence_count > 0 else 0
    )

    return {
        "word_count": word_count,
        "average_word_length": average_word_length,
        "sentence_count": sentence_count,
        "average_sentence_length": average_sentence_length,
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "difficult_words": textstat.difficult_words(text)
    }


def build_readme_features():

    print("Loading ReadMe++...")

    dataset = load_dataset("UniversalCEFR/readme_en")

    data = dataset["train"]

    rows = []

    print(f"Processing {len(data)} examples...")

    for i, example in enumerate(data):

        text = example["text"]
        cefr = example["cefr_level"]

        features = extract_features(text)

        features["cefr_level"] = cefr
        features["cefr_numeric"] = CEFR_MAPPING[cefr]

        rows.append(features)

    df = pd.DataFrame(rows)

    return df


if __name__ == "__main__":

    df = build_readme_features()

    print("\n===== README++ FEATURE DATASET =====")

    print("Shape:", df.shape)

    print("\n===== CEFR DISTRIBUTION =====")

    print(df["cefr_level"].value_counts().sort_index())

    print("\n===== FIRST 5 ROWS =====")

    print(df.head())

    print("\n===== MISSING VALUES =====")

    print(df.isnull().sum())

    output_path = "data/processed/readme_features.csv"

    df.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path}")