import re
import pandas as pd
import textstat
from datasets import load_dataset


def extract_features(text):
    """
    Convert one piece of text into numerical readability
    and linguistic features for the Cogcess Text Branch.
    """

    text = str(text).replace("\n", " ").strip()

    if not text:
        return None

    words = re.findall(r"\b[a-zA-Z]+\b", text)

    word_count = len(words)
    character_count = len(text)

    average_word_length = (
        sum(len(word) for word in words) / word_count
        if word_count > 0 else 0
    )

    sentence_count = max(textstat.sentence_count(text), 1)

    average_sentence_length = word_count / sentence_count

    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    gunning_fog = textstat.gunning_fog(text)
    smog_index = textstat.smog_index(text)

    difficult_words = textstat.difficult_words(text)

    difficult_word_ratio = (
        difficult_words / word_count
        if word_count > 0 else 0
    )

    return {
        "word_count": word_count,
        "character_count": character_count,
        "average_word_length": average_word_length,
        "sentence_count": sentence_count,
        "average_sentence_length": average_sentence_length,
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "gunning_fog": gunning_fog,
        "smog_index": smog_index,
        "difficult_words": difficult_words,
        "difficult_word_ratio": difficult_word_ratio,
    }


def build_agentlans_features(sample_size=1000):
    """
    Load AgentLans and convert a sample into a feature dataset.
    """

    print("Loading AgentLans dataset...")

    dataset = load_dataset("agentlans/readability")

    train_data = dataset["train"].select(
        range(min(sample_size, len(dataset["train"])))
    )

    rows = []

    print(f"Processing {len(train_data)} texts...")

    for i, example in enumerate(train_data):

        features = extract_features(example["text"])

        if features is not None:

            features["grade"] = example["grade"]
            features["source"] = example["source"]

            rows.append(features)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(train_data)}")

    df = pd.DataFrame(rows)

    return df


if __name__ == "__main__":

    df = build_agentlans_features(sample_size=1000)

    print("\n===== DATASET CREATED =====")
    print("Shape:", df.shape)

    print("\n===== COLUMNS =====")
    print(df.columns.tolist())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== FEATURE STATISTICS =====")
    print(df.describe())

    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    # Save sample
    output_path = "data/processed/agentlans_features_sample.csv"

    df.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path}")