from datasets import load_dataset
import pandas as pd

print("Loading AgentLans...")

dataset = load_dataset("agentlans/readability")

train = dataset["train"].to_pandas()

print("\n===== SHAPE =====")
print(train.shape)

print("\n===== COLUMNS =====")
print(train.columns.tolist())

print("\n===== GRADE STATISTICS =====")
print(train["grade"].describe())

print("\n===== SOURCES =====")
print(train["source"].value_counts())

print("\n===== SAMPLE =====")
print(train[["text", "grade", "source"]].head(10).to_string())

print("\n\nLoading ReadMe++...")

readme = load_dataset("UniversalCEFR/readme_en")

readme_train = readme["train"].to_pandas()

print("\n===== README++ SHAPE =====")
print(readme_train.shape)

print("\n===== COLUMNS =====")
print(readme_train.columns.tolist())

print("\n===== CEFR DISTRIBUTION =====")
print(readme_train["cefr_level"].value_counts().sort_index())

print("\n===== FORMAT DISTRIBUTION =====")
print(readme_train["format"].value_counts())

print("\n===== CATEGORY DISTRIBUTION =====")
print(readme_train["category"].value_counts())

print("\n===== SAMPLE =====")
print(
    readme_train[
        ["text", "cefr_level", "format", "category"]
    ].head(10).to_string()
)