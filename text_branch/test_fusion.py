import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import textstat
import joblib

from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

FUSION_MODEL_PATH = "models/cogcess_fusion_model.joblib"

READABILITY_MODEL_PATH = (
    "models/cogcess_text_readability_final.pth"
)

CEFR_MODEL_PATH = (
    "models/cogcess_cefr_final.pth"
)

LEXICAL_MODEL_PATH = (
    "models/cogcess_lexical_v3_clean_model.pth"
)

MENDELEY_PATH = "data/raw/dataset_english.csv"


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class CogcessTextModel(nn.Module):

    def __init__(self, input_size=11):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


class CogcessCEFRModel(nn.Module):

    def __init__(self, input_size=9, num_classes=6):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class CogcessLexicalV3(nn.Module):

    def __init__(self, input_size=417):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 60)
print("LOADING COGCESS MODELS")
print("=" * 60)

fusion = joblib.load(
    FUSION_MODEL_PATH
)

fusion_model = fusion["model"]
feature_columns = fusion["feature_columns"]

readability_checkpoint = torch.load(
    READABILITY_MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

cefr_checkpoint = torch.load(
    CEFR_MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

lexical_checkpoint = torch.load(
    LEXICAL_MODEL_PATH,
    map_location="cpu",
    weights_only=False
)


readability_model = CogcessTextModel(
    readability_checkpoint["input_size"]
)

readability_model.load_state_dict(
    readability_checkpoint["model_state_dict"]
)

readability_model.eval()


cefr_model = CogcessCEFRModel(
    cefr_checkpoint["input_size"],
    cefr_checkpoint["num_classes"]
)

cefr_model.load_state_dict(
    cefr_checkpoint["model_state_dict"]
)

cefr_model.eval()


lexical_model = CogcessLexicalV3(
    lexical_checkpoint["input_size"]
)

lexical_model.load_state_dict(
    lexical_checkpoint["model_state_dict"]
)

lexical_model.eval()


# ============================================================
# SEMANTIC MODEL
# ============================================================

print("\nLoading semantic encoder...")

semantic_encoder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# MENDELEY
# ============================================================

print("Loading Mendeley...")

mendeley = pd.read_csv(
    MENDELEY_PATH
)

mendeley["word"] = (
    mendeley["word"]
    .astype(str)
    .str.lower()
    .str.strip()
)

frequency = (
    mendeley
    .drop_duplicates("word")
    .set_index("word")["fre"]
    .to_dict()
)

MIN_FREQUENCY = float(
    mendeley["fre"].min()
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(text):

    text = str(text).replace(
        "\n", " "
    ).strip()

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text
    )

    word_count = len(words)

    character_count = len(text)

    average_word_length = (
        sum(len(w) for w in words)
        / word_count
        if word_count
        else 0
    )

    sentence_count = max(
        textstat.sentence_count(text),
        1
    )

    average_sentence_length = (
        word_count / sentence_count
    )

    return np.array(
        [
            word_count,
            character_count,
            average_word_length,
            sentence_count,
            average_sentence_length,
            textstat.flesch_reading_ease(text),
            textstat.flesch_kincaid_grade(text),
            textstat.gunning_fog(text),
            textstat.smog_index(text),
            textstat.difficult_words(text),
            (
                textstat.difficult_words(text)
                / word_count
                if word_count
                else 0
            ),
        ],
        dtype=np.float32
    )


# ============================================================
# LEXICAL
# ============================================================

def lexical_prediction(text):

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text.lower()
    )

    if not words:
        return 0, 0, 0, 0, 0

    frequencies = np.array(
        [
            frequency.get(
                word,
                MIN_FREQUENCY
            )
            for word in words
        ],
        dtype=np.float32
    )

    log_frequency = np.log(
        frequencies + 1
    )

    lengths = np.array(
        [len(w) for w in words],
        dtype=np.float32
    )

    import nltk

    try:
        nltk.data.find(
            "taggers/averaged_perceptron_tagger_eng"
        )
    except LookupError:
        nltk.download(
            "averaged_perceptron_tagger_eng",
            quiet=True
        )

    tags = [
        tag
        for _, tag in nltk.pos_tag(words)
    ]

    pos_columns = list(
        lexical_checkpoint["pos_columns"]
    )

    pos_index = {
        tag: i
        for i, tag in enumerate(pos_columns)
    }

    pos = np.zeros(
        (len(words), len(pos_columns)),
        dtype=np.float32
    )

    for i, tag in enumerate(tags):

        if tag in pos_index:
            pos[i, pos_index[tag]] = 1


    embeddings = semantic_encoder.encode(
        words,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    features = np.concatenate(
        [
            log_frequency[:, None],
            lengths[:, None],
            pos,
            embeddings
        ],
        axis=1
    )

    mean = np.asarray(
        lexical_checkpoint["scaler_mean"],
        dtype=np.float32
    )

    scale = np.asarray(
        lexical_checkpoint["scaler_scale"],
        dtype=np.float32
    )

    features[:, :2] = (
        features[:, :2] - mean
    ) / scale

    with torch.no_grad():

        predictions = lexical_model(
            torch.tensor(
                features,
                dtype=torch.float32
            )
        ).numpy().flatten() * 100

    return (
        float(np.mean(predictions)),
        float(np.median(predictions)),
        float(np.percentile(predictions, 90)),
        float(np.max(predictions)),
        float(np.mean(predictions >= 20))
    )


# ============================================================
# BUILD FUSION FEATURES
# ============================================================

def build_fusion_features(text):

    readability = extract_features(text)

    read_mean = np.asarray(
        readability_checkpoint["feature_mean"],
        dtype=np.float32
    )

    read_std = np.asarray(
        readability_checkpoint["feature_std"],
        dtype=np.float32
    )

    read_scaled = (
        readability - read_mean
    ) / read_std

    with torch.no_grad():

        predicted_grade = readability_model(
            torch.tensor(
                read_scaled,
                dtype=torch.float32
            ).unsqueeze(0)
        ).item()


    cefr_input = np.array(
        [
            readability[0],
            readability[2],
            readability[3],
            readability[4],
            readability[5],
            readability[6],
            readability[7],
            readability[8],
            readability[9],
        ],
        dtype=np.float32
    )

    cefr_mean = np.asarray(
        cefr_checkpoint["scaler_mean"],
        dtype=np.float32
    )

    cefr_scale = np.asarray(
        cefr_checkpoint["scaler_scale"],
        dtype=np.float32
    )

    cefr_scaled = (
        cefr_input - cefr_mean
    ) / cefr_scale

    with torch.no_grad():

        logits = cefr_model(
            torch.tensor(
                cefr_scaled,
                dtype=torch.float32
            ).unsqueeze(0)
        )

        cefr_probs = torch.softmax(
            logits,
            dim=1
        ).numpy()[0]


    lexical = lexical_prediction(
        text
    )

    row = np.array(
        [
            predicted_grade,
            readability[5],
            readability[6],
            readability[7],
            readability[8],

            cefr_probs[0],
            cefr_probs[1],
            cefr_probs[2],
            cefr_probs[3],
            cefr_probs[4],
            cefr_probs[5],

            lexical[0],
            lexical[1],
            lexical[2],
            lexical[3],
            lexical[4],
        ],
        dtype=np.float64
    )

    return row


# ============================================================
# LOAD UNTAPPED TEST SET
# ============================================================

print("\n" + "=" * 60)
print("LOADING AGENTLANS TEST SET")
print("=" * 60)

dataset = load_dataset(
    "agentlans/readability"
)

test_data = dataset["test"]

print(
    "Test samples:",
    len(test_data)
)


# ============================================================
# FINAL EVALUATION
# ============================================================

predictions = []
actuals = []

print("\nGenerating final test predictions...")

for i, example in enumerate(test_data):

    features = build_fusion_features(
        example["text"]
    )

    prediction = fusion_model.predict(
        features.reshape(1, -1)
    )[0]

    predictions.append(
        prediction
    )

    actuals.append(
        example["grade"]
    )

    if (i + 1) % 500 == 0:

        print(
            f"Processed "
            f"{i + 1}/{len(test_data)}"
        )


predictions = np.asarray(
    predictions
)

actuals = np.asarray(
    actuals
)


# ============================================================
# METRICS
# ============================================================

mae = np.mean(
    np.abs(
        predictions - actuals
    )
)

rmse = np.sqrt(
    np.mean(
        (predictions - actuals) ** 2
    )
)

ss_res = np.sum(
    (actuals - predictions) ** 2
)

ss_tot = np.sum(
    (actuals - np.mean(actuals)) ** 2
)

r2 = 1 - (
    ss_res / ss_tot
)


print("\n" + "=" * 60)
print("FINAL COGCESS FUSION RESULTS")
print("=" * 60)

print(
    f"MAE:  {mae:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"R²:   {r2:.4f}"
)


# ============================================================
# SAMPLE RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for actual, prediction in zip(
    actuals[:20],
    predictions[:20]
):

    print(
        f"Actual: {actual:7.2f} | "
        f"Predicted: {prediction:7.2f}"
    )