import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import textstat

from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 130951

OUTPUT_PATH = (
    "data/processed/fusion_features_sample.csv"
)

READABILITY_MODEL_PATH = (
    "models/cogcess_text_readability_final.pth"
)

CEFR_MODEL_PATH = (
    "models/cogcess_cefr_final.pth"
)

LEXICAL_MODEL_PATH = (
    "models/cogcess_lexical_v3_clean_model.pth"
)

MENDELEY_PATH = (
    "data/raw/dataset_english.csv"
)

SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"


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

    def __init__(
        self,
        input_size=9,
        num_classes=6
    ):

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
# LOAD CHECKPOINTS
# ============================================================

print("=" * 60)
print("LOADING COGCESS MODELS")
print("=" * 60)

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
# LOAD SEMANTIC MODEL
# ============================================================

print("\nLoading semantic encoder...")

semantic_encoder = SentenceTransformer(
    SEMANTIC_MODEL_NAME
)


# ============================================================
# LOAD MENDELEY FREQUENCIES
# ============================================================

print("\nLoading Mendeley frequency dictionary...")

mendeley = pd.read_csv(
    MENDELEY_PATH
)

mendeley["word"] = (
    mendeley["word"]
    .astype(str)
    .str.lower()
    .str.strip()
)

mendeley_frequency = (
    mendeley
    .drop_duplicates("word")
    .set_index("word")["fre"]
    .to_dict()
)

MIN_FREQUENCY = float(
    mendeley["fre"].min()
)

print(
    "Frequency entries:",
    len(mendeley_frequency)
)


# ============================================================
# READABILITY FEATURES
# ============================================================

def extract_features(text):

    text = (
        str(text)
        .replace("\n", " ")
        .strip()
    )

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text
    )

    word_count = len(words)

    character_count = len(text)

    average_word_length = (
        sum(len(word) for word in words)
        / word_count
        if word_count > 0
        else 0
    )

    sentence_count = max(
        textstat.sentence_count(text),
        1
    )

    average_sentence_length = (
        word_count / sentence_count
    )

    flesch = textstat.flesch_reading_ease(
        text
    )

    fk_grade = textstat.flesch_kincaid_grade(
        text
    )

    fog = textstat.gunning_fog(
        text
    )

    smog = textstat.smog_index(
        text
    )

    difficult_words = textstat.difficult_words(
        text
    )

    difficult_ratio = (
        difficult_words / word_count
        if word_count > 0
        else 0
    )

    return np.array(
        [
            word_count,
            character_count,
            average_word_length,
            sentence_count,
            average_sentence_length,
            flesch,
            fk_grade,
            fog,
            smog,
            difficult_words,
            difficult_ratio,
        ],
        dtype=np.float32
    )


# ============================================================
# POS TAGGING
# ============================================================

def get_pos_tags(words):

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

    tagged = nltk.pos_tag(words)

    return [
        tag
        for _, tag in tagged
    ]


# ============================================================
# LEXICAL TEXT FEATURES
# ============================================================

def lexical_features(text):

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text.lower()
    )

    if not words:

        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "difficult_ratio": 0.0,
        }


    frequencies = []

    for word in words:

        frequency = (
            mendeley_frequency.get(
                word,
                MIN_FREQUENCY
            )
        )

        frequencies.append(
            float(frequency)
        )


    frequencies = np.asarray(
        frequencies,
        dtype=np.float32
    )

    log_frequency = np.log(
        frequencies + 1.0
    )


    lengths = np.asarray(
        [
            len(word)
            for word in words
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # POS
    # --------------------------------------------------------

    pos_tags = get_pos_tags(words)

    pos_columns = list(
        lexical_checkpoint["pos_columns"]
    )

    pos_index = {
        tag: i
        for i, tag in enumerate(pos_columns)
    }

    pos_vectors = np.zeros(
        (
            len(words),
            len(pos_columns)
        ),
        dtype=np.float32
    )

    for row, tag in enumerate(pos_tags):

        if tag in pos_index:

            pos_vectors[
                row,
                pos_index[tag]
            ] = 1.0


    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    embeddings = semantic_encoder.encode(
        words,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )


    # --------------------------------------------------------
    # BUILD FEATURES
    # --------------------------------------------------------

    features = np.concatenate(
        [
            log_frequency.reshape(-1, 1),
            lengths.reshape(-1, 1),
            pos_vectors,
            embeddings,
        ],
        axis=1
    )


    # --------------------------------------------------------
    # SCALE ONLY NUMERIC FEATURES
    # --------------------------------------------------------

    scaler_mean = np.asarray(
        lexical_checkpoint["scaler_mean"],
        dtype=np.float32
    )

    scaler_scale = np.asarray(
        lexical_checkpoint["scaler_scale"],
        dtype=np.float32
    )

    features[:, :2] = (
        features[:, :2] - scaler_mean
    ) / scaler_scale


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    with torch.no_grad():

        predictions = lexical_model(
            torch.tensor(
                features,
                dtype=torch.float32
            )
        ).numpy().flatten()


    predictions *= 100.0


    return {
        "mean": float(
            np.mean(predictions)
        ),

        "median": float(
            np.median(predictions)
        ),

        "p90": float(
            np.percentile(
                predictions,
                90
            )
        ),

        "max": float(
            np.max(predictions)
        ),

        "difficult_ratio": float(
            np.mean(
                predictions >= 20
            )
        ),
    }


# ============================================================
# LOAD AGENTLANS
# ============================================================

print("\nLoading AgentLans...")

dataset = load_dataset(
    "agentlans/readability"
)

train_data = dataset["train"].select(
    range(
        min(
            SAMPLE_SIZE,
            len(dataset["train"])
        )
    )
)

print(
    "Processing:",
    len(train_data),
    "texts"
)


# ============================================================
# FEATURE GENERATION
# ============================================================

rows = []


for i, example in enumerate(
    train_data
):

    text = example["text"]

    readability = extract_features(
        text
    )


    # --------------------------------------------------------
    # READABILITY MODEL
    # --------------------------------------------------------

    read_mean = np.asarray(
        readability_checkpoint[
            "feature_mean"
        ],
        dtype=np.float32
    )

    read_std = np.asarray(
        readability_checkpoint[
            "feature_std"
        ],
        dtype=np.float32
    )

    readability_scaled = (
        readability - read_mean
    ) / read_std


    with torch.no_grad():

        predicted_grade = (
            readability_model(
                torch.tensor(
                    readability_scaled,
                    dtype=torch.float32
                ).unsqueeze(0)
            )
            .item()
        )


    # --------------------------------------------------------
    # CEFR MODEL
    # --------------------------------------------------------

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
        cefr_checkpoint[
            "scaler_mean"
        ],
        dtype=np.float32
    )

    cefr_scale = np.asarray(
        cefr_checkpoint[
            "scaler_scale"
        ],
        dtype=np.float32
    )

    cefr_scaled = (
        cefr_input - cefr_mean
    ) / cefr_scale


    with torch.no_grad():

        cefr_logits = cefr_model(
            torch.tensor(
                cefr_scaled,
                dtype=torch.float32
            ).unsqueeze(0)
        )

        cefr_probabilities = (
            torch.softmax(
                cefr_logits,
                dim=1
            )
            .numpy()[0]
        )


    # --------------------------------------------------------
    # LEXICAL MODEL
    # --------------------------------------------------------

    lexical = lexical_features(
        text
    )


    # --------------------------------------------------------
    # FUSION ROW
    # --------------------------------------------------------

    row = {

        "predicted_grade":
            predicted_grade,

        "flesch_reading_ease":
            readability[5],

        "flesch_kincaid_grade":
            readability[6],

        "gunning_fog":
            readability[7],

        "smog_index":
            readability[8],

        "cefr_a1":
            cefr_probabilities[0],

        "cefr_a2":
            cefr_probabilities[1],

        "cefr_b1":
            cefr_probabilities[2],

        "cefr_b2":
            cefr_probabilities[3],

        "cefr_c1":
            cefr_probabilities[4],

        "cefr_c2":
            cefr_probabilities[5],

        "lexical_mean":
            lexical["mean"],

        "lexical_median":
            lexical["median"],

        "lexical_p90":
            lexical["p90"],

        "lexical_max":
            lexical["max"],

        "lexical_difficult_ratio":
            lexical["difficult_ratio"],

        # SUPERVISED TARGET
        "target_grade":
            example["grade"],
    }


    rows.append(row)


    if (i + 1) % 100 == 0:

        print(
            f"Processed "
            f"{i + 1}/{len(train_data)}"
        )


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(rows)


print("\n" + "=" * 60)
print("FUSION DATASET")
print("=" * 60)

print(
    "Shape:",
    df.shape
)

print(
    "\nMissing values:"
)

print(
    df.isnull().sum()
)

print(
    "\nFeature statistics:"
)

print(
    df.describe()
)


df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved to: {OUTPUT_PATH}"
)