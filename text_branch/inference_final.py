import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import textstat
import joblib

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

READABILITY_MODEL_PATH = "models/cogcess_text_readability_final.pth"
CEFR_MODEL_PATH = "models/cogcess_cefr_final.pth"
LEXICAL_MODEL_PATH = "models/cogcess_lexical_v3_clean_model.pth"
FUSION_MODEL_PATH = "models/cogcess_fusion_model.joblib"

MENDELEY_PATH = "data/raw/dataset_english.csv"

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

fusion_checkpoint = joblib.load(
    FUSION_MODEL_PATH
)

fusion_model = fusion_checkpoint["model"]

# ============================================================
# CREATE MODELS
# ============================================================

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

print(
    "Embedding dimension:",
    semantic_encoder.get_embedding_dimension()
)


# ============================================================
# LOAD MENDELEY FREQUENCY DICTIONARY
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

print(
    "Frequency entries:",
    len(mendeley_frequency)
)

MIN_FREQUENCY = float(
    mendeley["fre"].min()
)

print(
    "Minimum observed frequency:",
    MIN_FREQUENCY
)


# ============================================================
# READABILITY FEATURES
# ============================================================

def extract_readability_features(text):

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

    flesch_reading_ease = (
        textstat.flesch_reading_ease(text)
    )

    flesch_kincaid_grade = (
        textstat.flesch_kincaid_grade(text)
    )

    gunning_fog = (
        textstat.gunning_fog(text)
    )

    smog_index = (
        textstat.smog_index(text)
    )

    difficult_words = (
        textstat.difficult_words(text)
    )

    difficult_word_ratio = (
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
            flesch_reading_ease,
            flesch_kincaid_grade,
            gunning_fog,
            smog_index,
            difficult_words,
            difficult_word_ratio,
        ],
        dtype=np.float32
    )


# ============================================================
# POS TAGGING
# ============================================================

def get_word_pos(words):

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

    return nltk.pos_tag(words)


# ============================================================
# LEXICAL WORD FEATURES
# ============================================================

def build_lexical_features(text):

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text.lower()
    )

    if not words:
        return None

    # --------------------------------------------------------
    # REAL FREQUENCY LOOKUP
    # --------------------------------------------------------

    frequencies = []

    known_words = 0
    unknown_words = 0

    for word in words:

        if word in mendeley_frequency:

            frequency = float(
                mendeley_frequency[word]
            )

            known_words += 1

        else:

            frequency = MIN_FREQUENCY

            unknown_words += 1

        frequencies.append(
            frequency
        )

    frequencies = np.asarray(
        frequencies,
        dtype=np.float32
    )

    log_frequencies = np.log(
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

    tagged_words = get_word_pos(words)

    pos_tags = [
        tag
        for _, tag in tagged_words
    ]

    pos_columns = list(
        lexical_checkpoint["pos_columns"]
    )

    pos_vectors = np.zeros(
        (
            len(words),
            len(pos_columns)
        ),
        dtype=np.float32
    )

    pos_index = {
        tag: i
        for i, tag in enumerate(pos_columns)
    }

    for row, tag in enumerate(pos_tags):

        if tag in pos_index:

            pos_vectors[
                row,
                pos_index[tag]
            ] = 1.0


    # --------------------------------------------------------
    # SEMANTIC EMBEDDINGS
    # --------------------------------------------------------

    embeddings = semantic_encoder.encode(
        words,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )


    # --------------------------------------------------------
    # BUILD 417 FEATURES
    # --------------------------------------------------------

    features = np.concatenate(
        [
            log_frequencies.reshape(-1, 1),
            lengths.reshape(-1, 1),
            pos_vectors,
            embeddings
        ],
        axis=1
    )

    return (
        words,
        features,
        known_words,
        unknown_words
    )


# ============================================================
# LEXICAL PREDICTION
# ============================================================

def predict_lexical_difficulty(text):

    result = build_lexical_features(text)

    if result is None:
        return None

    (
        words,
        features,
        known_words,
        unknown_words
    ) = result

    # --------------------------------------------------------
    # SCALE NUMERIC FEATURES
    # --------------------------------------------------------

    scaler_mean = np.asarray(
        lexical_checkpoint["scaler_mean"],
        dtype=np.float32
    )

    scaler_scale = np.asarray(
        lexical_checkpoint["scaler_scale"],
        dtype=np.float32
    )

    # Only the first two features were standardized
    # during V3-clean training:
    #   0 = log_frequency
    #   1 = word length

    features_scaled = features.copy()

    features_scaled[:, :2] = (
        features_scaled[:, :2] - scaler_mean
    ) / scaler_scale


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    with torch.no_grad():

        predictions = lexical_model(
            torch.tensor(
                features_scaled,
                dtype=torch.float32
            )
        ).numpy().flatten()


    predictions = (
        predictions * 100.0
    )

    return {
        "words": words,
        "predictions": predictions,
        "known_words": known_words,
        "unknown_words": unknown_words
    }


# ============================================================
# FULL TEXT ANALYSIS
# ============================================================

def analyze_text(text):

    print("\n" + "=" * 60)
    print("COGCESS TEXT BRANCH FINAL")
    print("=" * 60)

    print("\nText:")
    print(text)


    # ========================================================
    # READABILITY
    # ========================================================

    readability_features = (
        extract_readability_features(text)
    )

    readability_mean = np.asarray(
        readability_checkpoint["feature_mean"],
        dtype=np.float32
    )

    readability_std = np.asarray(
        readability_checkpoint["feature_std"],
        dtype=np.float32
    )

    readability_scaled = (
        readability_features
        - readability_mean
    ) / readability_std

    with torch.no_grad():

        readability_prediction = (
            readability_model(
                torch.tensor(
                    readability_scaled,
                    dtype=torch.float32
                ).unsqueeze(0)
            )
            .item()
        )


    # ========================================================
    # CEFR
    # ========================================================

    cefr_features = np.array(
        [
            readability_features[0],
            readability_features[2],
            readability_features[3],
            readability_features[4],
            readability_features[5],
            readability_features[6],
            readability_features[7],
            readability_features[8],
            readability_features[9],
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
        cefr_features - cefr_mean
    ) / cefr_scale

    with torch.no_grad():

        logits = cefr_model(
            torch.tensor(
                cefr_scaled,
                dtype=torch.float32
            ).unsqueeze(0)
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        ).numpy()[0]

    cefr_mapping = cefr_checkpoint[
        "cefr_mapping"
    ]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_cefr = cefr_mapping[
        predicted_index
    ]


    # ========================================================
    # LEXICAL
    # ========================================================

    lexical = predict_lexical_difficulty(
        text
    )

    if lexical is None:

        lexical_mean = 0
        lexical_median = 0
        lexical_p90 = 0
        lexical_max = 0
        difficult_ratio = 0

    else:

        predictions = lexical[
            "predictions"
        ]

        lexical_mean = float(
            np.mean(predictions)
        )

        lexical_median = float(
            np.median(predictions)
        )

        lexical_p90 = float(
            np.percentile(
                predictions,
                90
            )
        )

        lexical_max = float(
            np.max(predictions)
        )

        difficult_ratio = float(
            np.mean(
                predictions >= 20
            )
        )


    # ========================================================
    # FUSION
    # ========================================================

    fusion_features = np.array(
        [
            readability_prediction,

            readability_features[5],
            readability_features[6],
            readability_features[7],
            readability_features[8],

            probabilities[0],
            probabilities[1],
            probabilities[2],
            probabilities[3],
            probabilities[4],
            probabilities[5],

            lexical_mean,
            lexical_median,
            lexical_p90,
            lexical_max,
            difficult_ratio,
        ],
        dtype=np.float64
    )

    final_grade = float(
        fusion_model.predict(
            fusion_features.reshape(1, -1)
        )[0]
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n===== FINAL COGCESS ASSESSMENT =====")

    print(
        f"Fused predicted grade: "
        f"{final_grade:.2f}"
    )

    print(
        f"Base readability grade: "
        f"{readability_prediction:.2f}"
    )


    print("\n===== READABILITY =====")

    print(
        f"Predicted grade: "
        f"{readability_prediction:.2f}"
    )


    print("\n===== CEFR =====")

    print(
        f"Prediction: "
        f"{predicted_cefr}"
    )

    for i, level in enumerate(
        cefr_mapping
    ):

        print(
            f"{level}: "
            f"{probabilities[i] * 100:.2f}%"
        )


    print("\n===== LEXICAL =====")

    print(
        f"Mean difficulty: "
        f"{lexical_mean:.2f}/100"
    )

    print(
        f"Median difficulty: "
        f"{lexical_median:.2f}/100"
    )

    print(
        f"90th percentile: "
        f"{lexical_p90:.2f}/100"
    )

    print(
        f"Maximum difficulty: "
        f"{lexical_max:.2f}/100"
    )

    print(
        f"Difficult word ratio: "
        f"{difficult_ratio * 100:.2f}%"
    )

    if lexical is not None:

        total_words = (
            lexical["known_words"]
            + lexical["unknown_words"]
        )

        known_ratio = (
            lexical["known_words"]
            / total_words
            if total_words > 0
            else 0
        )

        print(
            f"Known Mendeley words: "
            f"{lexical['known_words']}/{total_words} "
            f"({known_ratio * 100:.1f}%)"
        )

        print(
            f"Unknown words: "
            f"{lexical['unknown_words']}"
        )


    print("\n" + "=" * 60)


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    sample_text = (
        "Artificial intelligence has transformed many "
        "areas of modern computing. Machine learning "
        "systems can identify complex patterns in large "
        "datasets and use these patterns to make "
        "predictions about previously unseen information."
    )

    analyze_text(sample_text)