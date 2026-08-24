import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import textstat

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

READABILITY_MODEL_PATH = (
    "models/cogcess_text_readability_final.pth"
)

CEFR_MODEL_PATH = (
    "models/cogcess_cefr_final.pth"
)

LEXICAL_MODEL_PATH = (
    "models/cogcess_lexical_v3_clean_model.pth"
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
print("LOADING COGCESS TEXT MODELS")
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


# ============================================================
# READABILITY MODEL
# ============================================================

readability_model = CogcessTextModel(
    input_size=readability_checkpoint["input_size"]
)

readability_model.load_state_dict(
    readability_checkpoint["model_state_dict"]
)

readability_model.eval()


# ============================================================
# CEFR MODEL
# ============================================================

cefr_model = CogcessCEFRModel(
    input_size=cefr_checkpoint["input_size"],
    num_classes=cefr_checkpoint["num_classes"]
)

cefr_model.load_state_dict(
    cefr_checkpoint["model_state_dict"]
)

cefr_model.eval()


# ============================================================
# LEXICAL MODEL
# ============================================================

lexical_model = CogcessLexicalV3(
    input_size=lexical_checkpoint["input_size"]
)

lexical_model.load_state_dict(
    lexical_checkpoint["model_state_dict"]
)

lexical_model.eval()


# ============================================================
# LOAD SEMANTIC ENCODER
# ============================================================

print(
    "\nLoading semantic encoder..."
)

semantic_encoder = SentenceTransformer(
    SEMANTIC_MODEL_NAME
)

print(
    "Embedding dimension:",
    semantic_encoder.get_embedding_dimension()
)


# ============================================================
# READABILITY FEATURES
# ============================================================

READABILITY_FEATURES = [
    "word_count",
    "character_count",
    "average_word_length",
    "sentence_count",
    "average_sentence_length",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
    "smog_index",
    "difficult_words",
    "difficult_word_ratio",
]


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
        sum(
            len(word)
            for word in words
        )
        / word_count
        if word_count > 0
        else 0
    )

    sentence_count = max(
        textstat.sentence_count(text),
        1
    )

    average_sentence_length = (
        word_count
        / sentence_count
    )

    flesch_reading_ease = (
        textstat.flesch_reading_ease(
            text
        )
    )

    flesch_kincaid_grade = (
        textstat.flesch_kincaid_grade(
            text
        )
    )

    gunning_fog = (
        textstat.gunning_fog(
            text
        )
    )

    smog_index = (
        textstat.smog_index(
            text
        )
    )

    difficult_words = (
        textstat.difficult_words(
            text
        )
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
# CEFR FEATURES
# ============================================================

CEFR_FEATURES = [
    "word_count",
    "average_word_length",
    "sentence_count",
    "average_sentence_length",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
    "smog_index",
    "difficult_words",
]


def extract_cefr_features(
    readability_features
):

    # Map the 11 readability features
    # to the exact 9 features used by CEFR.

    return np.array(
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


# ============================================================
# POS EXTRACTION
# ============================================================

def get_word_pos(words):

    try:

        import nltk

        try:
            nltk.data.find(
                "taggers/averaged_perceptron_tagger_eng"
            )
        except LookupError:

            print(
                "Downloading NLTK POS tagger..."
            )

            nltk.download(
                "averaged_perceptron_tagger_eng",
                quiet=True
            )

        return nltk.pos_tag(
            words
        )

    except Exception as e:

        print(
            "POS tagging unavailable:",
            e
        )

        return [
            (word, "NN")
            for word in words
        ]


# ============================================================
# LEXICAL FEATURES
# ============================================================

def extract_lexical_features(
    text
):

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text.lower()
    )

    if not words:

        return None


    # --------------------------------------------------------
    # NUMERIC FEATURES
    # --------------------------------------------------------

    word_frequencies = []

    for word in words:

        # Frequency isn't directly available from Mendeley
        # for arbitrary new words.
        #
        # For inference, use a conservative frequency proxy.
        word_frequencies.append(1.0)


    frequencies = np.array(
        word_frequencies,
        dtype=np.float32
    )


    log_frequency = np.log(
        frequencies + 1
    )

    lengths = np.array(
        [
            len(word)
            for word in words
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # POS
    # --------------------------------------------------------

    tagged_words = get_word_pos(
        words
    )

    pos_tags = [
        tag
        for _, tag
        in tagged_words
    ]


    pos_columns = (
        lexical_checkpoint["pos_columns"]
    )


    pos_vector = np.zeros(
        len(pos_columns),
        dtype=np.float32
    )


    for tag in pos_tags:

        if tag in pos_columns:

            index = pos_columns.index(
                tag
            )

            pos_vector[index] += 1


    # Convert counts to proportions.

    pos_vector /= max(
        len(words),
        1
    )


    # --------------------------------------------------------
    # SEMANTIC EMBEDDINGS
    # --------------------------------------------------------

    embeddings = (
        semantic_encoder.encode(
            words,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    )


    word_features = []


    numeric_mean = np.array(
        [
            np.mean(log_frequency),
            np.mean(lengths)
        ],
        dtype=np.float32
    )


    for i in range(
        len(words)
    ):

        feature = np.concatenate(
            [
                np.array(
                    [
                        log_frequency[i],
                        lengths[i]
                    ],
                    dtype=np.float32
                ),

                pos_vector,

                embeddings[i]
            ]
        )

        word_features.append(
            feature
        )


    return (
        words,
        np.array(
            word_features,
            dtype=np.float32
        )
    )


# ============================================================
# TEXT ANALYSIS
# ============================================================

def analyze_text(text):

    print(
        "\n" + "=" * 60
    )

    print(
        "COGCESS TEXT BRANCH"
    )

    print(
        "=" * 60
    )

    print(
        "\nText:"
    )

    print(
        text
    )


    # ========================================================
    # READABILITY
    # ========================================================

    readability_features = (
        extract_readability_features(
            text
        )
    )


    # IMPORTANT:
    # Normalize using the training statistics saved
    # in the checkpoint.

    readability_scaled = (
        readability_features
        - readability_checkpoint[
            "feature_mean"
        ]
    ) / (
        readability_checkpoint[
            "feature_std"
        ]
    )


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

    cefr_features = (
        extract_cefr_features(
            readability_features
        )
    )


    cefr_scaled = (
        cefr_features
        - cefr_checkpoint[
            "scaler_mean"
        ]
    ) / (
        cefr_checkpoint[
            "scaler_scale"
        ]
    )


    with torch.no_grad():

        cefr_logits = cefr_model(
            torch.tensor(
                cefr_scaled,
                dtype=torch.float32
            ).unsqueeze(0)
        )

        cefr_probabilities = torch.softmax(
            cefr_logits,
            dim=1
        ).numpy()[0]


    cefr_mapping = (
        cefr_checkpoint[
            "cefr_mapping"
        ]
    )


    predicted_cefr_index = int(
        np.argmax(
            cefr_probabilities
        )
    )

    predicted_cefr = (
        cefr_mapping[
            predicted_cefr_index
        ]
    )


    # ========================================================
    # LEXICAL
    # ========================================================

    lexical_result = (
        extract_lexical_features(
            text
        )
    )


    lexical_predictions = []

    lexical_words = []


    if lexical_result is not None:

        lexical_words, word_features = (
            lexical_result
        )


        lexical_scaled = (
            word_features[:, :2]
            - np.array(
                lexical_checkpoint[
                    "scaler_mean"
                ][:2]
            )
        ) / (
            np.array(
                lexical_checkpoint[
                    "scaler_scale"
                ][:2]
            )
        )


        # Recombine standardized numeric features
        # with POS and semantic features.

        lexical_features_scaled = np.concatenate(
            [
                lexical_scaled,

                word_features[:, 2:]
            ],
            axis=1
        )


        with torch.no_grad():

            lexical_output = (
                lexical_model(
                    torch.tensor(
                        lexical_features_scaled,
                        dtype=torch.float32
                    )
                )
                .numpy()
                .flatten()
            )


        lexical_predictions = (
            lexical_output * 100
        )


    # ========================================================
    # LEXICAL AGGREGATION
    # ========================================================

    if len(lexical_predictions) > 0:

        lexical_mean = float(
            np.mean(
                lexical_predictions
            )
        )

        lexical_max = float(
            np.max(
                lexical_predictions
            )
        )

        lexical_median = float(
            np.median(
                lexical_predictions
            )
        )

        lexical_p90 = float(
            np.percentile(
                lexical_predictions,
                90
            )
        )

        lexical_difficult_ratio = float(
            np.mean(
                lexical_predictions >= 20
            )
        )

    else:

        lexical_mean = 0.0
        lexical_max = 0.0
        lexical_median = 0.0
        lexical_p90 = 0.0
        lexical_difficult_ratio = 0.0


    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        "\n===== READABILITY ====="
    )

    print(
        f"Predicted grade: "
        f"{readability_prediction:.2f}"
    )


    print(
        "\n===== CEFR ====="
    )

    print(
        f"Prediction: "
        f"{predicted_cefr}"
    )


    for index in range(6):

        level = cefr_mapping[
            index
        ]

        probability = (
            cefr_probabilities[index]
            * 100
        )

        print(
            f"{level}: "
            f"{probability:.2f}%"
        )


    print(
        "\n===== LEXICAL ====="
    )

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
        f"{lexical_difficult_ratio * 100:.2f}%"
    )


    print(
        "\n" + "=" * 60
    )


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

    analyze_text(
        sample_text
    )