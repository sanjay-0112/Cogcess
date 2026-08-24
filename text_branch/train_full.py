import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from textstat import (
    flesch_reading_ease,
    flesch_kincaid_grade,
    gunning_fog,
    smog_index,
    difficult_words,
    lexicon_count,
    sentence_count,
)


# ============================================================
# SETTINGS
# ============================================================

DATASET_NAME = "agentlans/readability"
MODEL_PATH = "models/cogcess_text_readability_final.pth"

RANDOM_STATE = 42

EPOCHS = 150
LEARNING_RATE = 0.001
PATIENCE = 15

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ============================================================
# LOAD FULL AGENTLANS DATASET
# ============================================================

print("=" * 60)
print("LOADING FULL AGENTLANS DATASET")
print("=" * 60)

dataset = load_dataset(DATASET_NAME)

print(dataset)


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

print("\nConverting dataset...")

train_df = dataset["train"].to_pandas()
val_df = dataset["validation"].to_pandas()
test_df = dataset["test"].to_pandas()

print(
    "Original train:",
    len(train_df)
)

print(
    "Original validation:",
    len(val_df)
)

print(
    "Original test:",
    len(test_df)
)


# ============================================================
# COMBINE FOR FEATURE PROCESSING
# ============================================================

# We keep the original AgentLans split.
#
# This is preferable to randomly reshuffling everything because
# the dataset already provides train/validation/test splits.

all_df = pd.concat(
    [
        train_df,
        val_df,
        test_df
    ],
    ignore_index=True
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

FEATURE_COLUMNS = [
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


def extract_features(text):

    text = str(text)

    words = lexicon_count(
        text,
        removepunct=True
    )

    characters = len(
        text
    )

    sentences = sentence_count(
        text
    )

    if words > 0:

        average_word_length = (
            characters / words
        )

    else:

        average_word_length = 0.0


    if sentences > 0:

        average_sentence_length = (
            words / sentences
        )

    else:

        average_sentence_length = 0.0


    difficult = difficult_words(
        text
    )


    if words > 0:

        difficult_ratio = (
            difficult / words
        )

    else:

        difficult_ratio = 0.0


    return [

        words,

        characters,

        average_word_length,

        sentences,

        average_sentence_length,

        flesch_reading_ease(text),

        flesch_kincaid_grade(text),

        gunning_fog(text),

        smog_index(text),

        difficult,

        difficult_ratio,

    ]


# ============================================================
# PROCESS DATASET
# ============================================================

print("\n" + "=" * 60)
print("EXTRACTING FEATURES")
print("=" * 60)

print(
    "Total texts:",
    len(all_df)
)

features = []

grades = []

sources = []


for i, row in all_df.iterrows():

    text = row["text"]

    try:

        row_features = extract_features(
            text
        )

        features.append(
            row_features
        )

        grades.append(
            float(row["grade"])
        )

        sources.append(
            row["source"]
        )

    except Exception as e:

        print(
            f"Warning: failed at row {i}: {e}"
        )


    if (i + 1) % 5000 == 0:

        print(
            f"Processed {i + 1}/{len(all_df)}"
        )


X = np.array(
    features,
    dtype=np.float32
)

y = np.array(
    grades,
    dtype=np.float32
)


print(
    "\nFeature matrix:",
    X.shape
)

print(
    "Target shape:",
    y.shape
)


# ============================================================
# HANDLE INVALID VALUES
# ============================================================

print("\nChecking numerical values...")

invalid_mask = (
    ~np.isfinite(X).all(axis=1)
    |
    ~np.isfinite(y)
)


invalid_count = int(
    invalid_mask.sum()
)


print(
    "Invalid rows:",
    invalid_count
)


if invalid_count > 0:

    X = X[
        ~invalid_mask
    ]

    y = y[
        ~invalid_mask
    ]


# ============================================================
# RECREATE ORIGINAL SPLIT SIZES
# ============================================================

train_size = len(train_df)
val_size = len(val_df)
test_size = len(test_df)


# Since feature extraction preserves row order,
# use the same dataset boundaries.

X_train = X[
    :train_size
]

y_train = y[
    :train_size
]


X_val = X[
    train_size:
    train_size + val_size
]

y_val = y[
    train_size:
    train_size + val_size
]


X_test = X[
    train_size + val_size:
]

y_test = y[
    train_size + val_size:
]


print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)

print(
    "Test samples:",
    len(X_test)
)


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

# Calculate normalization ONLY from training data.

feature_mean = X_train.mean(
    axis=0
)

feature_std = X_train.std(
    axis=0
)

# Prevent division by zero.

feature_std[
    feature_std == 0
] = 1.0


X_train = (
    X_train - feature_mean
) / feature_std


X_val = (
    X_val - feature_mean
) / feature_std


X_test = (
    X_test - feature_mean
) / feature_std


# ============================================================
# TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).unsqueeze(1)


X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.float32
).unsqueeze(1)


X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).unsqueeze(1)


# ============================================================
# MODEL
# ============================================================

class CogcessTextModel(
    nn.Module
):

    def __init__(
        self,
        input_size
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                1
            )
        )


    def forward(self, x):

        return self.network(x)


model = CogcessTextModel(
    X_train.shape[1]
)


print("\n" + "=" * 60)
print("MODEL")
print("=" * 60)

print(model)


# ============================================================
# TRAINING
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)


best_val_loss = float(
    "inf"
)

best_state = None

epochs_without_improvement = 0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    optimizer.zero_grad()

    train_output = model(
        X_train_tensor
    )

    train_loss = criterion(
        train_output,
        y_train_tensor
    )

    train_loss.backward()

    optimizer.step()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        val_output = model(
            X_val_tensor
        )

        val_loss = criterion(
            val_output,
            y_val_tensor
        )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss.item() < best_val_loss:

        best_val_loss = (
            val_loss.item()
        )

        best_state = {
            key: value.detach().cpu().clone()
            for key, value
            in model.state_dict().items()
        }

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1


    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss.item():.4f} "
            f"Validation Loss: {val_loss.item():.4f}"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            f"Early stopping at epoch "
            f"{epoch + 1}"
        )

        break


# ============================================================
# RESTORE BEST CHECKPOINT
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

model.eval()

with torch.no_grad():

    predictions = (
        model(
            X_test_tensor
        )
        .numpy()
        .flatten()
    )


predictions = np.asarray(
    predictions
)

actual = np.asarray(
    y_test
)


mae = mean_absolute_error(
    actual,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)

r2 = r2_score(
    actual,
    predictions
)


print(
    "MAE:",
    round(mae, 4)
)

print(
    "RMSE:",
    round(rmse, 4)
)

print(
    "R²:",
    round(r2, 4)
)


# ============================================================
# TARGET STATISTICS
# ============================================================

print("\n===== TARGET STATISTICS =====")

print(
    "Mean grade:",
    round(float(y.mean()), 4)
)

print(
    "Minimum grade:",
    round(float(y.min()), 4)
)

print(
    "Maximum grade:",
    round(float(y.max()), 4)
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n===== SAMPLE PREDICTIONS =====")


for i in range(
    min(20, len(actual))
):

    print(
        f"Actual: "
        f"{actual[i]:7.2f} | "
        f"Predicted: "
        f"{predictions[i]:7.2f}"
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "input_size":
        X_train.shape[1],

    "feature_columns":
        FEATURE_COLUMNS,

    "feature_mean":
        feature_mean,

    "feature_std":
        feature_std,

    "dataset":
        DATASET_NAME,

    "training_samples":
        len(X_train),

    "validation_samples":
        len(X_val),

    "test_samples":
        len(X_test),

    "target":
        "AgentLans readability grade",

}


torch.save(
    checkpoint,
    MODEL_PATH
)


print(
    f"\nFinal model saved to: "
    f"{MODEL_PATH}"
)