import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/processed/mendeley_lexical.csv"
SEMANTIC_PATH = "data/processed/mendeley_semantic_features.npz"
MODEL_PATH = "models/cogcess_lexical_v3_model.pth"

RANDOM_STATE = 42
EPOCHS = 150
LEARNING_RATE = 0.001

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Mendeley lexical dataset...")

df = pd.read_csv(DATA_PATH)

df = (
    df.drop_duplicates(
        subset=["word"],
        keep="first"
    )
    .reset_index(drop=True)
)

print("Unique words:", len(df))


# ============================================================
# LOAD SEMANTIC EMBEDDINGS
# ============================================================

print("\nLoading semantic embeddings...")

semantic_data = np.load(
    SEMANTIC_PATH,
    allow_pickle=True
)

embeddings = semantic_data["embeddings"]
embedding_words = semantic_data["words"].astype(str)

dataset_words = df["word"].astype(str).values


# ============================================================
# VERIFY ALIGNMENT
# ============================================================

if len(dataset_words) != len(embedding_words):
    raise ValueError(
        "Dataset and embedding counts do not match."
    )

if not np.array_equal(
    dataset_words,
    embedding_words
):
    raise ValueError(
        "Word order mismatch between dataset and embeddings."
    )

print("Word alignment: OK")


# ============================================================
# CREATE HUMAN DIFFICULTY PROPORTION
# ============================================================

# Mendeley undergraduate difficulty count
#
# Maximum observed response count = 25
#
# Example:
# 0  -> 0.00
# 5  -> 0.20
# 12 -> 0.48
# 25 -> 1.00

df["difficulty_probability"] = (
    df["difficult_ug"] / 25.0
)

df["difficulty_probability"] = (
    df["difficulty_probability"]
    .clip(0, 1)
)


y = df[
    "difficulty_probability"
].values.astype(np.float32)


# ============================================================
# FEATURE 1: NUMERIC
# ============================================================

numeric_features = [
    "log_frequency",
    "len"
]

X_numeric = (
    df[numeric_features]
    .astype(float)
    .values
)


scaler = StandardScaler()

X_numeric = scaler.fit_transform(
    X_numeric
).astype(np.float32)


# ============================================================
# FEATURE 2: POS
# ============================================================

pos_encoded = pd.get_dummies(
    df["pos"],
    prefix="pos",
    dtype=float
)

X_pos = pos_encoded.values.astype(
    np.float32
)


# ============================================================
# FEATURE 3: SEMANTIC
# ============================================================

X_semantic = embeddings.astype(
    np.float32
)


# ============================================================
# COMBINE
# ============================================================

X = np.concatenate(
    [
        X_numeric,
        X_pos,
        X_semantic
    ],
    axis=1
)


print("\n===== FEATURES =====")

print(
    "Numeric features:",
    X_numeric.shape[1]
)

print(
    "POS features:",
    X_pos.shape[1]
)

print(
    "Semantic features:",
    X_semantic.shape[1]
)

print(
    "TOTAL FEATURES:",
    X.shape[1]
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n===== TARGET =====")

print(
    "Minimum:",
    y.min()
)

print(
    "Maximum:",
    y.max()
)

print(
    "Mean:",
    y.mean()
)

print(
    "Median:",
    np.median(y)
)

print(
    "Non-zero difficulty:",
    np.mean(y > 0)
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

indices = np.arange(
    len(df)
)


# Stratify using whether difficulty is zero/non-zero.
#
# This keeps the large easy-word group and
# difficult-word group represented in both sets.

stratify_labels = (
    y > 0
).astype(int)


train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=stratify_labels
)


X_train = X[
    train_idx
]

X_test = X[
    test_idx
]

y_train = y[
    train_idx
]

y_test = y[
    test_idx
]


print("\n===== SPLIT =====")

print(
    "Training samples:",
    len(train_idx)
)

print(
    "Testing samples:",
    len(test_idx)
)


# ============================================================
# TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train
)

X_test_tensor = torch.tensor(
    X_test
)

y_train_tensor = torch.tensor(
    y_train
).unsqueeze(1)

y_test_tensor = torch.tensor(
    y_test
).unsqueeze(1)


# ============================================================
# MODEL
# ============================================================

class CogcessLexicalV3(
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
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                1
            ),

            nn.Sigmoid()
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# CREATE MODEL
# ============================================================

model = CogcessLexicalV3(
    X.shape[1]
)


print("\n===== COGCESS LEXICAL V3 =====")

print(model)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\n===== TRAINING =====")

best_test_loss = float("inf")

best_state = None


for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()

    predictions = model(
        X_train_tensor
    )

    loss = criterion(
        predictions,
        y_train_tensor
    )

    loss.backward()

    optimizer.step()


    # --------------------------------------------------------
    # VALIDATION / TEST MONITORING
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        test_predictions = model(
            X_test_tensor
        )

        test_loss = criterion(
            test_predictions,
            y_test_tensor
        )


    # Save best model
    if test_loss.item() < best_test_loss:

        best_test_loss = test_loss.item()

        best_state = {
            key: value.cpu().clone()
            for key, value
            in model.state_dict().items()
        }


    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {loss.item():.6f} "
            f"Test Loss: {test_loss.item():.6f}"
        )


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# FINAL PREDICTIONS
# ============================================================

model.eval()

with torch.no_grad():

    predictions = (
        model(
            X_test_tensor
        )
        .numpy()
        .flatten()
    )


# Convert from 0–1 to 0–100
predictions_100 = (
    predictions * 100
)

actual_100 = (
    y_test * 100
)


# ============================================================
# FINAL METRICS
# ============================================================

print("\n===== V3 RESULTS =====")

print(
    "MAE:",
    round(
        mean_absolute_error(
            actual_100,
            predictions_100
        ),
        4
    )
)

print(
    "RMSE:",
    round(
        np.sqrt(
            mean_squared_error(
                actual_100,
                predictions_100
            )
        ),
        4
    )
)

print(
    "R²:",
    round(
        r2_score(
            actual_100,
            predictions_100
        ),
        4
    )
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n===== SAMPLE PREDICTIONS =====")

sample_indices = np.random.choice(
    len(test_idx),
    size=min(15, len(test_idx)),
    replace=False
)


for i in sample_indices:

    word = df.iloc[
        test_idx[i]
    ]["word"]

    actual = actual_100[i]

    predicted = predictions_100[i]


    print(
        f"{word:20s} "
        f"Actual: {actual:6.2f}/100 | "
        f"Predicted: {predicted:6.2f}/100"
    )


# ============================================================
# DIFFICULTY BANDS
# ============================================================

def difficulty_band(score):

    if score < 20:
        return "Easy"

    elif score < 40:
        return "Moderate"

    elif score < 60:
        return "Difficult"

    elif score < 80:
        return "Very Difficult"

    else:
        return "Extremely Difficult"


print("\n===== SAMPLE DIFFICULTY BANDS =====")

for i in sample_indices[:10]:

    word = df.iloc[
        test_idx[i]
    ]["word"]

    score = predictions_100[i]

    print(
        f"{word:20s} "
        f"{score:6.2f}/100 "
        f"→ {difficulty_band(score)}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "input_size":
        X.shape[1],

    "pos_columns":
        list(pos_encoded.columns),

    "numeric_features":
        numeric_features,

    "scaler_mean":
        scaler.mean_,

    "scaler_scale":
        scaler.scale_,

    "embedding_model":
        "all-MiniLM-L6-v2",

    "embedding_dimension":
        embeddings.shape[1],

    "target":
        "Undergraduate human difficulty proportion",

    "target_formula":
        "difficult_ug / 25",

    "difficulty_scale":
        "0-100",

    "difficulty_bands": {

        "0-20": "Easy",
        "20-40": "Moderate",
        "40-60": "Difficult",
        "60-80": "Very Difficult",
        "80-100": "Extremely Difficult"

    }

}


torch.save(
    checkpoint,
    MODEL_PATH
)


print(
    f"\nV3 model saved to: {MODEL_PATH}"
)