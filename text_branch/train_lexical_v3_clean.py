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
MODEL_PATH = "models/cogcess_lexical_v3_clean_model.pth"

RANDOM_STATE = 42
EPOCHS = 150
LEARNING_RATE = 0.001
PATIENCE = 20

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

embeddings = semantic_data["embeddings"].astype(
    np.float32
)

embedding_words = (
    semantic_data["words"]
    .astype(str)
)

dataset_words = (
    df["word"]
    .astype(str)
    .values
)


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
# TARGET
# ============================================================

# Human undergraduate difficulty responses.
#
# difficult_ug represents the number of students
# who marked the word as difficult.
#
# 25 students -> maximum observed response count.
#
# Normalize to 0-1 for training.

df["difficulty_probability"] = (
    df["difficult_ug"] / 25.0
)

df["difficulty_probability"] = (
    df["difficulty_probability"]
    .clip(0, 1)
)

y = (
    df["difficulty_probability"]
    .values
    .astype(np.float32)
)


# ============================================================
# FEATURES
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


# Standardize numeric features
scaler = StandardScaler()

X_numeric = scaler.fit_transform(
    X_numeric
).astype(np.float32)


# POS one-hot encoding
pos_encoded = pd.get_dummies(
    df["pos"],
    prefix="pos",
    dtype=float
)

X_pos = (
    pos_encoded
    .values
    .astype(np.float32)
)


# Semantic embeddings
X_semantic = embeddings


# Combine
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
    round(float(y.mean()), 4)
)

print(
    "Median:",
    np.median(y)
)

print(
    "Non-zero difficulty:",
    f"{np.mean(y > 0) * 100:.2f}%"
)


# ============================================================
# PROPER 70 / 15 / 15 SPLIT
# ============================================================

indices = np.arange(
    len(df)
)

# 15% final test set.
#
# This set will NOT be used during training
# or early stopping.

train_val_idx, test_idx = train_test_split(
    indices,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=(y > 0).astype(int)
)


# Split remaining 85% into:
#
# 70% total training
# 15% validation

train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=(15 / 85),
    random_state=RANDOM_STATE,
    stratify=(y[train_val_idx] > 0).astype(int)
)


X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]


print("\n===== SPLIT =====")

print(
    "Training samples:",
    len(train_idx)
)

print(
    "Validation samples:",
    len(val_idx)
)

print(
    "Final test samples:",
    len(test_idx)
)


# ============================================================
# TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train
)

X_val_tensor = torch.tensor(
    X_val
)

X_test_tensor = torch.tensor(
    X_test
)

y_train_tensor = torch.tensor(
    y_train
).unsqueeze(1)

y_val_tensor = torch.tensor(
    y_val
).unsqueeze(1)

y_test_tensor = torch.tensor(
    y_test
).unsqueeze(1)


# ============================================================
# V3 MODEL
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
# MODEL
# ============================================================

model = CogcessLexicalV3(
    X.shape[1]
)


print("\n===== COGCESS LEXICAL V3 CLEAN =====")

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

best_val_loss = float("inf")

best_state = None

epochs_without_improvement = 0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    optimizer.zero_grad()

    train_predictions = model(
        X_train_tensor
    )

    train_loss = criterion(
        train_predictions,
        y_train_tensor
    )

    train_loss.backward()

    optimizer.step()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        val_predictions = model(
            X_val_tensor
        )

        val_loss = criterion(
            val_predictions,
            y_val_tensor
        )


    # --------------------------------------------------------
    # SAVE BEST VALIDATION MODEL
    # --------------------------------------------------------

    if val_loss.item() < best_val_loss:

        best_val_loss = val_loss.item()

        best_state = {
            key: value.detach().cpu().clone()
            for key, value
            in model.state_dict().items()
        }

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss.item():.6f} "
            f"Val Loss: {val_loss.item():.6f}"
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
# RESTORE BEST VALIDATION CHECKPOINT
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# FINAL TEST
# ============================================================

print("\n===== FINAL TEST EVALUATION =====")

model.eval()

with torch.no_grad():

    test_predictions = (
        model(
            X_test_tensor
        )
        .numpy()
        .flatten()
    )


# Convert 0-1 → 0-100
actual_100 = (
    y_test * 100
)

predicted_100 = (
    test_predictions * 100
)

predicted_100 = np.clip(
    predicted_100,
    0,
    100
)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    actual_100,
    predicted_100
)

rmse = np.sqrt(
    mean_squared_error(
        actual_100,
        predicted_100
    )
)

r2 = r2_score(
    actual_100,
    predicted_100
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
# PERFORMANCE ON NON-ZERO WORDS
# ============================================================

nonzero_mask = (
    actual_100 > 0
)

print(
    "\n===== NON-ZERO DIFFICULTY ====="
)

print(
    "Samples:",
    int(nonzero_mask.sum())
)

if nonzero_mask.sum() > 0:

    nonzero_mae = mean_absolute_error(
        actual_100[nonzero_mask],
        predicted_100[nonzero_mask]
    )

    print(
        "MAE:",
        round(nonzero_mae, 4)
    )


# ============================================================
# PERFORMANCE BY DIFFICULTY BAND
# ============================================================

def difficulty_band(
    score
):

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


print(
    "\n===== PERFORMANCE BY DIFFICULTY ====="
)


bands = [
    ("Easy", 0, 20),
    ("Moderate", 20, 40),
    ("Difficult", 40, 60),
    ("Very Difficult", 60, 80),
    ("Extremely Difficult", 80, 101),
]


for name, lower, upper in bands:

    mask = (
        (actual_100 >= lower)
        &
        (actual_100 < upper)
    )

    count = int(
        mask.sum()
    )

    if count == 0:
        continue

    group_mae = mean_absolute_error(
        actual_100[mask],
        predicted_100[mask]
    )

    print(
        f"{name:18s} "
        f"Samples: {count:4d} | "
        f"MAE: {group_mae:.4f}"
    )


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print(
    "\n===== SAMPLE PREDICTIONS ====="
)


sample_count = min(
    20,
    len(test_idx)
)

sample_indices = np.random.choice(
    len(test_idx),
    size=sample_count,
    replace=False
)


for i in sample_indices:

    word = df.iloc[
        test_idx[i]
    ]["word"]

    actual = actual_100[i]

    predicted = predicted_100[i]

    print(
        f"{word:20s} "
        f"Actual: {actual:6.2f}/100 | "
        f"Predicted: {predicted:6.2f}/100 | "
        f"{difficulty_band(predicted)}"
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

    "split":
        "70% train / 15% validation / 15% test",

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
    f"\nV3 clean model saved to: {MODEL_PATH}"
)