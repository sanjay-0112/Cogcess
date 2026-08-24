import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/processed/mendeley_lexical.csv"
SEMANTIC_PATH = "data/processed/mendeley_semantic_features.npz"
MODEL_PATH = "models/cogcess_lexical_v4_model.pth"

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
# TARGETS
# ============================================================

# Human undergraduate responses.
#
# difficult_ug = number of students who marked
# the word as difficult.
#
# We normalize by 25 to obtain a 0-1 proportion.

df["difficulty_probability"] = (
    df["difficult_ug"] / 25.0
)

df["difficulty_probability"] = (
    df["difficulty_probability"]
    .clip(0, 1)
)

# Binary target:
# 0 = no students marked difficult
# 1 = at least one student marked difficult

df["is_difficult"] = (
    df["difficult_ug"] > 0
).astype(int)


y_score = (
    df["difficulty_probability"]
    .values
    .astype(np.float32)
)

y_class = (
    df["is_difficult"]
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


# Standardize ONLY the numeric features.
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


# Semantic features
X_semantic = embeddings


# Combine everything
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

print("\n===== TARGET DISTRIBUTION =====")

print(
    "Mean difficulty:",
    round(float(y_score.mean()), 4)
)

print(
    "Median difficulty:",
    round(float(np.median(y_score)), 4)
)

print(
    "Non-zero difficulty:",
    f"{(y_class.mean() * 100):.2f}%"
)

print(
    "Difficult words:",
    int(y_class.sum())
)

print(
    "Easy words:",
    int((y_class == 0).sum())
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

indices = np.arange(
    len(df)
)

# First:
# 85% temporary train/validation
# 15% final test

train_val_idx, test_idx = train_test_split(
    indices,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=y_class
)

# Then split the 85%:
# 70% total train
# 15% validation
#
# Relative validation size:
# 15 / 85

train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=(15 / 85),
    random_state=RANDOM_STATE,
    stratify=y_class[train_val_idx]
)


X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]

y_score_train = y_score[train_idx]
y_score_val = y_score[val_idx]
y_score_test = y_score[test_idx]

y_class_train = y_class[train_idx]
y_class_val = y_class[val_idx]
y_class_test = y_class[test_idx]


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

y_score_train_tensor = torch.tensor(
    y_score_train
).unsqueeze(1)

y_score_val_tensor = torch.tensor(
    y_score_val
).unsqueeze(1)

y_score_test_tensor = torch.tensor(
    y_score_test
).unsqueeze(1)

y_class_train_tensor = torch.tensor(
    y_class_train
).unsqueeze(1)

y_class_val_tensor = torch.tensor(
    y_class_val
).unsqueeze(1)

y_class_test_tensor = torch.tensor(
    y_class_test
).unsqueeze(1)


# ============================================================
# MULTI-TASK MODEL
# ============================================================

class CogcessLexicalV4(
    nn.Module
):

    def __init__(
        self,
        input_size
    ):

        super().__init__()

        # Shared representation
        self.shared = nn.Sequential(

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

            nn.ReLU()
        )

        # Task 1:
        # Is the word difficult?
        self.classifier = nn.Linear(
            32,
            1
        )

        # Task 2:
        # How difficult is it?
        self.regressor = nn.Sequential(

            nn.Linear(
                32,
                16
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                1
            ),

            nn.Sigmoid()
        )


    def forward(self, x):

        shared = self.shared(x)

        class_logits = (
            self.classifier(shared)
        )

        score = (
            self.regressor(shared)
        )

        return class_logits, score


# ============================================================
# CREATE MODEL
# ============================================================

model = CogcessLexicalV4(
    X.shape[1]
)


print("\n===== COGCESS LEXICAL V4 =====")

print(model)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

# Classification imbalance
positive_count = (
    y_class_train.sum()
)

negative_count = (
    len(y_class_train)
    - positive_count
)

pos_weight = torch.tensor(
    [
        negative_count /
        positive_count
    ],
    dtype=torch.float32
)


classification_loss = (
    nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )
)


# ------------------------------------------------------------
# Weighted regression loss
# ------------------------------------------------------------
#
# Easy words receive weight 1.
# Difficult words receive more weight.
#
# Extremely difficult words receive even more weight.

def weighted_regression_loss(
    predictions,
    targets
):

    weights = torch.ones_like(
        targets
    )

    # Any non-zero difficulty
    weights = torch.where(
        targets > 0,
        torch.tensor(
            3.0,
            device=targets.device
        ),
        weights
    )

    # Higher weight for genuinely
    # difficult vocabulary.
    weights = torch.where(
        targets >= 0.50,
        torch.tensor(
            5.0,
            device=targets.device
        ),
        weights
    )

    # Very high difficulty
    weights = torch.where(
        targets >= 0.80,
        torch.tensor(
            7.0,
            device=targets.device
        ),
        weights
    )

    error = (
        predictions - targets
    ) ** 2

    return (
        weights * error
    ).mean()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Weight between the two tasks.
CLASSIFICATION_WEIGHT = 1.0
REGRESSION_WEIGHT = 1.0


# ============================================================
# TRAINING
# ============================================================

print("\n===== V4 TRAINING =====")

best_val_loss = float(
    "inf"
)

best_state = None

patience = 20
epochs_without_improvement = 0


for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()

    train_logits, train_scores = (
        model(
            X_train_tensor
        )
    )

    class_loss = (
        classification_loss(
            train_logits,
            y_class_train_tensor
        )
    )

    regression_loss = (
        weighted_regression_loss(
            train_scores,
            y_score_train_tensor
        )
    )

    total_loss = (
        CLASSIFICATION_WEIGHT *
        class_loss
        +
        REGRESSION_WEIGHT *
        regression_loss
    )

    total_loss.backward()

    optimizer.step()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        val_logits, val_scores = (
            model(
                X_val_tensor
            )
        )

        val_class_loss = (
            classification_loss(
                val_logits,
                y_class_val_tensor
            )
        )

        val_regression_loss = (
            weighted_regression_loss(
                val_scores,
                y_score_val_tensor
            )
        )

        val_total_loss = (
            val_class_loss
            +
            val_regression_loss
        )


    # --------------------------------------------------------
    # BEST MODEL / EARLY STOPPING
    # --------------------------------------------------------

    if (
        val_total_loss.item()
        <
        best_val_loss
    ):

        best_val_loss = (
            val_total_loss.item()
        )

        best_state = {
            key: value.detach().cpu().clone()
            for key, value
            in model.state_dict().items()
        }

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1


    if (
        epochs_without_improvement
        >= patience
    ):

        print(
            f"Early stopping at epoch "
            f"{epoch + 1}"
        )

        break


    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {total_loss.item():.5f} "
            f"Val Loss: {val_total_loss.item():.5f}"
        )


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

model.eval()

with torch.no_grad():

    test_logits, test_scores = (
        model(
            X_test_tensor
        )
    )

    test_probabilities = (
        torch.sigmoid(
            test_logits
        )
        .numpy()
        .flatten()
    )

    test_scores = (
        test_scores
        .numpy()
        .flatten()
    )


# ------------------------------------------------------------
# Classification
# ------------------------------------------------------------

test_predictions = (
    test_probabilities >= 0.5
).astype(int)


print("\n===== V4 CLASSIFICATION RESULTS =====")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_class_test,
            test_predictions
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_class_test,
            test_predictions,
            zero_division=0
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_class_test,
            test_predictions,
            zero_division=0
        ),
        4
    )
)

print(
    "F1:",
    round(
        f1_score(
            y_class_test,
            test_predictions,
            zero_division=0
        ),
        4
    )
)


# ------------------------------------------------------------
# Regression
# ------------------------------------------------------------

test_scores = np.clip(
    test_scores,
    0,
    1
)

actual_100 = (
    y_score_test * 100
)

predicted_100 = (
    test_scores * 100
)


print("\n===== V4 DIFFICULTY RESULTS =====")

print(
    "MAE:",
    round(
        mean_absolute_error(
            actual_100,
            predicted_100
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
                predicted_100
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
            predicted_100
        ),
        4
    )
)


# ============================================================
# DIFFICULTY BANDS
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


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n===== V4 SAMPLE PREDICTIONS =====")

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

    probability = (
        test_probabilities[i]
        * 100
    )

    print(
        f"{word:20s} "
        f"Actual: {actual:6.2f} | "
        f"Predicted: {predicted:6.2f} | "
        f"Difficult prob: {probability:6.2f}% | "
        f"{difficulty_band(predicted)}"
    )


# ============================================================
# PERFORMANCE BY DIFFICULTY GROUP
# ============================================================

print(
    "\n===== PERFORMANCE BY ACTUAL DIFFICULTY ====="
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
        f"MAE: {group_mae:7.3f}"
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

    "architecture":
        "multi-task classifier + weighted regression",

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
    f"\nV4 model saved to: {MODEL_PATH}"
)