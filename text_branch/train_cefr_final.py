import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/processed/readme_features.csv"
MODEL_PATH = "models/cogcess_cefr_final.pth"

RANDOM_STATE = 42
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 25

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING FULL README++ DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# FEATURES / TARGET
# ============================================================

FEATURE_COLUMNS = [
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

TARGET_COLUMN = "cefr_numeric"

X = (
    df[FEATURE_COLUMNS]
    .values
    .astype(np.float32)
)

y = (
    df[TARGET_COLUMN]
    .values
    .astype(np.int64)
)


# ============================================================
# CEFR INFORMATION
# ============================================================

CEFR_NAMES = {
    0: "A1",
    1: "A2",
    2: "B1",
    3: "B2",
    4: "C1",
    5: "C2",
}


print("\n===== CEFR DISTRIBUTION =====")

for class_id in range(6):

    count = int(
        np.sum(y == class_id)
    )

    percentage = (
        count / len(y) * 100
    )

    print(
        f"{CEFR_NAMES[class_id]}: "
        f"{count:4d} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# PROPER 70 / 15 / 15 SPLIT
# ============================================================

indices = np.arange(
    len(df)
)


# First: 15% untouched test set

train_val_idx, test_idx = train_test_split(
    indices,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=y
)


# Remaining 85%:
# take 15/85 = 17.647% for validation.
#
# Overall result:
# 70% train
# 15% validation
# 15% test

train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=(15 / 85),
    random_state=RANDOM_STATE,
    stratify=y[train_val_idx]
)


X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]


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
    "Final test samples:",
    len(X_test)
)


# ============================================================
# STANDARDIZATION
# ============================================================

# IMPORTANT:
# Fit the scaler ONLY on training data.

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
).astype(np.float32)

X_val = scaler.transform(
    X_val
).astype(np.float32)

X_test = scaler.transform(
    X_test
).astype(np.float32)


# ============================================================
# TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.long
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


# ============================================================
# MODEL
# ============================================================

class CogcessCEFRModel(nn.Module):

    def __init__(
        self,
        input_size=9,
        num_classes=6
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
                num_classes
            )
        )


    def forward(self, x):

        return self.network(x)


model = CogcessCEFRModel(
    input_size=len(FEATURE_COLUMNS),
    num_classes=6
)


print("\n" + "=" * 60)
print("COGCESS CEFR FINAL MODEL")
print("=" * 60)

print(model)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    y_train,
    minlength=6
)

class_weights = (
    len(y_train)
    /
    (
        6 * class_counts
    )
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)


print("\n===== CLASS WEIGHTS =====")

print(
    "Class counts:",
    class_counts
)

print(
    "Class weights:",
    class_weights
)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

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

    train_logits = model(
        X_train_tensor
    )

    train_loss = criterion(
        train_logits,
        y_train_tensor
    )

    train_loss.backward()

    optimizer.step()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        val_logits = model(
            X_val_tensor
        )

        val_loss = criterion(
            val_logits,
            y_val_tensor
        )

        val_predictions = torch.argmax(
            val_logits,
            dim=1
        )


    # --------------------------------------------------------
    # VALIDATION METRICS
    # --------------------------------------------------------

    val_accuracy = (
        val_predictions == y_val_tensor
    ).float().mean().item()


    # --------------------------------------------------------
    # BEST CHECKPOINT
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


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss.item():.4f} "
            f"Val Loss: {val_loss.item():.4f} "
            f"Accuracy: {val_accuracy:.4f}"
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
# RESTORE BEST VALIDATION MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

model.eval()

with torch.no_grad():

    test_logits = model(
        X_test_tensor
    )

    test_predictions = torch.argmax(
        test_logits,
        dim=1
    ).numpy()


actual = y_test


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    actual,
    test_predictions
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        actual,
        test_predictions,
        average="macro",
        zero_division=0
    )
)


print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Macro Precision:",
    round(precision, 4)
)

print(
    "Macro Recall:",
    round(recall, 4)
)

print(
    "Macro F1:",
    round(f1, 4)
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    actual,
    test_predictions,
    labels=np.arange(6)
)


print(
    "\n===== CONFUSION MATRIX ====="
)

print(
    "Rows = actual"
)

print(
    "Columns = predicted"
)

print(
    "      A1 A2 B1 B2 C1 C2"
)

for i, row in enumerate(cm):

    print(
        f"{CEFR_NAMES[i]:>2}  "
        + " ".join(
            f"{value:3d}"
            for value in row
        )
    )


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print(
    "\n===== PER-CLASS RESULTS ====="
)


per_class_precision, per_class_recall, per_class_f1, support = (
    precision_recall_fscore_support(
        actual,
        test_predictions,
        labels=np.arange(6),
        zero_division=0
    )
)


for i in range(6):

    print(
        f"{CEFR_NAMES[i]} | "
        f"Precision: {per_class_precision[i]:.4f} | "
        f"Recall: {per_class_recall[i]:.4f} | "
        f"F1: {per_class_f1[i]:.4f} | "
        f"Samples: {support[i]}"
    )


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print(
    "\n===== SAMPLE PREDICTIONS ====="
)


for i in range(
    min(20, len(test_idx))
):

    predicted_class = (
        int(test_predictions[i])
    )

    actual_class = (
        int(actual[i])
    )

    print(
        f"Actual: {CEFR_NAMES[actual_class]} "
        f"| Predicted: {CEFR_NAMES[predicted_class]}"
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "input_size":
        len(FEATURE_COLUMNS),

    "num_classes":
        6,

    "feature_columns":
        FEATURE_COLUMNS,

    "cefr_mapping":
        CEFR_NAMES,

    "scaler_mean":
        scaler.mean_,

    "scaler_scale":
        scaler.scale_,

    "dataset":
        "UniversalCEFR/readme_en",

    "dataset_size":
        len(df),

    "training_samples":
        len(X_train),

    "validation_samples":
        len(X_val),

    "test_samples":
        len(X_test),

    "split":
        "70% train / 15% validation / 15% test",

    "best_validation_loss":
        best_val_loss,

}


torch.save(
    checkpoint,
    MODEL_PATH
)


print(
    f"\nFinal CEFR model saved to: "
    f"{MODEL_PATH}"
)