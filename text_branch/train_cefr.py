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

from model import CogcessTextModel


# ==========================================
# 1. LOAD DATA
# ==========================================

DATA_PATH = "data/processed/readme_features.csv"

print("Loading ReadMe++ feature dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

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

X = df[FEATURE_COLUMNS].values
y = df[TARGET_COLUMN].values


# ==========================================
# 3. TRAIN / VALIDATION SPLIT
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# ==========================================
# 4. STANDARDIZE FEATURES
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# ==========================================
# 5. CONVERT TO PYTORCH TENSORS
# ==========================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.long
)


# ==========================================
# 6. CREATE CLASSIFICATION MODEL
# ==========================================

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


model = CogcessCEFRModel(
    input_size=len(FEATURE_COLUMNS),
    num_classes=6
)

print("\n===== CEFR MODEL =====")
print(model)


# ==========================================
# 7. CLASS WEIGHTS
# ==========================================

class_counts = np.bincount(y_train)

class_weights = len(y_train) / (
    len(class_counts) * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)

print("\nClass counts:", class_counts)
print("Class weights:", class_weights)


# ==========================================
# 8. LOSS + OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 9. TRAINING
# ==========================================

EPOCHS = 100

print("\n===== TRAINING =====")

for epoch in range(EPOCHS):

    model.train()

    logits = model(X_train)

    loss = criterion(
        logits,
        y_train
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    # Validation

    model.eval()

    with torch.no_grad():

        val_logits = model(X_val)

        val_loss = criterion(
            val_logits,
            y_val
        )

        predictions = torch.argmax(
            val_logits,
            dim=1
        )


    if (epoch + 1) % 10 == 0:

        accuracy = (
            predictions == y_val
        ).float().mean().item()

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {loss.item():.4f} "
            f"Validation Loss: {val_loss.item():.4f} "
            f"Accuracy: {accuracy:.4f}"
        )


# ==========================================
# 10. FINAL EVALUATION
# ==========================================

model.eval()

with torch.no_grad():

    logits = model(X_val)

    predictions = torch.argmax(
        logits,
        dim=1
    ).numpy()

actual = y_val.numpy()


accuracy = accuracy_score(
    actual,
    predictions
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        actual,
        predictions,
        average="macro",
        zero_division=0
    )
)


print("\n===== FINAL RESULTS =====")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Macro F1:  {f1:.4f}")


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

matrix = confusion_matrix(
    actual,
    predictions
)

print("\n===== CONFUSION MATRIX =====")

print(matrix)


# ==========================================
# 12. SAVE MODEL
# ==========================================

MODEL_PATH = "models/cogcess_cefr_model.pth"

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print(
    f"\nModel saved to: {MODEL_PATH}"
)