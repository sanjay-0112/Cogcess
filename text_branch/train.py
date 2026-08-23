import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import CogcessTextModel


# ==========================================
# 1. LOAD PROCESSED DATA
# ==========================================

DATA_PATH = "data/processed/agentlans_features_sample.csv"

print("Loading processed dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ==========================================
# 2. SELECT FEATURES AND TARGET
# ==========================================

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

TARGET_COLUMN = "grade"


X = df[FEATURE_COLUMNS].values
y = df[TARGET_COLUMN].values


# ==========================================
# 3. TRAIN / VALIDATION SPLIT
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# ==========================================
# 4. STANDARDIZE FEATURES
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# Convert NumPy arrays to PyTorch tensors

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)


# ==========================================
# 5. CREATE MODEL
# ==========================================

model = CogcessTextModel(input_size=len(FEATURE_COLUMNS))

print("\n===== MODEL =====")
print(model)


# ==========================================
# 6. LOSS FUNCTION AND OPTIMIZER
# ==========================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 7. TRAINING LOOP
# ==========================================

EPOCHS = 100

print("\n===== TRAINING =====")

for epoch in range(EPOCHS):

    # Training mode
    model.train()

    # Forward pass
    predictions = model(X_train)

    # Calculate loss
    loss = criterion(predictions, y_train)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    # Validation
    model.eval()

    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion(val_predictions, y_val)

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {loss.item():.4f} "
            f"Validation Loss: {val_loss.item():.4f}"
        )


# ==========================================
# 8. FINAL EVALUATION
# ==========================================

model.eval()

with torch.no_grad():
    predictions = model(X_val).numpy().flatten()

actual = y_val.numpy().flatten()


mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
r2 = r2_score(actual, predictions)


print("\n===== FINAL RESULTS =====")

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# ==========================================
# 9. SHOW SAMPLE PREDICTIONS
# ==========================================

print("\n===== SAMPLE PREDICTIONS =====")

for i in range(min(10, len(predictions))):

    print(
        f"Actual: {actual[i]:6.2f} "
        f"| Predicted: {predictions[i]:6.2f}"
    )


# ==========================================
# 10. SAVE MODEL
# ==========================================

MODEL_PATH = "models/cogcess_text_model.pth"

torch.save(model.state_dict(), MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")