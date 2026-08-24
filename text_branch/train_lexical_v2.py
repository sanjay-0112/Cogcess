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
    r2_score
)


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/processed/mendeley_lexical.csv"
SEMANTIC_PATH = "data/processed/mendeley_semantic_features.npz"
MODEL_PATH = "models/cogcess_lexical_v2_model.pth"

RANDOM_STATE = 42
EPOCHS = 100
LEARNING_RATE = 0.001

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Mendeley lexical dataset...")

df = pd.read_csv(DATA_PATH)

print("Rows:", len(df))


# ============================================================
# REMOVE DUPLICATE WORDS
# ============================================================

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
embedding_words = semantic_data["words"]


print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# VERIFY WORD ALIGNMENT
# ============================================================

dataset_words = df["word"].astype(str).values
embedding_words = embedding_words.astype(str)


if len(dataset_words) != len(embedding_words):

    raise ValueError(
        "Number of dataset words and embeddings do not match."
    )


if not np.array_equal(
    dataset_words,
    embedding_words
):

    raise ValueError(
        "Word order mismatch between lexical dataset "
        "and semantic embeddings."
    )


print("Word alignment: OK")


# ============================================================
# TARGET
# ============================================================

df["is_difficult"] = (
    df["difficult_ug"] > 0
).astype(int)


y_class = df[
    "is_difficult"
].values.astype(np.float32)


y_reg = df[
    "lexical_difficulty"
].values.astype(np.float32)


# ============================================================
# TRADITIONAL FEATURES
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


# ============================================================
# STANDARDIZE NUMERIC FEATURES
# ============================================================

scaler = StandardScaler()

X_numeric = scaler.fit_transform(
    X_numeric
).astype(np.float32)


# ============================================================
# POS ONE-HOT ENCODING
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
# COMBINE FEATURES
# ============================================================

X = np.concatenate(
    [
        X_numeric,
        X_pos,
        embeddings.astype(np.float32)
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
    embeddings.shape[1]
)

print(
    "TOTAL FEATURES:",
    X.shape[1]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

indices = np.arange(
    len(df)
)


train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_class
)


X_train = X[
    train_idx
]

X_test = X[
    test_idx
]


y_class_train = y_class[
    train_idx
]

y_class_test = y_class[
    test_idx
]


y_reg_train = y_reg[
    train_idx
]

y_reg_test = y_reg[
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

y_class_train_tensor = torch.tensor(
    y_class_train
).unsqueeze(1)

y_class_test_tensor = torch.tensor(
    y_class_test
).unsqueeze(1)


# ============================================================
# CLASSIFIER
# ============================================================

class CogcessLexicalV2Classifier(
    nn.Module
):

    def __init__(self, input_size):

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
            )
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# REGRESSOR
# ============================================================

class CogcessLexicalV2Regressor(
    nn.Module
):

    def __init__(self, input_size):

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
            )
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# CLASSIFIER TRAINING
# ============================================================

classifier = CogcessLexicalV2Classifier(
    X.shape[1]
)


print("\n===== V2 CLASSIFIER =====")

print(classifier)


positive_count = y_class_train.sum()

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


print(
    "\nPositive class weight:",
    pos_weight.item()
)


criterion_class = (
    nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )
)


optimizer_class = torch.optim.Adam(
    classifier.parameters(),
    lr=LEARNING_RATE
)


print("\n===== V2 CLASSIFIER TRAINING =====")


for epoch in range(EPOCHS):

    classifier.train()

    optimizer_class.zero_grad()

    logits = classifier(
        X_train_tensor
    )

    loss = criterion_class(
        logits,
        y_class_train_tensor
    )

    loss.backward()

    optimizer_class.step()


    if (epoch + 1) % 10 == 0:

        classifier.eval()

        with torch.no_grad():

            test_logits = classifier(
                X_test_tensor
            )

            test_probs = torch.sigmoid(
                test_logits
            )

            test_preds = (
                test_probs >= 0.5
            ).float()


            test_loss = criterion_class(
                test_logits,
                y_class_test_tensor
            )


            accuracy = accuracy_score(
                y_class_test,
                test_preds.numpy().flatten()
            )


        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {loss.item():.4f} "
            f"Test Loss: {test_loss.item():.4f} "
            f"Accuracy: {accuracy:.4f}"
        )


# ============================================================
# CLASSIFIER EVALUATION
# ============================================================

classifier.eval()

with torch.no_grad():

    probabilities = torch.sigmoid(
        classifier(
            X_test_tensor
        )
    ).numpy().flatten()


predictions = (
    probabilities >= 0.5
).astype(int)


print("\n===== V2 CLASSIFIER RESULTS =====")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_class_test,
            predictions
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_class_test,
            predictions,
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
            predictions,
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
            predictions,
            zero_division=0
        ),
        4
    )
)


# ============================================================
# REGRESSION DATA
# ONLY DIFFICULT WORDS
# ============================================================

train_difficult_mask = (
    y_class_train == 1
)

test_difficult_mask = (
    y_class_test == 1
)


X_reg_train = X_train[
    train_difficult_mask
]

y_reg_train_filtered = y_reg_train[
    train_difficult_mask
]


X_reg_test = X_test[
    test_difficult_mask
]

y_reg_test_filtered = y_reg_test[
    test_difficult_mask
]


X_reg_train_tensor = torch.tensor(
    X_reg_train
)

y_reg_train_tensor = torch.tensor(
    y_reg_train_filtered
).unsqueeze(1)

X_reg_test_tensor = torch.tensor(
    X_reg_test
)


# ============================================================
# REGRESSOR TRAINING
# ============================================================

regressor = CogcessLexicalV2Regressor(
    X.shape[1]
)


print("\n===== V2 REGRESSOR =====")

print(regressor)


criterion_reg = nn.MSELoss()

optimizer_reg = torch.optim.Adam(
    regressor.parameters(),
    lr=LEARNING_RATE
)


print("\n===== V2 REGRESSOR TRAINING =====")


for epoch in range(EPOCHS):

    regressor.train()

    optimizer_reg.zero_grad()

    outputs = regressor(
        X_reg_train_tensor
    )

    loss = criterion_reg(
        outputs,
        y_reg_train_tensor
    )

    loss.backward()

    optimizer_reg.step()


    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {loss.item():.4f}"
        )


# ============================================================
# REGRESSION EVALUATION
# ============================================================

regressor.eval()

with torch.no_grad():

    reg_predictions = (
        regressor(
            X_reg_test_tensor
        )
        .numpy()
        .flatten()
    )


reg_predictions = np.clip(
    reg_predictions,
    0,
    100
)


print("\n===== V2 REGRESSION RESULTS =====")

print(
    "Difficult test words:",
    len(y_reg_test_filtered)
)

print(
    "MAE:",
    round(
        mean_absolute_error(
            y_reg_test_filtered,
            reg_predictions
        ),
        4
    )
)

print(
    "RMSE:",
    round(
        np.sqrt(
            mean_squared_error(
                y_reg_test_filtered,
                reg_predictions
            )
        ),
        4
    )
)

print(
    "R²:",
    round(
        r2_score(
            y_reg_test_filtered,
            reg_predictions
        ),
        4
    )
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n===== V2 SAMPLE PREDICTIONS =====")

sample_count = min(
    10,
    len(y_reg_test_filtered)
)


for i in range(sample_count):

    print(
        f"Actual: "
        f"{y_reg_test_filtered[i]:6.2f} | "
        f"Predicted: "
        f"{reg_predictions[i]:6.2f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

checkpoint = {

    "classifier_state_dict":
        classifier.state_dict(),

    "regressor_state_dict":
        regressor.state_dict(),

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

    "difficulty_scale":
        "0-100 normalized from undergraduate difficulty count / 25"

}


torch.save(
    checkpoint,
    MODEL_PATH
)


print(
    f"\nV2 model saved to: {MODEL_PATH}"
)