import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer


# ============================================================
# SETTINGS
# ============================================================

INPUT_PATH = "data/processed/mendeley_lexical.csv"
OUTPUT_PATH = "data/processed/mendeley_semantic_features.npz"

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Mendeley lexical dataset...")

df = pd.read_csv(INPUT_PATH)

print("Rows:", len(df))


# ============================================================
# LOAD SEMANTIC ENCODER
# ============================================================

print("\nLoading semantic encoder...")

encoder = SentenceTransformer(MODEL_NAME)

print(
    "Embedding dimension:",
    encoder.get_embedding_dimension()
)


# ============================================================
# PREPARE WORDS
# ============================================================

words = df["word"].astype(str).tolist()


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating semantic embeddings...")

embeddings = encoder.encode(
    words,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


print("\nEmbedding shape:", embeddings.shape)


# ============================================================
# SAVE
# ============================================================

np.savez_compressed(
    OUTPUT_PATH,
    embeddings=embeddings,
    words=np.array(words)
)


print(
    f"\nSaved semantic features to: {OUTPUT_PATH}"
)