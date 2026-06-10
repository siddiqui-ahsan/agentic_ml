# evaluate.py — lokal ausfuehren: uv run python evaluate.py

import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

RAILWAY_URL = "https://agenticml-production.up.railway.app/predict"

# ── 1. Split train.csv + Embeddings in 80/20 ─────────────────────────────────
print("Lade Daten...")
train_df = pd.read_csv("data/train.csv")
train_emb = pd.read_csv("data/train_embeddings.csv")

train_split, val_split, emb_train, emb_val = train_test_split(
    train_df, train_emb,
    test_size=0.2, random_state=42, stratify=train_df["price_tier"]
)

val_features = val_split.drop(columns=["price_tier"])

# Temporaer speichern
train_split.to_csv("/tmp/train_split.csv",     index=False)
val_features.to_csv("/tmp/val_features.csv",   index=False)
emb_train.to_csv("/tmp/emb_train.csv",         index=False)
emb_val.to_csv("/tmp/emb_val.csv",             index=False)

print(f"Training:   {len(train_split)} rows")
print(f"Validation: {len(val_split)} rows")
print(f"Label-Verteilung:\n{val_split['price_tier'].value_counts().sort_index()}\n")

# ── 2. Request an Railway ─────────────────────────────────────────────────────
print(f"Sende Request an {RAILWAY_URL}...")

with open("/tmp/train_split.csv", "rb") as train_f, \
     open("/tmp/val_features.csv", "rb") as val_f, \
     open("/tmp/emb_train.csv", "rb") as emb_train_f, \
     open("/tmp/emb_val.csv", "rb") as emb_val_f:

    response = requests.post(
        RAILWAY_URL,
        files={
            "train_file":           ("train.csv",      train_f,     "text/csv"),
            "validation_file":      ("validation.csv", val_f,       "text/csv"),
            "train_embeddings":     ("emb_train.csv",  emb_train_f, "text/csv"),
            "validation_embeddings":("emb_val.csv",    emb_val_f,   "text/csv"),
        },
        timeout=120,
    )

if response.status_code != 200:
    print(f"Fehler: {response.status_code} — {response.text}")
    exit(1)

with open("/tmp/val_predictions.csv", "wb") as f:
    f.write(response.content)

print("Predictions erhalten!\n")

# ── 3. F1-Score berechnen ────────────────────────────────────────────────────
y_true = val_split["price_tier"].values
y_pred = pd.read_csv("/tmp/val_predictions.csv")["predicted_price_tier"].values

macro_f1 = f1_score(y_true, y_pred, average="macro")

print("=" * 45)
print(f"  MACRO F1-SCORE:  {macro_f1:.4f}")
print("=" * 45)
print()
print(classification_report(
    y_true, y_pred,
    target_names=["0-Budget", "1-Standard", "2-Premium", "3-Ultra-Luxury"]
))
