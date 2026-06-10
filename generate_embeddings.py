# generate_embeddings.py — einmalig lokal ausführen!
# Berechnet BERT-Embeddings fuer train.csv und test.csv
# und speichert sie als data/train_embeddings.csv und data/test_embeddings.csv

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re

print("Lade BERT Modell (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\(website hidden by airbnb\)', '', text)
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_embeddings(csv_path: str, output_path: str):
    print(f"\nVerarbeite {csv_path}...")
    df = pd.read_csv(csv_path)

    descriptions = df["description"].fillna("").apply(clean_text).tolist()

    print(f"  Berechne BERT Embeddings fuer {len(descriptions)} Zeilen...")
    embeddings = model.encode(
        descriptions,
        batch_size=64,
        show_progress_bar=True,
    )

    # 384 Spalten: bert_0, bert_1, ..., bert_383
    cols = [f"bert_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=cols)

    emb_df.to_csv(output_path, index=False)
    print(f"  Gespeichert: {output_path} ({emb_df.shape[1]} Spalten)")

compute_embeddings("data/train.csv",            "data/train_embeddings.csv")
compute_embeddings("data/validation_full.csv", "data/validation_full_embeddings.csv")

print("\nFertig! Fuehre jetzt evaluate.py aus.")
