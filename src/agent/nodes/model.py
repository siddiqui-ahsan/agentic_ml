# src/agent/nodes/model.py
# Kombiniert: ml_train.py + predict.py + output.py

import os
import re
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ── 1. Column Definitions ─────────────────────────────────────────────────────

TABULAR_GEO    = ["latitude", "longitude"]
TABULAR_SKEWED = ["minimum_nights", "number_of_reviews",
                  "calculated_host_listings_count", "availability_365"]
TABULAR_CAT    = ["neighbourhood_group", "neighbourhood", "room_type"]
TEXT_COL       = "description"
TARGET_COL     = "price_tier"

LANDMARK_COLS  = [
    "dist_central_park",
    "dist_hudson_yards",
    "dist_tribeca",
    "dist_soho",
    "dist_times_square",
    "dist_wall_street",
]

ALL_FEATURE_COLS = TABULAR_GEO + TABULAR_SKEWED + TABULAR_CAT + [TEXT_COL] + LANDMARK_COLS

# ── 2. NYC Landmarks ──────────────────────────────────────────────────────────

LANDMARKS = {
    "central_park": (40.7851, -73.9683),
    "hudson_yards": (40.7527, -74.0022),
    "tribeca":      (40.7163, -74.0086),
    "soho":         (40.7234, -74.0020),
    "times_square": (40.7580, -73.9855),
    "wall_street":  (40.7074, -74.0113),
}

def add_landmark_distances(df: pd.DataFrame) -> pd.DataFrame:
    df  = df.copy()
    lat = pd.to_numeric(df["latitude"],  errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    for name, (lm_lat, lm_lon) in LANDMARKS.items():
        df[f"dist_{name}"] = np.sqrt((lat - lm_lat)**2 + (lon - lm_lon)**2)
    return df

# ── 3. Column Normalization ───────────────────────────────────────────────────

COLUMN_ALIASES = {
    "description":        ["desc", "listing_description", "name", "about",
                           "summary", "neighborhood_overview", "space"],
    "neighbourhood_group": ["borough", "area", "neighborhood_group", "district"],
    "neighbourhood":      ["neighborhood", "location", "zone"],
    "room_type":          ["room", "type", "property_type", "accommodation"],
    "property_id":        ["id", "listing_id", "pid"],
    "latitude":           ["lat"],
    "longitude":          ["lon", "lng", "long"],
    "minimum_nights":     ["min_nights", "minimum_stay"],
    "number_of_reviews":  ["num_reviews", "reviews", "review_count"],
    "availability_365":   ["availability", "avail_365"],
    "calculated_host_listings_count": ["host_listings", "host_count"],
    "price_tier":         ["price_cat", "tier", "label", "target"],
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns
                    .str.lower()
                    .str.strip()
                    .str.replace(r'[\s\-]+', '_', regex=True))
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df = df.rename(columns={alias: canonical})
                    print(f"  [normalize] '{alias}' → '{canonical}'")
                    break
    return df

def safe_select_features(df: pd.DataFrame, desired_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in desired_cols:
        if col not in df.columns:
            df[col] = np.nan
            print(f"  [safe_select] Injected missing column '{col}' as NaN")
    return df[desired_cols]

# ── 4. BERT + TF-IDF Transformer ─────────────────────────────────────────────

class BertTfidfTransformer(BaseEstimator, TransformerMixin):
    """
    Combines BERT sentence embeddings with TF-IDF.
    Falls back to TF-IDF only if sentence-transformers is not installed.
    """

    def __init__(self, use_bert: bool = True, tfidf_max_features: int = 300):
        self.use_bert           = use_bert
        self.tfidf_max_features = tfidf_max_features
        self._bert_model        = None
        self._tfidf             = None
        self._bert_available    = False

    def _load_bert(self):
        if self._bert_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._bert_model     = SentenceTransformer("all-MiniLM-L6-v2")
                self._bert_available = True
                print("  [BertTfidf] BERT loaded (all-MiniLM-L6-v2)")
            except ImportError:
                print("  [BertTfidf] sentence-transformers not installed — TF-IDF only")
                self._bert_available = False

    def _to_series(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].fillna("").astype(str)
        return pd.Series(X).fillna("").astype(str)

    def _clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\(website hidden by airbnb\)', '', text)
        text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fit(self, X, y=None):
        self._load_bert()
        texts = self._to_series(X).apply(self._clean).tolist()
        self._tfidf = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=(1, 3),
            stop_words=None,
            min_df=2,
            sublinear_tf=True,
        )
        self._tfidf.fit(texts)
        return self

    def transform(self, X, y=None):
        texts        = self._to_series(X).apply(self._clean).tolist()
        tfidf_matrix = self._tfidf.transform(texts).toarray()
        if self.use_bert and self._bert_available:
            bert_matrix = self._bert_model.encode(
                texts, batch_size=64, show_progress_bar=False,
            )
            return np.hstack([bert_matrix, tfidf_matrix])
        return tfidf_matrix

# ── 5. Pipeline Builder ───────────────────────────────────────────────────────

def _build_pipeline(flag_cols: list, use_tfidf_only: bool) -> Pipeline:
    transformers = [
        ("geo",
         Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("sc",  StandardScaler())]),
         TABULAR_GEO),

        ("skewed",
         Pipeline([("imp",   SimpleImputer(strategy="median")),
                   ("log1p", FunctionTransformer(np.log1p, validate=False)),
                   ("sc",    StandardScaler())]),
         TABULAR_SKEWED),

        ("cat",
         Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                   ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         TABULAR_CAT),

        ("landmarks",
         Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("sc",  StandardScaler())]),
         LANDMARK_COLS),

        ("text",
         BertTfidfTransformer(use_bert=True, tfidf_max_features=300),
         TEXT_COL),
    ]

    if not use_tfidf_only and flag_cols:
        transformers.append((
            "flags",
            SimpleImputer(strategy="constant", fill_value=0),
            flag_cols,
        ))

    preprocessor = ColumnTransformer(transformers, remainder="drop")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])

# ── 6. Train Node ─────────────────────────────────────────────────────────────

def ml_train_node(state: dict) -> dict:
    print("▶ Node: ml_train")
    print(f"  LLM fallback active: {state['llm_failed']}")

    train_df = normalize_columns(state["train_df"])
    print("  Adding landmark distance features...")
    train_df = add_landmark_distances(train_df)

    train_flags = state.get("train_flags")
    if not state["llm_failed"] and train_flags is not None:
        flag_cols = list(train_flags.columns)
        train_df  = pd.concat([train_df, train_flags], axis=1)
    else:
        flag_cols = []

    feature_cols = ALL_FEATURE_COLS + flag_cols
    X_train = safe_select_features(train_df, feature_cols)
    y_train = train_df[TARGET_COL]

    print(f"  Training on {len(X_train)} rows, {len(feature_cols)} features")
    print(f"  LLM flag columns: {flag_cols if flag_cols else 'none — BERT+TF-IDF only'}")

    pipeline       = _build_pipeline(flag_cols, use_tfidf_only=state["llm_failed"])
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    print(f"  Sample weights: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)

    return {**state,
        "pipeline":     pipeline,
        "flag_cols":    flag_cols,
        "current_node": "ml_train",
    }

# ── 7. Predict Node ───────────────────────────────────────────────────────────

def predict_node(state: dict) -> dict:
    print("▶ Node: predict")

    pipeline  = state["pipeline"]
    flag_cols = state.get("flag_cols", [])

    test_df = normalize_columns(state["test_df"])
    test_df = add_landmark_distances(test_df)

    test_flags = state.get("test_flags")
    if not state["llm_failed"] and test_flags is not None:
        test_df = pd.concat([test_df, test_flags], axis=1)

    feature_cols = ALL_FEATURE_COLS + flag_cols
    X_test = safe_select_features(test_df, feature_cols)

    print(f"  Predicting on {len(X_test)} rows")
    predictions = pipeline.predict(X_test)

    if "property_id" in test_df.columns:
        property_ids = test_df["property_id"]
    else:
        print("  [predict] 'property_id' missing — using row index")
        property_ids = pd.Series(range(len(test_df)), name="property_id")

    return {**state,
        "predictions":  predictions,
        "property_ids": property_ids,
        "current_node": "predict",
    }

# ── 8. Output Node ────────────────────────────────────────────────────────────

def output_node(state: dict) -> dict:
    print("▶ Node: output")

    predictions  = state["predictions"]
    property_ids = state["property_ids"]
    output_path  = state.get("output_path", "data/predictions.csv")

    output_df = pd.DataFrame({
        "property_id":          property_ids.values,
        "predicted_price_tier": predictions,
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"  Saved {len(output_df)} predictions → {output_path}")
    print(f"  Distribution:\n{output_df['predicted_price_tier'].value_counts().to_string()}")

    if state["errors"]:
        print(f"  Warnings during run: {state['errors']}")

    return {**state,
        "output_path":  output_path,
        "current_node": "output",
    }
