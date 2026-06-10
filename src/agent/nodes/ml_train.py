# src/agent/nodes/ml_train.py

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
from xgboost import XGBClassifier                          # ← XGBoost statt GradientBoosting

# ── 1. Column Definitions ─────────────────────────────────────────────────────

TABULAR_GEO    = ["latitude", "longitude"]
TABULAR_SKEWED = ["minimum_nights", "number_of_reviews",
                  "calculated_host_listings_count", "availability_365"]
TABULAR_CAT    = ["neighbourhood_group", "neighbourhood", "room_type"]
TEXT_COL       = "description"
TARGET_COL     = "price_tier"

ALL_FEATURE_COLS = TABULAR_GEO + TABULAR_SKEWED + TABULAR_CAT + [TEXT_COL]

# ── 2. Column Normalization ───────────────────────────────────────────────────

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
    """Lowercase + strip + alias matching. Safe to call multiple times."""
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
    """
    Returns df with exactly desired_cols.
    Missing columns are injected as NaN — imputer handles the rest.
    Never raises KeyError.
    """
    df = df.copy()
    for col in desired_cols:
        if col not in df.columns:
            df[col] = np.nan
            print(f"  [safe_select] Injected missing column '{col}' as NaN")
    return df[desired_cols]


# ── 3. TextCleaner (multilingual-safe) ───────────────────────────────────────

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans text for TF-IDF.
    Uses unicode-safe regex — preserves é, ü, ñ etc.
    Critical for multilingual validation set curveball.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0].fillna("").astype(str)
        else:
            series = pd.Series(X).fillna("").astype(str)
        return series.apply(self._clean)

    def _clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\(website hidden by airbnb\)', '', text)
        text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


# ── 4. Pipeline Builder ───────────────────────────────────────────────────────

def _build_pipeline(flag_cols: list, use_tfidf: bool) -> Pipeline:
    transformers = [
        ("geo",
         Pipeline([
             ("imp", SimpleImputer(strategy="median")),
             ("sc",  StandardScaler()),
         ]),
         TABULAR_GEO),

        ("skewed",
         Pipeline([
             ("imp",   SimpleImputer(strategy="median")),
             ("log1p", FunctionTransformer(np.log1p, validate=False)),
             ("sc",    StandardScaler()),
         ]),
         TABULAR_SKEWED),

        ("cat",
         Pipeline([
             ("imp", SimpleImputer(strategy="most_frequent")),
             ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
         ]),
         TABULAR_CAT),
    ]

    if use_tfidf:
        # Fallback: LLM unavailable → TF-IDF directly on description
        transformers.append((
            "text",
            Pipeline([
                ("cleaner", TextCleaner()),
                ("tfidf",   TfidfVectorizer(
                    max_features=500,
                    ngram_range=(1, 2),
                    stop_words=None,    # multilingual: don't strip non-english words
                )),
            ]),
            TEXT_COL,
        ))
    elif flag_cols:
        # Happy path: use LLM-extracted boolean flags
        transformers.append((
            "flags",
            SimpleImputer(strategy="constant", fill_value=0),
            flag_cols,
        ))

    preprocessor = ColumnTransformer(transformers, remainder="drop")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(        # ← XGBoost
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            eval_metric="mlogloss",          # verhindert sklearn-Warning
            random_state=42,
        )),
    ])


# ── 5. The Node ───────────────────────────────────────────────────────────────

def ml_train_node(state: dict) -> dict:
    print("▶ Node: ml_train")
    print(f"  LLM fallback active: {state['llm_failed']}")

    # Step 1: Normalize columns
    train_df = normalize_columns(state["train_df"])

    # Step 2: Merge LLM flags if available
    train_flags = state.get("train_flags")
    if not state["llm_failed"] and train_flags is not None:
        flag_cols = list(train_flags.columns)
        train_df  = pd.concat([train_df, train_flags], axis=1)
    else:
        flag_cols = []

    # Step 3: Safely select only the columns we need
    feature_cols = ALL_FEATURE_COLS + flag_cols
    X_train = safe_select_features(train_df, feature_cols)
    y_train = train_df[TARGET_COL]

    print(f"  Training on {len(X_train)} rows, {len(feature_cols)} features")
    print(f"  Flag columns: {flag_cols if flag_cols else 'none — using TF-IDF fallback'}")

    # Step 4: Build pipeline
    pipeline = _build_pipeline(flag_cols, use_tfidf=state["llm_failed"])

    # Step 5: Compute sample weights so Tier 3 (10% of data) is not ignored
    # "balanced" automatically calculates: n_samples / (n_classes * n_samples_per_class)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    print(f"  Sample weights — unique values: {np.unique(sample_weights).round(2)}")

    # Step 6: Fit — pass weights through the pipeline via classifier__ prefix
    pipeline.fit(
        X_train,
        y_train,
        classifier__sample_weight=sample_weights,   # ← XGBoost bekommt die Gewichte
    )

    return {**state,
        "pipeline":     pipeline,
        "flag_cols":    flag_cols,
        "current_node": "ml_train",
    }
