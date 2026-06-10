# Agentic NYC Airbnb Price Tier Predictor

Predicts Airbnb listing price tiers (0–3) using a LangGraph agent that combines
geospatial features, BERT text embeddings, and XGBoost classification.

---

## Project Structure

```
agentic_ml/
├── data/
│   ├── train.csv               # 1800 labelled listings
│   ├── test.csv                # 600 listings to predict
│   └── predictions.csv         # output (generated on run)
│
├── src/
│   └── agent/
│       ├── state.py            # shared AgentState TypedDict
│       ├── graph.py            # LangGraph wiring + routing logic
│       ├── run.py              # entry point called by main.py
│       └── nodes/
│           ├── data.py         # ingest + schema repair + llm_extract (pass-through)
│           └── model.py        # ml_train + predict + output
│
├── main.py                     # FastAPI app (Railway deployment)
├── run_local.py                # local test runner
└── .env                        # OPENROUTER_API_KEY (never commit this)
```

---

## Price Tiers

| Tier | Label | Share in train |
|------|-------|---------------|
| 0 | Budget | 25% |
| 1 | Mid | 25% |
| 2 | Upper-Mid | 40% |
| 3 Ultra-Luxury | 10% |

Tier 3 is rare and the hardest to predict — XGBoost uses `sample_weight="balanced"`
to compensate for the class imbalance.

---

## Agent Flow

```
ingest → schema_repair → llm_extract → ml_train → predict → output
```

### Node descriptions

**`ingest`** (`data.py`)
Loads `train.csv` and `test.csv`. Tries `utf-8`, `latin-1`, and `cp1252` encodings
automatically — survives malformed files.

**`schema_repair`** (`data.py`)
Normalises column names (lowercase, strip, replace spaces with underscores).
Fuzzy-matches renamed columns back to canonical names using an alias table
(e.g. `borough` → `neighbourhood_group`, `lat` → `latitude`).
Injects any still-missing columns as `NaN` — the imputer handles the rest.

**`llm_extract`** (`data.py`)
Pass-through node. LLM flag extraction was removed in favour of BERT embeddings
which handle text semantics more robustly. `llm_failed` is set to `True` so
`ml_train` uses the BERT + TF-IDF branch.

**`ml_train`** (`model.py`)
Builds and fits the full sklearn `Pipeline`:
- Geo features (latitude, longitude) → median impute → StandardScaler
- Skewed numerics (reviews, nights, etc.) → log1p → StandardScaler
- Categorical (room_type, neighbourhood, borough) → most-frequent impute → OneHotEncoder
- Landmark distances (6 NYC landmarks) → median impute → StandardScaler
- Description text → `BertTfidfTransformer` (BERT 384-dim + TF-IDF 300-dim = 684 features)
- Classifier: `XGBClassifier` with `sample_weight="balanced"`

**`predict`** (`model.py`)
Applies the same normalisation and landmark distance steps to the test set,
then runs `pipeline.predict()`. Falls back to row index if `property_id` is missing.

**`output`** (`model.py`)
Writes `predictions.csv` with columns `property_id` and `predicted_price_tier`.
Prints tier distribution and any warnings collected during the run.

---

## Feature Engineering

### Landmark Distances
Six Euclidean distances to key NYC landmarks are added as features.
Proximity to expensive areas (Hudson Yards, Tribeca, SoHo) is a strong price signal
that XGBoost can exploit non-linearly.

| Landmark | Coordinates | Signal strength |
|----------|-------------|-----------------|
| Hudson Yards | 40.7527, -74.0022 | Strong (median $5.58M) |
| SoHo | 40.7234, -74.0020 | Strong (median $3.73M) |
| Tribeca | 40.7163, -74.0086 | Strong (median $3.70M) |
| Central Park | 40.7851, -73.9683 | Medium |
| Times Square | 40.7580, -73.9855 | Medium |
| Wall Street | 40.7074, -74.0113 | Medium |

### BERT + TF-IDF Text Features (`BertTfidfTransformer`)
- **BERT** (`all-MiniLM-L6-v2`, 80MB): 384-dim sentence embeddings.
  Understands context and negation — "no skyline view" is correctly negative.
- **TF-IDF** (300 features, trigrams): captures specific luxury/budget terms.
- Combined: 684 text features per listing.
- Falls back to TF-IDF only if `sentence-transformers` is not installed.
- Unicode-safe cleaning preserves accented characters (é, ü, ñ) for multilingual descriptions.

---

## Routing Logic (`graph.py`)

```
route_after_schema:
  train_df is None AND retry_count < 3  →  retry ingest
  otherwise                             →  llm_extract

route_after_llm:
  llm_failed = True   →  ml_train (BERT+TF-IDF branch)
  llm_failed = False  →  ml_train (LLM flags branch)
```

---

## Curveball Handling

| Curveball | Handled by |
|-----------|-----------|
| Renamed columns (e.g. `borough`) | `schema_repair` alias table |
| Wrong encoding (latin-1, cp1252) | `ingest` encoding loop |
| Missing columns | `schema_repair` injects NaN; imputer fills |
| Missing description | `BertTfidfTransformer` fills NaN with empty string |
| Non-English descriptions | Unicode-safe cleaning preserves characters |
| Class imbalance (Tier 3 = 10%) | `compute_sample_weight("balanced")` |

---

## Local Setup

```bash
# Install dependencies
uv add langchain langgraph langchain-community \
    fastapi uvicorn scikit-learn xgboost \
    pandas numpy sentence-transformers python-dotenv

# Run locally
uv run python run_local.py
```

Output is written to `data/predictions_local.csv`.

---

## Evaluation Targets

| Metric | Target |
|--------|--------|
| Macro F1 (cross-val) | > 0.65 |
| F1 Tier 3 (Ultra-Luxury) | > 0.40 |
| No crash on curveball CSV | ✓ |
| `/predict` latency | < 30s |

---

## Team

| Person | Responsibility |
|--------|---------------|
| A | XGBoost model, feature engineering, `model.py` (train side) |
| B | LangGraph architecture, BERT text features, `graph.py`, `data.py` |
| C | FastAPI, `ingest`, schema repair, Railway deployment |
