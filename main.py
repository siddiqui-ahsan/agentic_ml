# main.py

import os
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response

from src.agent.nodes.model import (
    normalize_columns,
    add_landmark_distances,
    safe_select_features,
    ALL_FEATURE_COLS,
    ml_train_node,
)

app = FastAPI(
    title="Airbnb Price Tier Agent",
    description="LangGraph agent predicting Airbnb price tiers (0=Budget → 3=Ultra-Luxury)",
    version="1.0.0",
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Agent is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(
    train_file:            UploadFile = File(...),
    validation_file:       UploadFile = File(...),
    train_embeddings:      UploadFile = File(...),
    validation_embeddings: UploadFile = File(...),
):
    import pandas as pd
    import numpy as np

    tmp_dir = tempfile.mkdtemp()
    try:
        # ── Save uploaded files ───────────────────────────────────────────────
        train_path   = os.path.join(tmp_dir, "train.csv")
        val_path     = os.path.join(tmp_dir, "validation.csv")
        emb_tr_path  = os.path.join(tmp_dir, "emb_train.csv")
        emb_val_path = os.path.join(tmp_dir, "emb_val.csv")

        for path, upload in [
            (train_path,   train_file),
            (val_path,     validation_file),
            (emb_tr_path,  train_embeddings),
            (emb_val_path, validation_embeddings),
        ]:
            with open(path, "wb") as f:
                f.write(await upload.read())

        # ── Load CSVs ─────────────────────────────────────────────────────────
        train_df  = pd.read_csv(train_path)
        val_df    = pd.read_csv(val_path)
        emb_train = pd.read_csv(emb_tr_path)
        emb_val   = pd.read_csv(emb_val_path)

        # ── Merge embeddings as extra columns ─────────────────────────────────
        # Reset index so concat aligns correctly
        train_df  = train_df.reset_index(drop=True)
        val_df    = val_df.reset_index(drop=True)
        emb_train = emb_train.reset_index(drop=True)
        emb_val   = emb_val.reset_index(drop=True)

        train_df = pd.concat([train_df, emb_train], axis=1)
        val_df   = pd.concat([val_df,   emb_val],   axis=1)

        bert_cols = list(emb_train.columns)  # ["bert_0", ..., "bert_383"]

        # ── Normalize + landmark distances ────────────────────────────────────
        train_df = normalize_columns(train_df)
        train_df = add_landmark_distances(train_df)
        val_df   = normalize_columns(val_df)
        val_df   = add_landmark_distances(val_df)

        # ── Train (no BERT loading needed — embeddings already here) ──────────
        state = {
            "train_df":    train_df,
            "llm_failed":  True,
            "train_flags": None,
            "errors":      [],
            "_bert_cols":  bert_cols,   # signal to pipeline
        }

        # Temporarily add bert_cols to ALL_FEATURE_COLS for this request
        import src.agent.nodes.model as model_module
        original_feature_cols = model_module.ALL_FEATURE_COLS.copy()
        model_module.ALL_FEATURE_COLS = original_feature_cols + bert_cols

        # Patch BertTfidfTransformer to use precomputed embeddings
        result   = _train_with_embeddings(train_df, bert_cols)
        pipeline = result["pipeline"]
        flag_cols = result["flag_cols"]

        feature_cols = model_module.ALL_FEATURE_COLS + flag_cols
        X_val = safe_select_features(val_df, feature_cols)

        # Restore original
        model_module.ALL_FEATURE_COLS = original_feature_cols

        predictions = pipeline.predict(X_val)

        # ── Property IDs ──────────────────────────────────────────────────────
        if "property_id" in val_df.columns:
            property_ids = val_df["property_id"].values
        else:
            property_ids = list(range(len(val_df)))

        result_csv = "property_id,predicted_price_tier\n" + "\n".join(
            f"{pid},{pred}" for pid, pred in zip(property_ids, predictions)
        )

        return Response(
            content=result_csv.encode(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _train_with_embeddings(train_df, bert_cols: list) -> dict:
    """Train XGBoost using precomputed BERT columns instead of running BERT."""
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier
    from src.agent.nodes.model import (
        TABULAR_GEO, TABULAR_SKEWED, TABULAR_CAT, LANDMARK_COLS, TARGET_COL,
        safe_select_features,
    )

    flag_cols    = []
    feature_cols = TABULAR_GEO + TABULAR_SKEWED + TABULAR_CAT + LANDMARK_COLS + bert_cols

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
        ("bert",
         Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0)),
                   ("sc",  StandardScaler())]),
         bert_cols),
    ]

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    pipeline = Pipeline([
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

    X_train = safe_select_features(train_df, feature_cols)
    y_train = train_df[TARGET_COL]

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)

    return {"pipeline": pipeline, "flag_cols": flag_cols}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
    )
