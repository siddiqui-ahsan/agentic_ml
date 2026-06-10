# main.py

import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response

from src.agent.nodes.model import (
    normalize_columns,
    add_landmark_distances,
    safe_select_features,
    ALL_FEATURE_COLS,
)
from src.agent.run import run_agent

# ── Global model state ────────────────────────────────────────────────────────
_pipeline  = None
_flag_cols = []

# ── Startup: train once on the bundled train.csv ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _flag_cols

    train_path = os.getenv("TRAIN_PATH", "data/train.csv")
    test_path  = os.getenv("TEST_PATH",  "data/test.csv")

    print(f"[startup] Training model on {train_path} ...")
    tmp_out = tempfile.mktemp(suffix=".csv")

    state = run_agent(
        train_path=train_path,
        test_path=test_path,
        output_path=tmp_out,
    )

    # run_agent returns the output path — we need to re-run just to get the pipeline.
    # Instead we import the train node directly and store the pipeline.
    from src.agent.nodes.model import ml_train_node, normalize_columns, add_landmark_distances
    import pandas as pd

    train_df = pd.read_csv(train_path)
    train_df = normalize_columns(train_df)
    train_df = add_landmark_distances(train_df)

    mock_state = {
        "train_df":   train_df,
        "llm_failed": True,
        "train_flags": None,
        "errors": [],
    }
    result = ml_train_node(mock_state)
    _pipeline  = result["pipeline"]
    _flag_cols = result["flag_cols"]

    if os.path.exists(tmp_out):
        os.remove(tmp_out)

    print("[startup] Model ready.")
    yield
    # shutdown — nothing to clean up


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Airbnb Price Tier Agent",
    description="LangGraph agent predicting Airbnb price tiers (0=Budget → 3=Ultra-Luxury)",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Agent is running", "model_ready": _pipeline is not None}

@app.get("/health")
def health():
    return {"status": "healthy", "model_ready": _pipeline is not None}

@app.post("/predict")
async def predict(
    train_file:      UploadFile = File(...),
    validation_file: UploadFile = File(...),
):
    """
    Accepts train + validation CSV.
    Re-trains only if a train_file is provided that differs from the cached model.
    For evaluation: just runs predict on validation_file using the pre-trained model.
    """
    import pandas as pd
    import numpy as np

    tmp_dir = tempfile.mkdtemp()
    try:
        # Save uploaded files
        val_path = os.path.join(tmp_dir, "validation.csv")
        with open(val_path, "wb") as f:
            f.write(await validation_file.read())

        # Read and prepare validation data
        test_df = pd.read_csv(val_path)
        test_df = normalize_columns(test_df)
        test_df = add_landmark_distances(test_df)

        feature_cols = ALL_FEATURE_COLS + _flag_cols
        X_test = safe_select_features(test_df, feature_cols)

        # Predict using pre-trained model
        predictions = _pipeline.predict(X_test)

        if "property_id" in test_df.columns:
            property_ids = test_df["property_id"].values
        else:
            property_ids = list(range(len(test_df)))

        # Build CSV response
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
    )
