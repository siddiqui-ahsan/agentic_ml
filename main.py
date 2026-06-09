# main.py — FastAPI wrapper for Railway deployment

import os
import shutil
import tempfile

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from src.agent.run import run_agent

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Airbnb Price Tier Agent",
    description="LangGraph agent that predicts Airbnb listing price tiers (0=Budget → 3=Ultra-Luxury)",
    version="1.0.0",
)

# ── Health Check ──────────────────────────────────────────────────────────────
# Railway uses this to know if your app is alive.
# Without this, Railway thinks the deployment failed.

@app.get("/")
def root():
    return {"status": "ok", "message": "Agent is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}


# ── Main Endpoint: Train + Predict ────────────────────────────────────────────
# Instructor calls this with the hidden validation.csv
# We also need train.csv to fit the model each time.

@app.post("/predict")
async def predict(
    train_file:      UploadFile = File(..., description="train.csv with price_tier labels"),
    validation_file: UploadFile = File(..., description="validation.csv to predict on"),
):
    """
    Accepts train.csv + validation.csv, runs the full LangGraph agent,
    and returns predictions.csv as a download.
    """

    # Use a temp directory — Railway's filesystem is ephemeral anyway
    tmp_dir = tempfile.mkdtemp()

    try:
        # Save uploaded files to disk
        train_path  = os.path.join(tmp_dir, "train.csv")
        test_path   = os.path.join(tmp_dir, "validation.csv")
        output_path = os.path.join(tmp_dir, "predictions.csv")

        with open(train_path, "wb") as f:
            f.write(await train_file.read())

        with open(test_path, "wb") as f:
            f.write(await validation_file.read())

        # Run the full agent
        result_path = run_agent(
            train_path=train_path,
            test_path=test_path,
            output_path=output_path,
        )

        # Return predictions.csv as a downloadable file
        return FileResponse(
            path=result_path,
            media_type="text/csv",
            filename="predictions.csv",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")

    finally:
        # Always clean up temp files — Railway has limited disk
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Bonus Endpoint: Quick Status Check ───────────────────────────────────────
# Useful to verify Railway deployment worked without uploading files

@app.get("/info")
def info():
    return {
        "model":     "GradientBoostingClassifier",
        "llm":       os.getenv("LLM_MODEL", "llama3.1:8b"),
        "framework": "LangGraph + FastAPI",
        "tiers": {
            0: "Budget",
            1: "Standard",
            2: "Premium",
            3: "Ultra-Luxury",
        }
    }


# ── Local Development Entry Point ─────────────────────────────────────────────
# `python main.py` starts the server locally for testing
# Railway uses the Procfile / start command instead (see below)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,   # auto-restart on code changes during dev
    )
