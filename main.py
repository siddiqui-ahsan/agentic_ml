# main.py

import os
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from src.agent.run import run_agent

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

@app.get("/info")
def info():
    return {
        "model":     "GradientBoostingClassifier + TF-IDF fallback",
        "framework": "LangGraph + FastAPI",
        "tiers":     {0: "Budget", 1: "Standard", 2: "Premium", 3: "Ultra-Luxury"},
    }

@app.post("/predict")
async def predict(
    train_file:      UploadFile = File(...),
    validation_file: UploadFile = File(...),
):
    tmp_dir = tempfile.mkdtemp()

    try:
        train_path  = os.path.join(tmp_dir, "train.csv")
        test_path   = os.path.join(tmp_dir, "validation.csv")
        output_path = os.path.join(tmp_dir, "predictions.csv")

        with open(train_path, "wb") as f:
            f.write(await train_file.read())
        with open(test_path, "wb") as f:
            f.write(await validation_file.read())

        result_path = run_agent(
            train_path=train_path,
            test_path=test_path,
            output_path=output_path,
        )

        # Cleanup happens AFTER the file is sent — not before
        return FileResponse(
            path=result_path,
            media_type="text/csv",
            filename="predictions.csv",
            background=BackgroundTask(shutil.rmtree, tmp_dir, True),
        )

    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
