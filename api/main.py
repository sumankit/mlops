"""
FastAPI inference service for the Student Dropout/Performance-Risk model.

Loads the joblib-exported best model (produced by ml/train.py) and serves
predictions at POST /predict. Every request is logged (features hash,
prediction, probability, latency) to logs/prediction_log.csv for the
monitoring module (monitoring/monitor.py) to consume.

Run locally:
    uvicorn api.main:app --reload --port 8000

Run in Docker:
    docker build -t student-risk-api .
    docker run -p 8000:8000 student-risk-api
"""
import os
import sys
import json
import time
import hashlib
import logging
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROJECT_ROOT, LOG_DIR
from api.schemas import StudentFeatures, PredictionResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("inference_api")

MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "best_model.joblib")
MODEL_INFO_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "model_info.json")
PREDICTION_LOG_CSV = os.path.join(LOG_DIR, "prediction_log.csv")

app = FastAPI(
    title="Student Dropout-Risk Prediction API",
    description="Serves early-intervention risk predictions from the Part 2 MLOps pipeline.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_state = {"model": None, "model_info": None}


@app.on_event("startup")
def load_model():
    if not os.path.exists(MODEL_PATH):
        log.error(f"Model file not found at {MODEL_PATH}. Run `python ml/train.py` first.")
        return
    _state["model"] = joblib.load(MODEL_PATH)
    with open(MODEL_INFO_PATH) as f:
        _state["model_info"] = json.load(f)
    log.info(f"Loaded model '{_state['model_info']['model_name']}' "
              f"(registry v{_state['model_info']['registered_version']})")


def _log_prediction(payload: dict, prediction: str, probability: float, latency_ms: float):
    row = {
        "timestamp": datetime.now().isoformat(),
        "input_hash": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12],
        "prediction": prediction,
        "probability": round(probability, 4),
        "latency_ms": round(latency_ms, 2),
        **payload,
    }
    df = pd.DataFrame([row])
    header = not os.path.exists(PREDICTION_LOG_CSV)
    df.to_csv(PREDICTION_LOG_CSV, mode="a", header=header, index=False)


@app.get("/health", response_model=HealthResponse)
def health():
    info = _state["model_info"]
    return HealthResponse(
        status="ok" if _state["model"] is not None else "model_not_loaded",
        model_loaded=_state["model"] is not None,
        model_name=info["model_name"] if info else None,
        model_version=info["registered_version"] if info else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: StudentFeatures):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run ml/train.py and restart the API.")

    start = time.time()
    payload = features.model_dump()
    X = pd.DataFrame([payload])

    try:
        proba = float(_state["model"].predict_proba(X)[0, 1])
    except Exception as exc:
        log.error(f"Inference failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    prediction = "High Risk" if proba >= 0.5 else "Low Risk"
    latency_ms = (time.time() - start) * 1000

    _log_prediction(payload, prediction, proba, latency_ms)

    info = _state["model_info"]
    return PredictionResponse(
        risk_prediction=prediction,
        risk_probability=round(proba, 4),
        model_name=info["model_name"],
        model_version=info["registered_version"],
        latency_ms=round(latency_ms, 2),
    )


@app.get("/")
def root():
    return {
        "service": "Student Dropout-Risk Prediction API",
        "docs": "/docs",
        "endpoints": ["/health", "/predict"],
    }
