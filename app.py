"""
FastAPI inference service for the NSL-KDD IDS model.

Loads the artifacts produced by train_pipeline.py and exposes:
    GET  /health          - liveness check
    GET  /models          - list every trained model + its evaluation metrics
    POST /predict         - single record -> risk score + alert (best model)
    POST /predict_all     - single record -> risk score + alert from EVERY
                             trained model, for real side-by-side comparison
    POST /predict_batch   - list of records -> list of decisions (best model)

Run locally:
    pip install -r requirements.txt
    python train_pipeline.py        # produces artifacts/
    uvicorn app:app --host 0.0.0.0 --port 8000

Then:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_record.json
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastapi.exceptions import RequestValidationError
from api_schemas import (
    ConnectionBatch,
    ConnectionRecord,
    ThresholdValue,
    safe_validation_error_response,
)

# Needed so joblib can deserialize anomaly-model adapters.
from model_wrappers import AnomalyProbabilityWrapper  # noqa: F401

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

model = joblib.load(ARTIFACT_DIR / "model.joblib")
scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
encoders = joblib.load(ARTIFACT_DIR / "encoders.joblib")
meta = json.loads((ARTIFACT_DIR / "deploy_meta.json").read_text())

FEATURE_COLS = meta["feature_cols"]
SKEWED_COLS = meta["skewed_cols"]
MEDIANS = meta["medians"]
MODES = meta["modes"]
DEFAULT_THRESHOLD = meta.get("default_threshold", 0.5)
ALL_MODELS_META = meta.get("all_models", [])

# Load every individually-saved model for real multi-model comparison.
# Falls back gracefully to just the best model if older artifacts (built
# before this change) don't have the per-model files.
ALL_MODELS = {}
for m in ALL_MODELS_META:
    path = ARTIFACT_DIR / f"model_{m['slug']}.joblib"
    if path.exists():
        ALL_MODELS[m["slug"]] = joblib.load(path)
if not ALL_MODELS:
    ALL_MODELS[meta["best_model_name"]] = model
    ALL_MODELS_META = [{"name": meta["best_model_name"], "slug": meta["best_model_name"], "group": "ours"}]

SAMPLE_DATASET_PATH = ARTIFACT_DIR / "sample_dataset.json"
SAMPLE_DATASET = json.loads(SAMPLE_DATASET_PATH.read_text()) if SAMPLE_DATASET_PATH.exists() else []

# Comma-separated list of allowed origins, e.g. "https://you.github.io".
# Defaults to "*" for easy local/demo use — tighten this for production.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(
    title="ML-IDS Inference Service",
    description="Serves risk scores for network connection records using the trained IDS model(s).",
    version="2.0.0",
)

app.add_exception_handler(RequestValidationError, safe_validation_error_response)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    risk_score: float
    alert: bool
    threshold_used: float


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same missing-value, log-transform and encoding steps used at
    training time. Unknown categorical values fall back to the first known
    class rather than raising, since a novel protocol/service string showing
    up in production shouldn't crash the service — it should still get scored."""
    for c, m in MEDIANS.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace(-1, np.nan).fillna(m)

    for c, m in MODES.items():
        if c in df.columns:
            df[c] = df[c].fillna(m)

    for c in SKEWED_COLS:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))

    for c, le in encoders.items():
        if c in df.columns:
            known = set(le.classes_)
            df[c] = df[c].apply(lambda v: v if v in known else le.classes_[0])
            df[c] = le.transform(df[c])

    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0

    return df[FEATURE_COLS]


@app.get("/health")
def health():
    return {"status": "ok", "model": meta["best_model_name"]}


@app.get("/models")
def list_models():
    """Every trained model plus its held-out evaluation metrics, so a
    dashboard can show a real comparison rather than one model's opinion."""
    return {"best_model": meta["best_model_name"], "models": ALL_MODELS_META, "imbalance_strategies": meta.get("supported_imbalance_strategies", [])}


@app.get("/dataset")
def dataset():
    """A small, class-diverse sample of REAL test-set connections (several
    normal + several distinct attack types), each in the exact shape
    /predict and /predict_all expect. Powers the dashboard's live dataset
    picker instead of hand-set demo numbers."""
    return {"count": len(SAMPLE_DATASET), "records": SAMPLE_DATASET}


@app.post("/predict", response_model=PredictionResponse)
def predict(record: ConnectionRecord, threshold: Optional[ThresholdValue] = Query(default=None)):
    t = threshold if threshold is not None else DEFAULT_THRESHOLD
    df = pd.DataFrame([record.model_dump()])
    X = _prepare_frame(df)
    X_scaled = scaler.transform(X)
    score = float(model.predict_proba(X_scaled)[0, 1])
    return PredictionResponse(risk_score=score, alert=score >= t, threshold_used=t)


@app.post("/predict_all")
def predict_all(record: ConnectionRecord, threshold: Optional[ThresholdValue] = Query(default=None)):
    """Score one connection with every trained model side-by-side — powers
    the live model-comparison view in the dashboard."""
    t = threshold if threshold is not None else DEFAULT_THRESHOLD
    df = pd.DataFrame([record.model_dump()])
    X = _prepare_frame(df)
    X_scaled = scaler.transform(X)

    predictions = {}
    for m in ALL_MODELS_META:
        mdl = ALL_MODELS.get(m["slug"])
        if mdl is None:
            continue
        score = float(mdl.predict_proba(X_scaled)[0, 1])
        predictions[m["name"]] = {
            "risk_score": round(score, 4),
            "alert": score >= t,
            "group": m.get("group", "baseline"),
            "metrics": {k: m[k] for k in ("precision", "recall", "f1", "pr_auc") if k in m},
        }
    return {"threshold_used": t, "predictions": predictions}


@app.post("/predict_batch", response_model=List[PredictionResponse])
def predict_batch(records: ConnectionBatch, threshold: Optional[ThresholdValue] = Query(default=None)):
    t = threshold if threshold is not None else DEFAULT_THRESHOLD
    df = pd.DataFrame([r.model_dump() for r in records])
    X = _prepare_frame(df)
    X_scaled = scaler.transform(X)
    scores = model.predict_proba(X_scaled)[:, 1]
    return [
        PredictionResponse(risk_score=float(s), alert=bool(s >= t), threshold_used=t)
        for s in scores
    ]
