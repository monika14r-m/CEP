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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ConnectionRecord(BaseModel):
    duration: float
    protocol_type: str
    service: str
    flag: str
    src_bytes: float
    dst_bytes: float
    land: int = 0
    wrong_fragment: int = 0
    urgent: int = 0
    hot: int = 0
    num_failed_logins: int = 0
    logged_in: int = 0
    num_compromised: int = 0
    root_shell: int = 0
    su_attempted: int = 0
    num_root: int = 0
    num_file_creations: int = 0
    num_shells: int = 0
    num_access_files: int = 0
    num_outbound_cmds: int = 0
    is_host_login: int = 0
    is_guest_login: int = 0
    count: int
    srv_count: int
    serror_rate: float = 0.0
    srv_serror_rate: float = 0.0
    rerror_rate: float = 0.0
    srv_rerror_rate: float = 0.0
    same_srv_rate: float = 0.0
    diff_srv_rate: float = 0.0
    srv_diff_host_rate: float = 0.0
    dst_host_count: int = 0
    dst_host_srv_count: int = 0
    dst_host_same_srv_rate: float = 0.0
    dst_host_diff_srv_rate: float = 0.0
    dst_host_same_src_port_rate: float = 0.0
    dst_host_srv_diff_host_rate: float = 0.0
    dst_host_serror_rate: float = 0.0
    dst_host_srv_serror_rate: float = 0.0
    dst_host_rerror_rate: float = 0.0
    dst_host_srv_rerror_rate: float = 0.0


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
    return {"best_model": meta["best_model_name"], "models": ALL_MODELS_META}


@app.get("/dataset")
def dataset():
    """A small, class-diverse sample of REAL test-set connections (several
    normal + several distinct attack types), each in the exact shape
    /predict and /predict_all expect. Powers the dashboard's live dataset
    picker instead of hand-set demo numbers."""
    return {"count": len(SAMPLE_DATASET), "records": SAMPLE_DATASET}


@app.post("/predict", response_model=PredictionResponse)
def predict(record: ConnectionRecord, threshold: Optional[float] = None):
    t = threshold if threshold is not None else DEFAULT_THRESHOLD
    df = pd.DataFrame([record.dict()])
    X = _prepare_frame(df)
    X_scaled = scaler.transform(X)
    score = float(model.predict_proba(X_scaled)[0, 1])
    return PredictionResponse(risk_score=score, alert=score >= t, threshold_used=t)


@app.post("/predict_all")
def predict_all(record: ConnectionRecord, threshold: Optional[float] = None):
    """Score one connection with every trained model side-by-side — powers
    the live model-comparison view in the dashboard."""
    t = threshold if threshold is not None else DEFAULT_THRESHOLD
    df = pd.DataFrame([record.dict()])
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
def predict_batch(records: List[ConnectionRecord], threshold: Optional[float] = None):
    t = threshold if threshold is not None else DEFAULT_THRESHOLD
    df = pd.DataFrame([r.dict() for r in records])
    X = _prepare_frame(df)
    X_scaled = scaler.transform(X)
    scores = model.predict_proba(X_scaled)[:, 1]
    return [
        PredictionResponse(risk_score=float(s), alert=bool(s >= t), threshold_used=t)
        for s in scores
    ]
