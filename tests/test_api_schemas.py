import json
import math
from typing import Optional

import pytest
from fastapi import FastAPI, Query
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from api_schemas import (
    MAX_BATCH_SIZE,
    ConnectionBatch,
    ConnectionRecord,
    ThresholdValue,
    safe_validation_error_response,
)

VALID_RECORD = {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 232,
    "dst_bytes": 8153,
    "count": 5,
    "srv_count": 5,
    "same_srv_rate": 1.0,
    "dst_host_count": 30,
    "dst_host_srv_count": 30,
    "logged_in": 1,
}

# Exercise the same Pydantic types and FastAPI parameter declarations used by
# app.py without importing app.py (which loads trained model artifacts at import).
validation_app = FastAPI()
validation_app.add_exception_handler(
    RequestValidationError, safe_validation_error_response
)


@validation_app.post("/predict")
def predict(
    record: ConnectionRecord,
    threshold: Optional[ThresholdValue] = Query(default=None),
):
    return {"accepted": True, "threshold": threshold}


@validation_app.post("/predict_batch")
def predict_batch(
    records: ConnectionBatch,
    threshold: Optional[ThresholdValue] = Query(default=None),
):
    return {"count": len(records), "threshold": threshold}


client = TestClient(validation_app)


def test_accepts_valid_record():
    response = client.post("/predict", json=VALID_RECORD)
    assert response.status_code == 200
    assert response.json()["accepted"] is True


def test_accepts_dataset_display_metadata():
    payload = {
        **VALID_RECORD,
        "attack_type": "neptune",
        "true_label": "attack",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_rejects_unknown_fields():
    response = client.post("/predict", json={**VALID_RECORD, "unexpected": "value"})
    assert response.status_code == 422


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite_feature_values_with_422(bad_value):
    payload = json.dumps({**VALID_RECORD, "src_bytes": bad_value}, allow_nan=True)
    response = client.post(
        "/predict",
        content=payload,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "finite" in response.text.lower()
    assert repr(bad_value).lower() not in response.text.lower()


@pytest.mark.parametrize("threshold", ["nan", "inf", "-inf", "-0.01", "1.01"])
def test_rejects_invalid_thresholds(threshold):
    response = client.post(f"/predict?threshold={threshold}", json=VALID_RECORD)
    assert response.status_code == 422


@pytest.mark.parametrize("threshold", ["0", "0.5", "1"])
def test_accepts_threshold_boundaries_and_midpoint(threshold):
    response = client.post(f"/predict?threshold={threshold}", json=VALID_RECORD)
    assert response.status_code == 200


def test_rejects_empty_batch():
    response = client.post("/predict_batch", json=[])
    assert response.status_code == 422


def test_rejects_batch_above_limit():
    response = client.post(
        "/predict_batch",
        json=[VALID_RECORD] * (MAX_BATCH_SIZE + 1),
    )
    assert response.status_code == 422


def test_accepts_batch_at_limit():
    response = client.post("/predict_batch", json=[VALID_RECORD] * MAX_BATCH_SIZE)
    assert response.status_code == 200
    assert response.json()["count"] == MAX_BATCH_SIZE


def test_batch_schema_rejects_non_finite_values():
    with pytest.raises(ValidationError):
        TypeAdapter(ConnectionBatch).validate_python(
            [{**VALID_RECORD, "src_bytes": math.inf}]
        )


def test_threshold_schema_rejects_non_finite_values():
    with pytest.raises(ValidationError):
        TypeAdapter(ThresholdValue).validate_python(math.nan)
