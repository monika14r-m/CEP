"""Validated request schemas for the ML-IDS inference API.

Numeric inputs must be finite, unknown fields are rejected, and batch requests
are bounded so one request cannot monopolize inference resources. Validation
errors are sanitized before JSON encoding so NaN/Infinity inputs still receive
a proper 422 response instead of a server error.
"""

import math
from typing import Annotated, List, Optional

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


MAX_BATCH_SIZE = 256

ThresholdValue = Annotated[
    float,
    Field(ge=0.0, le=1.0, allow_inf_nan=False),
]


class ConnectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

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

    # /dataset includes these display-only labels. They are accepted so the
    # live dashboard can send a selected sample unchanged; inference uses only
    # FEATURE_COLS from deploy_meta.json.
    attack_type: Optional[str] = Field(default=None, max_length=64)
    true_label: Optional[str] = Field(default=None, max_length=16)


ConnectionBatch = Annotated[
    List[ConnectionRecord],
    Field(min_length=1, max_length=MAX_BATCH_SIZE),
]


def _sanitize_error_value(value):
    """Convert non-finite floats inside validation details to JSON-safe text."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _sanitize_error_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_error_value(item) for item in value]
    return value


async def safe_validation_error_response(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return a 422 without echoing submitted feature values in the response."""
    errors = [
        {
            key: _sanitize_error_value(value)
            for key, value in error.items()
            if key != "input"
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": errors}),
    )
