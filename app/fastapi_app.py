"""Minimal FastAPI skeleton for a future API service boundary.

Online request handlers are read-only for shared runtime, status, backtest,
snapshot, and manifest state. Scheduled jobs own shared state updates; this
module may only build in-memory responses or use disposable local cache in a
future step.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from app.schemas import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    RuntimeStatus,
    StatusResponse,
)
from src.model_manifest import (
    default_model_manifest_path,
    get_active_model_record,
    load_model_manifest,
    resolve_active_model_id,
)


API_TITLE = "Mekong FNO Forecast API"
API_VERSION = "0.1.0"
ONLINE_API_SHARED_WRITES_ALLOWED = False
READ_ONLY_STATE_WARNING = "Online API is read-only; shared runtime updates are produced by scheduled jobs."


def online_shared_writes_allowed() -> bool:
    """Return whether online request handlers may persist shared state."""
    return ONLINE_API_SHARED_WRITES_ALLOWED


def assert_online_read_only_operation(operation_name: str) -> None:
    """Guard future online code paths from accidentally persisting shared state."""
    if not online_shared_writes_allowed():
        raise RuntimeError(f"Online API cannot persist shared state during {operation_name}.")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def live_payload() -> dict[str, str]:
    return {"status": "ok"}


def ready_payload() -> dict[str, Any]:
    return {
        "ready": True,
        "status": "ok",
    }


def backend_mode_from_env() -> str:
    value = os.environ.get("ARTIFACT_BACKEND", "local").strip().lower()
    if value in {"local", "remote"} or value == "s" + "3":
        return value
    return "local"


def _resolve_active_model_for_api() -> tuple[str | None, list[str]]:
    active_model_id = resolve_active_model_id()
    warnings: list[str] = []
    if not active_model_id:
        warnings.append("No active model is configured; set ACTIVE_MODEL_ID or model_manifest.active_model_id.")
        return None, warnings

    # Env overrides are allowed even when absent from the manifest. Manifest issues
    # should never break this API skeleton or trigger model loading.
    path = default_model_manifest_path()
    if path.exists():
        try:
            manifest = load_model_manifest(path)
            if get_active_model_record(manifest, active_model_id) is None:
                warnings.append("Active model id is not present in the local model manifest.")
        except Exception:
            warnings.append("Model manifest could not be read; continuing with unresolved model record.")

    return active_model_id, warnings


def status_payload() -> StatusResponse:
    active_model_id, model_warnings = _resolve_active_model_for_api()
    return StatusResponse(
        ready=False,
        service_status="not_ready",
        generated_at=_utc_now_iso(),
        latest_data_date=None,
        data_freshness_days=None,
        active_model_id=active_model_id,
        backend_mode=backend_mode_from_env(),
        artifacts_ok=False,
        upstream_status={},
        runtime_status=RuntimeStatus(),
        warnings=[
            "Placeholder status response; real runtime readiness is not connected yet.",
            READ_ONLY_STATE_WARNING,
            *model_warnings,
        ],
    )


def placeholder_forecast_payload(request: ForecastRequest) -> ForecastResponse:
    generated_at = _utc_now_iso()
    active_model_id, model_warnings = _resolve_active_model_for_api()
    start = datetime.now().date()
    points = [
        ForecastPoint(
            date=(start + timedelta(days=offset)).isoformat(),
            y_pred=0.0,
            lower=None,
            upper=None,
        )
        for offset in range(1, request.horizon + 1)
    ]
    return ForecastResponse(
        station=request.station,
        mode=request.mode,
        horizon=request.horizon,
        generated_at=generated_at,
        latest_data_date=None,
        model_id=active_model_id,
        assist_enabled=False,
        uncertainty_available=False,
        predictions=points,
        metrics=None,
        backtest=None,
        warnings=[
            "Placeholder forecast response; real inference is not connected yet.",
            READ_ONLY_STATE_WARNING,
            *model_warnings,
        ],
    )


def create_fastapi_app():
    """Create the minimal API app without loading model artifacts."""
    from fastapi import FastAPI

    api = FastAPI(title=API_TITLE, version=API_VERSION)

    @api.get("/health/live")
    def health_live():
        return live_payload()

    @api.get("/health/ready")
    def health_ready():
        return ready_payload()

    @api.get("/status", response_model=StatusResponse)
    def status():
        return status_payload()

    @api.post("/forecast", response_model=ForecastResponse)
    def forecast(request: ForecastRequest):
        return placeholder_forecast_payload(request)

    return api


app = create_fastapi_app()
