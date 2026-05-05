"""Minimal FastAPI skeleton for a future API service boundary."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.schemas import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    RuntimeStatus,
    StatusResponse,
)


API_TITLE = "Mekong FNO Forecast API"
API_VERSION = "0.1.0"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def live_payload() -> dict[str, str]:
    return {"status": "ok"}


def ready_payload() -> dict[str, Any]:
    return {
        "ready": True,
        "status": "ok",
    }


def status_payload() -> StatusResponse:
    return StatusResponse(
        ready=False,
        service_status="not_ready",
        generated_at=_utc_now_iso(),
        latest_data_date=None,
        data_freshness_days=None,
        active_model_id=None,
        backend_mode="local",
        artifacts_ok=False,
        upstream_status={},
        runtime_status=RuntimeStatus(),
        warnings=["Placeholder status response; real runtime readiness is not connected yet."],
    )


def placeholder_forecast_payload(request: ForecastRequest) -> ForecastResponse:
    generated_at = _utc_now_iso()
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
        model_id=None,
        assist_enabled=False,
        uncertainty_available=False,
        predictions=points,
        metrics=None,
        backtest=None,
        warnings=["Placeholder forecast response; real inference is not connected yet."],
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
