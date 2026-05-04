"""Minimal FastAPI skeleton for a future API service boundary."""

from __future__ import annotations

from typing import Any


API_TITLE = "Mekong FNO Forecast API"
API_VERSION = "0.1.0"


def live_payload() -> dict[str, str]:
    return {"status": "ok"}


def ready_payload() -> dict[str, Any]:
    return {
        "ready": False,
        "reason": "forecast service is not connected to this API skeleton yet",
    }


def status_payload() -> dict[str, Any]:
    return {
        "service": "mekong-fno",
        "api": "fastapi-skeleton",
        "version": API_VERSION,
        "forecast_enabled": False,
    }


def create_fastapi_app():
    """Create the minimal API app without loading model artifacts."""
    try:
        from fastapi import FastAPI
    except ModuleNotFoundError as exc:
        if exc.name == "fastapi":
            raise RuntimeError(
                "FastAPI is not installed. Install project dependencies before running the API app."
            ) from exc
        raise

    api = FastAPI(title=API_TITLE, version=API_VERSION)

    @api.get("/health/live")
    def health_live():
        return live_payload()

    @api.get("/health/ready")
    def health_ready():
        return ready_payload()

    @api.get("/status")
    def status():
        return status_payload()

    return api


try:
    app = create_fastapi_app()
except RuntimeError:
    app = None
