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
from src.config import ConfigError, load_settings
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


def _env_value(config, name: str, default: str = "") -> str:
    return str(config.get(name, default)).strip()


def live_payload() -> dict[str, str]:
    return {"status": "ok"}


def ready_payload() -> dict[str, Any]:
    return {
        "ready": True,
        "status": "ok",
    }


def backend_mode_from_env(env=None) -> str:
    try:
        return load_settings(env=os.environ if env is None else env, validate=False).artifact_backend
    except ConfigError:
        pass
    return "local"


def _resolve_active_model_for_api(env=None, *, include_missing_warning: bool = True) -> tuple[str | None, list[str]]:
    config = os.environ if env is None else env
    active_model_id = resolve_active_model_id(env=config)
    warnings: list[str] = []
    if not active_model_id:
        if include_missing_warning:
            warnings.append("No active model is configured; set ACTIVE_MODEL_ID or model_manifest.active_model_id.")
        return None, warnings

    # Env overrides are allowed even when absent from the manifest. Manifest issues
    # should never break this API skeleton or trigger model loading.
    path = default_model_manifest_path(env=config)
    if path.exists():
        try:
            manifest = load_model_manifest(path)
            if get_active_model_record(manifest, active_model_id) is None:
                warnings.append("Active model id is not present in the local model manifest.")
        except Exception:
            warnings.append("Model manifest could not be read; continuing with unresolved model record.")

    return active_model_id, warnings


def _freshness_days(latest_data_date: str | None) -> int | None:
    if not latest_data_date:
        return None
    try:
        parsed = datetime.fromisoformat(str(latest_data_date)[:10]).date()
    except ValueError:
        return None
    return (datetime.now(timezone.utc).date() - parsed).days


def _storage_exists(storage, ref, label: str, warnings: list[str]) -> bool:
    try:
        return bool(storage.exists(ref))
    except Exception as exc:
        warnings.append(f"{label} availability check failed: {exc.__class__.__name__}.")
        return False


def _placeholder_status_payload(
    *,
    active_model_id: str | None,
    backend_mode: str,
    model_warnings: list[str],
    extra_warnings: list[str] | None = None,
) -> StatusResponse:
    return StatusResponse(
        ready=False,
        service_status="not_ready",
        generated_at=_utc_now_iso(),
        latest_data_date=None,
        data_freshness_days=None,
        active_model_id=active_model_id,
        backend_mode=backend_mode,
        artifacts_ok=False,
        upstream_status={},
        runtime_status=RuntimeStatus(),
        warnings=[
            "Placeholder status response; real runtime readiness is not connected yet.",
            READ_ONLY_STATE_WARNING,
            *model_warnings,
            *(extra_warnings or []),
        ],
    )


def _s3_storage_from_env(config):
    from src.storage import S3StorageBackend

    settings = load_settings(env=config, validate=True)
    return S3StorageBackend(
        bucket=settings.s3_bucket,
        prefix=settings.s3_prefix,
        region_name=settings.aws_region or None,
    )


def _s3_manifest_active_model_id(storage, config, warnings: list[str]) -> str | None:
    try:
        manifest_key = load_settings(env=config, validate=False).model_manifest_key
    except ConfigError:
        manifest_key = _env_value(config, "MODEL_MANIFEST_KEY")
    if not manifest_key:
        return None
    try:
        manifest = storage.read_json(manifest_key)
    except Exception as exc:
        warnings.append(f"S3 model manifest could not be read: {exc.__class__.__name__}.")
        return None
    if isinstance(manifest, dict):
        value = manifest.get("active_model_id")
        if value:
            return str(value)
    warnings.append("S3 model manifest does not define active_model_id.")
    return None


def _s3_runtime_status_payload(*, storage=None, env=None) -> StatusResponse:
    config = os.environ if env is None else env
    active_model_id, model_warnings = _resolve_active_model_for_api(
        env=config,
        include_missing_warning=False,
    )
    warnings = [READ_ONLY_STATE_WARNING, *model_warnings]
    try:
        station = load_settings(env=config, validate=False).target_station
    except ConfigError:
        station = _env_value(config, "TARGET_STATION") or _env_value(config, "STATION_CODE", "014501") or "014501"

    try:
        runtime_storage = storage if storage is not None else _s3_storage_from_env(config)
    except Exception as exc:
        return _placeholder_status_payload(
            active_model_id=active_model_id,
            backend_mode="s3",
            model_warnings=model_warnings,
            extra_warnings=[f"S3 runtime status is not available: {exc.__class__.__name__}."],
        )

    if not active_model_id:
        active_model_id = _s3_manifest_active_model_id(runtime_storage, config, warnings)

    runtime_status = RuntimeStatus()
    status_doc: dict[str, Any] | None = None
    status_ref = runtime_storage.runtime_path("status.json", station=station, area="artifacts")
    latest_inputs_ref = runtime_storage.runtime_path("latest_inputs.json", station=station, area="artifacts")
    live_cache_ref = runtime_storage.runtime_path("live_cache.json", station=station, area="cache")

    status_exists = _storage_exists(runtime_storage, status_ref, "Runtime status artifact", warnings)
    if status_exists:
        try:
            loaded = runtime_storage.read_json(status_ref)
            if isinstance(loaded, dict):
                status_doc = loaded
                runtime_status.status_artifact_available = True
            else:
                warnings.append("Runtime status artifact JSON is not an object.")
        except ValueError:
            warnings.append("Runtime status artifact contains invalid JSON.")
        except Exception as exc:
            warnings.append(f"Runtime status artifact could not be read: {exc.__class__.__name__}.")
    else:
        warnings.append("Runtime status artifact is not available yet.")

    runtime_status.cache_available = _storage_exists(runtime_storage, live_cache_ref, "Live cache artifact", warnings)
    runtime_status.latest_inputs_available = _storage_exists(
        runtime_storage,
        latest_inputs_ref,
        "Latest inputs artifact",
        warnings,
    )

    latest_data_date = None
    if status_doc is not None:
        latest_data_date = status_doc.get("latest_data_date")
        if not latest_data_date:
            date_range = status_doc.get("range")
            if isinstance(date_range, list) and len(date_range) >= 2:
                latest_data_date = date_range[1]
        if not active_model_id:
            active_model_id = status_doc.get("active_model_id")

    if not active_model_id:
        warnings.append("No active model is configured in env, manifest, or runtime status.")

    if active_model_id:
        try:
            backtest_ref = runtime_storage.backtest_path("summary.json", station=station, model_id=active_model_id)
            runtime_status.backtest_available = _storage_exists(
                runtime_storage,
                backtest_ref,
                "Backtest summary artifact",
                warnings,
            )
        except Exception as exc:
            warnings.append(f"Backtest summary artifact could not be checked: {exc.__class__.__name__}.")

    artifacts_ok = runtime_status.status_artifact_available and runtime_status.cache_available
    ready = bool(artifacts_ok and latest_data_date and active_model_id)
    service_status = "ok" if ready else "not_ready"

    return StatusResponse(
        ready=ready,
        service_status=service_status,
        generated_at=_utc_now_iso(),
        latest_data_date=latest_data_date,
        data_freshness_days=_freshness_days(latest_data_date),
        active_model_id=active_model_id,
        backend_mode="s3",
        artifacts_ok=artifacts_ok,
        upstream_status={},
        runtime_status=runtime_status,
        warnings=warnings,
    )


def status_payload(*, storage=None, env=None) -> StatusResponse:
    backend_mode = backend_mode_from_env(env=env)
    if backend_mode == "s" + "3":
        return _s3_runtime_status_payload(storage=storage, env=env)

    active_model_id, model_warnings = _resolve_active_model_for_api(env=env)
    return _placeholder_status_payload(
        active_model_id=active_model_id,
        backend_mode=backend_mode,
        model_warnings=model_warnings,
    )


def placeholder_forecast_payload(request: ForecastRequest, *, env=None) -> ForecastResponse:
    generated_at = _utc_now_iso()
    active_model_id, model_warnings = _resolve_active_model_for_api(env=env)
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


def create_fastapi_app(*, status_storage=None, env=None):
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
        return status_payload(storage=status_storage, env=env)

    @api.post("/forecast", response_model=ForecastResponse)
    def forecast(request: ForecastRequest):
        return placeholder_forecast_payload(request, env=env)

    return api


app = create_fastapi_app()
