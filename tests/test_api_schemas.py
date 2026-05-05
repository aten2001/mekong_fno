import importlib.util
import sys
from pathlib import Path

from pydantic import ValidationError

from app.schemas import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    StatusResponse,
)


def test_forecast_request_accepts_valid_request():
    req = ForecastRequest(station="014501", horizon=7)

    assert req.station == "014501"
    assert req.horizon == 7
    assert req.mode == "live"
    assert req.include_uncertainty is True


def test_forecast_request_rejects_horizon_above_seven():
    try:
        ForecastRequest(station="014501", horizon=8)
    except ValidationError:
        return
    raise AssertionError("ForecastRequest should reject horizon > 7")


def test_forecast_request_rejects_horizon_below_one():
    try:
        ForecastRequest(station="014501", horizon=0)
    except ValidationError:
        return
    raise AssertionError("ForecastRequest should reject horizon < 1")


def test_forecast_response_accepts_forecast_points():
    resp = ForecastResponse(
        station="014501",
        mode="live",
        horizon=2,
        generated_at="2026-05-05T00:00:00+00:00",
        predictions=[
            ForecastPoint(date="2026-05-06", y_pred=1.0),
            ForecastPoint(date="2026-05-07", y_pred=1.1, lower=0.9, upper=1.3),
        ],
    )

    assert len(resp.predictions) == 2
    assert resp.predictions[1].upper == 1.3


def test_status_response_includes_required_fields():
    resp = StatusResponse(
        ready=False,
        service_status="not_ready",
        generated_at="2026-05-05T00:00:00+00:00",
    )

    assert resp.latest_data_date is None
    assert resp.active_model_id is None
    assert resp.backend_mode == "local"
    assert resp.artifacts_ok is False
    assert resp.upstream_status == {}


def test_fastapi_payload_helpers_return_contract_shapes():
    if importlib.util.find_spec("fastapi") is None:
        return

    import app.fastapi_app as fastapi_app

    assert fastapi_app.live_payload() == {"status": "ok"}
    assert fastapi_app.ready_payload() == {"ready": True, "status": "ok"}

    status = fastapi_app.status_payload()
    assert status.ready is False
    assert status.service_status == "not_ready"
    assert status.latest_data_date is None
    assert status.active_model_id is None
    assert status.backend_mode == "local"
    assert status.artifacts_ok is False
    assert status.upstream_status == {}

    forecast = fastapi_app.placeholder_forecast_payload(
        ForecastRequest(station="014501", horizon=4)
    )
    assert forecast.station == "014501"
    assert forecast.mode == "live"
    assert forecast.horizon == 4
    assert len(forecast.predictions) == 4
    assert forecast.latest_data_date is None
    assert forecast.model_id is None
    assert forecast.warnings


def test_fastapi_app_import_is_lightweight_and_static_safe():
    if importlib.util.find_spec("fastapi") is None:
        return

    before = set(sys.modules)

    import app.fastapi_app as fastapi_app

    imported = set(sys.modules) - before
    assert "app.app" not in imported
    assert "gradio" not in imported

    source = Path(fastapi_app.__file__).read_text(encoding="utf-8").lower()
    blocked = (
        "gr" + "adio",
        "tensorflow",
        "load_weights",
        "_load_service",
        "app.app",
        "bo" + "to3",
        "s" + "3",
    )
    for token in blocked:
        assert token not in source


def test_fastapi_health_status_and_forecast_routes_when_dependency_available():
    if importlib.util.find_spec("fastapi") is None:
        return

    import app.fastapi_app as fastapi_app

    api = fastapi_app.create_fastapi_app()
    routes = {route.path: route for route in api.routes}

    assert routes["/health/live"].endpoint() == {"status": "ok"}
    assert routes["/health/ready"].endpoint() == {"ready": True, "status": "ok"}

    status = routes["/status"].endpoint()
    assert status.ready is False
    assert status.latest_data_date is None
    assert status.active_model_id is None
    assert status.backend_mode == "local"
    assert status.artifacts_ok is False
    assert status.upstream_status == {}

    forecast = routes["/forecast"].endpoint(ForecastRequest(station="014501", horizon=3))
    assert forecast.station == "014501"
    assert forecast.horizon == 3
    assert forecast.mode == "live"
    assert len(forecast.predictions) == 3
    assert forecast.latest_data_date is None
    assert forecast.model_id is None
    assert forecast.warnings

    try:
        ForecastRequest(station="014501", horizon=8)
    except ValidationError:
        return
    raise AssertionError("/forecast request schema should reject horizon > 7")
