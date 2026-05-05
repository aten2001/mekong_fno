import importlib.util
import sys
from pathlib import Path


if importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx") is not None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
else:
    FastAPI = None
    TestClient = None


def _client():
    if TestClient is None:
        return None

    from app.fastapi_app import create_fastapi_app

    return TestClient(create_fastapi_app())


def test_create_fastapi_app_returns_fastapi_app_when_dependency_available():
    if FastAPI is None:
        return

    from app.fastapi_app import create_fastapi_app

    assert isinstance(create_fastapi_app(), FastAPI)


def test_fastapi_app_import_stays_lightweight():
    if FastAPI is None:
        return

    before = set(sys.modules)

    import app.fastapi_app as fastapi_app

    imported = set(sys.modules) - before
    assert "gradio" not in imported
    assert "app.app" not in imported
    assert callable(fastapi_app.create_fastapi_app)

    source = Path(fastapi_app.__file__).read_text(encoding="utf-8").lower()
    blocked = (
        "gr" + "adio",
        "app.app",
        "_load_service",
        "tensorflow",
        "load_weights",
        "bo" + "to3",
        "botocore",
        "event" + "bridge",
        "app" + " runner",
    )
    for token in blocked:
        assert token not in source


def test_health_live_returns_ok():
    client = _client()
    if client is None:
        return

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ready_returns_static_ready_response():
    client = _client()
    if client is None:
        return

    response = client.get("/health/ready")
    data = response.json()

    assert response.status_code == 200
    assert data["ready"] is True
    assert data["status"] == "ok"


def test_status_returns_required_placeholder_fields_without_local_paths():
    client = _client()
    if client is None:
        return

    response = client.get("/status")
    data = response.json()

    assert response.status_code == 200
    for field in [
        "ready",
        "service_status",
        "generated_at",
        "latest_data_date",
        "data_freshness_days",
        "active_model_id",
        "backend_mode",
        "artifacts_ok",
        "upstream_status",
        "runtime_status",
        "warnings",
    ]:
        assert field in data
    assert data["ready"] is False
    assert data["service_status"] == "not_ready"
    assert data["backend_mode"] == "local"
    assert data["artifacts_ok"] is False
    assert data["upstream_status"] == {}
    text = response.text.lower()
    assert "d:\\" not in text
    assert "/users/" not in text
    assert "\\users\\" not in text


def test_forecast_accepts_valid_request_and_returns_placeholder_response():
    client = _client()
    if client is None:
        return

    response = client.post(
        "/forecast",
        json={
            "station": "014501",
            "horizon": 3,
            "mode": "live",
            "include_backtest": False,
            "include_uncertainty": True,
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["station"] == "014501"
    assert data["horizon"] == 3
    assert data["mode"] == "live"
    assert data["latest_data_date"] is None
    assert data["model_id"] is None
    assert data["assist_enabled"] is False
    assert data["uncertainty_available"] is False
    assert len(data["predictions"]) == 3
    assert all(point["y_pred"] == 0.0 for point in data["predictions"])
    assert all(point["lower"] is None and point["upper"] is None for point in data["predictions"])
    assert data["warnings"]
    assert "placeholder" in data["warnings"][0].lower()


def test_forecast_rejects_invalid_horizons():
    client = _client()
    if client is None:
        return

    too_high = client.post("/forecast", json={"station": "014501", "horizon": 8})
    too_low = client.post("/forecast", json={"station": "014501", "horizon": 0})

    assert too_high.status_code == 422
    assert too_low.status_code == 422
