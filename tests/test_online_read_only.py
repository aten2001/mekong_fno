import importlib.util
import os
import sys
from pathlib import Path


if importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx") is not None:
    from fastapi.testclient import TestClient
else:
    TestClient = None


def _client():
    if TestClient is None:
        return None

    from app.fastapi_app import create_fastapi_app

    return TestClient(create_fastapi_app())


def _with_active_model(model_id: str):
    old_active = os.environ.get("ACTIVE_MODEL_ID")
    os.environ["ACTIVE_MODEL_ID"] = model_id
    return old_active


def _restore_active_model(old_active: str | None) -> None:
    if old_active is None:
        os.environ.pop("ACTIVE_MODEL_ID", None)
    else:
        os.environ["ACTIVE_MODEL_ID"] = old_active


def test_fastapi_app_import_stays_frontend_independent():
    before = set(sys.modules)

    import app.fastapi_app as fastapi_app

    newly_imported = set(sys.modules) - before
    assert "gradio" not in newly_imported
    assert "app.app" not in newly_imported
    assert callable(fastapi_app.create_fastapi_app)


def test_online_read_only_boundary_is_explicit():
    import app.fastapi_app as fastapi_app

    assert fastapi_app.online_shared_writes_allowed() is False
    try:
        fastapi_app.assert_online_read_only_operation("test")
    except RuntimeError as exc:
        assert "cannot persist shared state" in str(exc)
        return
    raise AssertionError("online read-only guard should reject shared persistence")


def test_health_status_and_forecast_remain_available():
    client = _client()
    if client is None:
        return

    live = client.get("/health/live")
    status = client.get("/status")
    forecast = client.post("/forecast", json={"station": "014501", "horizon": 3})

    assert live.status_code == 200
    assert live.json() == {"status": "ok"}
    assert status.status_code == 200
    assert status.json()["ready"] is False
    assert forecast.status_code == 200
    data = forecast.json()
    assert data["station"] == "014501"
    assert data["horizon"] == 3
    assert len(data["predictions"]) == 3


def test_status_and_forecast_report_active_model_from_env():
    client = _client()
    if client is None:
        return

    old_active = _with_active_model("env_model_v1")
    try:
        status = client.get("/status")
        forecast = client.post("/forecast", json={"station": "014501", "horizon": 2})
    finally:
        _restore_active_model(old_active)

    assert status.status_code == 200
    assert forecast.status_code == 200
    assert status.json()["active_model_id"] == "env_model_v1"
    assert forecast.json()["model_id"] == "env_model_v1"


def test_online_responses_advertise_read_only_shared_state():
    client = _client()
    if client is None:
        return

    status = client.get("/status").json()
    forecast = client.post("/forecast", json={"station": "014501", "horizon": 1}).json()

    assert any("read-only" in warning.lower() for warning in status["warnings"])
    assert any("read-only" in warning.lower() for warning in forecast["warnings"])


def test_fastapi_source_has_no_shared_write_or_refresh_calls():
    import app.fastapi_app as fastapi_app

    source = Path(fastapi_app.__file__).read_text(encoding="utf-8")
    forbidden_fragments = (
        "write_json",
        "write_text",
        "put_object",
        "save_model_manifest",
        "refresh_live_job",
        "refresh_backtest_job",
        "publish_status",
        "update_dataset_backfill",
        "sync_artifacts",
        "live_backfill",
        "runtime_lock",
        "_load_service",
        "gradio",
        "app.app",
    )

    for fragment in forbidden_fragments:
        assert fragment not in source
