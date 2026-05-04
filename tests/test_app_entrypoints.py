import sys
import importlib.util
from pathlib import Path


def test_gradio_entrypoint_importable_without_importing_legacy_app():
    before = "app.app" in sys.modules

    import app.gradio_app as gradio_app

    assert callable(gradio_app.build_gradio_app)
    assert callable(gradio_app.create_gradio_app)
    assert callable(gradio_app.launch_gradio_app)
    if not before:
        assert "app.app" not in sys.modules


def test_fastapi_entrypoint_importable_and_static_safe():
    import app.fastapi_app as fastapi_app

    assert callable(fastapi_app.create_fastapi_app)
    assert fastapi_app.live_payload() == {"status": "ok"}
    assert fastapi_app.ready_payload()["ready"] is True
    assert fastapi_app.status_payload().ready is False

    source = Path(fastapi_app.__file__).read_text(encoding="utf-8").lower()
    blocked = (
        "gr" + "adio",
        "tensorflow",
        "load_weights",
        "_load_service",
        "app.app",
    )
    for token in blocked:
        assert token not in source


def test_fastapi_health_routes_when_dependency_available():
    if importlib.util.find_spec("fastapi") is None:
        return

    import app.fastapi_app as fastapi_app

    api = fastapi_app.create_fastapi_app()
    routes = {route.path: route for route in api.routes}

    assert "/health/live" in routes
    assert "/health/ready" in routes
    assert "/status" in routes
    assert "/forecast" in routes
    assert routes["/health/live"].endpoint() == {"status": "ok"}
    assert routes["/health/ready"].endpoint()["ready"] is True
    assert routes["/status"].endpoint().service_status == "not_ready"


def test_legacy_gradio_entrypoint_still_exists():
    assert Path("app/app.py").exists()
