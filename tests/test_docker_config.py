from pathlib import Path

import importlib.util
import os


if importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx") is not None:
    from fastapi.testclient import TestClient
else:
    TestClient = None


def test_dockerfile_exists_and_runs_fastapi_app():
    dockerfile = Path("Dockerfile")

    assert dockerfile.exists()
    text = dockerfile.read_text(encoding="utf-8")

    assert "python:3.10-slim" in text
    assert "WORKDIR /app" in text
    assert "COPY requirements.txt" in text
    assert "uvicorn" in text
    assert "app.fastapi_app:app" in text
    assert "--host" in text
    assert "0.0.0.0" in text
    assert "--port" in text
    assert "8000" in text
    assert "EXPOSE 8000" in text
    assert "ARTIFACT_BACKEND=local" in text


def test_dockerfile_has_no_hardcoded_cloud_credentials():
    text = Path("Dockerfile").read_text(encoding="utf-8").lower()
    forbidden = (
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "hf_token=",
        "s3_bucket=",
    )

    for token in forbidden:
        assert token not in text


def test_dockerignore_excludes_local_runtime_and_cache_artifacts():
    dockerignore = Path(".dockerignore")

    assert dockerignore.exists()
    ignored = set(dockerignore.read_text(encoding="utf-8").splitlines())

    required = {
        ".git",
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        ".runtime",
        "data/runtime",
        "/data",
        ".env",
        "node_modules",
    }
    assert required.issubset(ignored)


def test_docker_docs_include_local_build_run_and_endpoint_checks():
    doc = Path("docs/docker.md")

    assert doc.exists()
    text = doc.read_text(encoding="utf-8")

    assert "docker build -t mekong-fno-api:local ." in text
    assert "docker run --rm -p 8000:8000" in text
    assert "ARTIFACT_BACKEND=local" in text
    assert "ACTIVE_MODEL_ID=seasonal_fno_v1" in text
    assert "MODEL_MANIFEST_PATH=/app/assets/model_manifest.json" in text
    assert "http://127.0.0.1:8000/health/live" in text
    assert "http://127.0.0.1:8000/status" in text
    assert "http://127.0.0.1:8000/docs" in text


def test_artifact_backend_env_is_reflected_in_status():
    if TestClient is None:
        return

    from app.fastapi_app import create_fastapi_app

    old_backend = os.environ.get("ARTIFACT_BACKEND")
    os.environ["ARTIFACT_BACKEND"] = "s" + "3"
    try:
        response = TestClient(create_fastapi_app()).get("/status")
    finally:
        if old_backend is None:
            os.environ.pop("ARTIFACT_BACKEND", None)
        else:
            os.environ["ARTIFACT_BACKEND"] = old_backend

    assert response.status_code == 200
    assert response.json()["backend_mode"] == "s" + "3"
