import importlib.util
import json
import sys
from io import BytesIO

from src.storage import S3StorageBackend


if importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx") is not None:
    from fastapi.testclient import TestClient
else:
    TestClient = None


class _NotFound(Exception):
    response = {"Error": {"Code": "404"}}


class _AccessDenied(Exception):
    response = {"Error": {"Code": "AccessDenied"}}


class _ReadOnlyFakeS3Client:
    def __init__(self, objects=None, *, fail=None):
        self.objects = objects or {}
        self.fail = fail
        self.put_calls = 0

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.put_calls += 1
        raise AssertionError("online status must not write to S3")

    def get_object(self, Bucket, Key):
        if self.fail:
            raise self.fail()
        try:
            return {"Body": BytesIO(self.objects[(Bucket, Key)])}
        except KeyError:
            raise _NotFound()

    def head_object(self, Bucket, Key):
        if self.fail:
            raise self.fail()
        if (Bucket, Key) not in self.objects:
            raise _NotFound()
        return {"ContentLength": len(self.objects[(Bucket, Key)])}


def _body(obj):
    return json.dumps(obj).encode("utf-8")


def _storage(client):
    return S3StorageBackend(bucket="test-bucket", prefix="mekong/v2/prod", client=client)


def _env():
    return {
        "ARTIFACT_BACKEND": "s3",
        "S3_BUCKET": "test-bucket",
        "S3_PREFIX": "mekong/v2/prod",
        "TARGET_STATION": "014501",
        "ACTIVE_MODEL_ID": "seasonal_fno_v1",
    }


def _client(storage, env=None):
    if TestClient is None:
        return None

    from app.fastapi_app import create_fastapi_app

    return TestClient(create_fastapi_app(status_storage=storage, env=env or _env()))


def test_status_reads_s3_runtime_artifacts_and_reports_ready():
    client_impl = _ReadOnlyFakeS3Client(
        {
            (
                "test-bucket",
                "mekong/v2/prod/runtime/014501/artifacts/status.json",
            ): _body(
                {
                    "latest_data_date": "2026-05-10",
                    "active_model_id": "seasonal_fno_v1",
                    "backend_mode": "s3",
                    "writer": "refresh_live_job",
                }
            ),
            (
                "test-bucket",
                "mekong/v2/prod/runtime/014501/artifacts/latest_inputs.json",
            ): _body({"latest_data_date": "2026-05-10"}),
            (
                "test-bucket",
                "mekong/v2/prod/runtime/014501/cache/live_cache.json",
            ): _body({"latest_data_date": "2026-05-10", "records": []}),
        }
    )
    client = _client(_storage(client_impl))
    if client is None:
        return

    response = client.get("/status")
    data = response.json()

    assert response.status_code == 200
    assert data["ready"] is True
    assert data["service_status"] == "ok"
    assert data["latest_data_date"] == "2026-05-10"
    assert data["active_model_id"] == "seasonal_fno_v1"
    assert data["backend_mode"] == "s3"
    assert data["artifacts_ok"] is True
    assert data["runtime_status"]["status_artifact_available"] is True
    assert data["runtime_status"]["cache_available"] is True
    assert data["runtime_status"]["latest_inputs_available"] is True
    assert data["runtime_status"]["backtest_available"] is False
    assert not any("placeholder" in warning.lower() for warning in data["warnings"])
    assert client_impl.put_calls == 0


def test_status_can_resolve_active_model_from_s3_manifest_key():
    client_impl = _ReadOnlyFakeS3Client(
        {
            (
                "test-bucket",
                "mekong/v2/prod/manifests/model_manifest.json",
            ): _body({"active_model_id": "seasonal_fno_v1", "models": {}}),
            (
                "test-bucket",
                "mekong/v2/prod/runtime/014501/artifacts/status.json",
            ): _body({"latest_data_date": "2026-05-10", "backend_mode": "s3"}),
            (
                "test-bucket",
                "mekong/v2/prod/runtime/014501/cache/live_cache.json",
            ): _body({"latest_data_date": "2026-05-10", "records": []}),
        }
    )
    env = _env()
    env.pop("ACTIVE_MODEL_ID")
    env["MODEL_MANIFEST_KEY"] = "manifests/model_manifest.json"
    client = _client(_storage(client_impl), env=env)
    if client is None:
        return

    data = client.get("/status").json()

    assert data["ready"] is True
    assert data["active_model_id"] == "seasonal_fno_v1"
    assert data["backend_mode"] == "s3"
    assert client_impl.put_calls == 0


def test_status_missing_status_artifact_returns_not_ready():
    client_impl = _ReadOnlyFakeS3Client(
        {
            (
                "test-bucket",
                "mekong/v2/prod/runtime/014501/cache/live_cache.json",
            ): _body({"latest_data_date": "2026-05-10"}),
        }
    )
    client = _client(_storage(client_impl))
    if client is None:
        return

    data = client.get("/status").json()

    assert data["ready"] is False
    assert data["artifacts_ok"] is False
    assert data["runtime_status"]["status_artifact_available"] is False
    assert data["runtime_status"]["cache_available"] is True
    assert any("status artifact" in warning.lower() for warning in data["warnings"])
    assert client_impl.put_calls == 0


def test_status_missing_live_cache_does_not_report_ready():
    client_impl = _ReadOnlyFakeS3Client(
        {
            (
                "test-bucket",
                "mekong/v2/prod/runtime/014501/artifacts/status.json",
            ): _body({"latest_data_date": "2026-05-10", "active_model_id": "seasonal_fno_v1"}),
        }
    )
    client = _client(_storage(client_impl))
    if client is None:
        return

    data = client.get("/status").json()

    assert data["ready"] is False
    assert data["artifacts_ok"] is False
    assert data["runtime_status"]["status_artifact_available"] is True
    assert data["runtime_status"]["cache_available"] is False
    assert data["runtime_status"]["backtest_available"] is False
    assert client_impl.put_calls == 0


def test_status_s3_access_denied_is_graceful():
    client_impl = _ReadOnlyFakeS3Client(fail=_AccessDenied)
    client = _client(_storage(client_impl))
    if client is None:
        return

    response = client.get("/status")
    data = response.json()

    assert response.status_code == 200
    assert data["ready"] is False
    assert data["backend_mode"] == "s3"
    assert any("AccessDenied" in warning for warning in data["warnings"])
    assert client_impl.put_calls == 0


def test_status_invalid_s3_status_json_is_graceful():
    client_impl = _ReadOnlyFakeS3Client(
        {
            (
                "test-bucket",
                "mekong/v2/prod/runtime/014501/artifacts/status.json",
            ): b"{",
            (
                "test-bucket",
                "mekong/v2/prod/runtime/014501/cache/live_cache.json",
            ): _body({"latest_data_date": "2026-05-10"}),
        }
    )
    client = _client(_storage(client_impl))
    if client is None:
        return

    data = client.get("/status").json()

    assert data["ready"] is False
    assert data["runtime_status"]["status_artifact_available"] is False
    assert data["runtime_status"]["cache_available"] is True
    assert any("invalid json" in warning.lower() for warning in data["warnings"])
    assert client_impl.put_calls == 0


def test_status_runtime_import_does_not_eagerly_load_ui_or_model_runtime():
    before = set(sys.modules)

    import app.fastapi_app as fastapi_app

    imported = set(sys.modules) - before
    assert "gradio" not in imported
    assert "tensorflow" not in imported
    assert "app.app" not in imported
    assert callable(fastapi_app.create_fastapi_app)
