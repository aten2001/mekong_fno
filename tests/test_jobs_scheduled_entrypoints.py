from io import BytesIO
from pathlib import Path

import pandas as pd

import src.jobs.refresh_backtest as refresh_backtest
import src.jobs.refresh_live as refresh_live
from src.storage import S3StorageBackend


class _NotFound(Exception):
    response = {"Error": {"Code": "404"}}


class _FakeS3Client:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else str(Body).encode("utf-8")
        return {"ETag": "fake"}

    def get_object(self, Bucket, Key):
        try:
            return {"Body": BytesIO(self.objects[(Bucket, Key)])}
        except KeyError:
            raise _NotFound()

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _NotFound()
        return {"ContentLength": len(self.objects[(Bucket, Key)])}


def test_refresh_live_s3_mode_does_not_require_hf_dataset_repo():
    client = _FakeS3Client()
    storage = S3StorageBackend(bucket="test-bucket", prefix="mekong/v2/prod", client=client)
    live_daily = pd.Series(
        [12.0],
        index=pd.to_datetime(["2020-05-01"]).date,
    )

    result = refresh_live.refresh_live_from_env(
        env={
            "ARTIFACT_BACKEND": "s3",
            "S3_BUCKET": "test-bucket",
            "S3_PREFIX": "mekong/v2/prod",
            "ACTIVE_MODEL_ID": "seasonal_fno_v1",
            "STATION_CODE": "014501",
        },
        storage=storage,
        live_daily=live_daily,
    )

    assert result["ok"] is True
    assert result["backend_mode"] == "s3"
    assert result["station"] == "014501"
    assert result["station_code"] == "014501"
    assert result["active_model_id"] == "seasonal_fno_v1"
    assert result["written"] == [
        "s3://test-bucket/mekong/v2/prod/runtime/014501/artifacts/status.json",
        "s3://test-bucket/mekong/v2/prod/runtime/014501/artifacts/latest_inputs.json",
        "s3://test-bucket/mekong/v2/prod/runtime/014501/cache/live_cache.json",
    ]
    assert ("test-bucket", "mekong/v2/prod/runtime/014501/artifacts/status.json") in client.objects


def test_refresh_live_requires_hf_dataset_repo_only_when_hf_publish_enabled():
    storage = S3StorageBackend(bucket="test-bucket", prefix="mekong/v2/prod", client=_FakeS3Client())
    live_daily = pd.Series(
        [12.0],
        index=pd.to_datetime(["2020-05-01"]).date,
    )

    try:
        refresh_live.refresh_live_from_env(
            env={
                "ARTIFACT_BACKEND": "s3",
                "S3_BUCKET": "test-bucket",
                "S3_PREFIX": "mekong/v2/prod",
                "HF_PUBLISH": "1",
                "STATION_CODE": "014501",
            },
            storage=storage,
            live_daily=live_daily,
        )
    except RuntimeError as exc:
        assert "HF_DATASET_REPO is missing" in str(exc)
        return
    raise AssertionError("HF_PUBLISH=1 should require HF_DATASET_REPO")


def test_refresh_backtest_s3_dry_run_reads_backend_and_active_model_env():
    result = refresh_backtest.refresh_backtest_from_env(
        env={
            "ARTIFACT_BACKEND": "s3",
            "S3_BUCKET": "test-bucket",
            "S3_PREFIX": "mekong/v2/prod",
            "ACTIVE_MODEL_ID": "seasonal_fno_v1",
            "STATION_CODE": "014501",
        }
    )

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["backend_mode"] == "s3"
    assert result["active_model_id"] == "seasonal_fno_v1"
    assert result["model_id"] == "seasonal_fno_v1"
    assert result["station"] == "014501"
    assert result["warnings"]


def test_scheduled_job_entrypoint_modules_import_without_ui_or_model_runtime():
    source = "\n".join(
        [
            Path(refresh_live.__file__).read_text(encoding="utf-8"),
            Path(refresh_backtest.__file__).read_text(encoding="utf-8"),
        ]
    ).lower()

    blocked = (
        "gr" + "adio",
        "app.app",
        "_load_service",
        "tensorflow",
        "load_weights",
        "event" + "bridge",
        "app" + " runner",
    )
    for token in blocked:
        assert token not in source
