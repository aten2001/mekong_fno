from io import BytesIO
from pathlib import Path

from src.storage import S3ObjectRef, S3StorageBackend
from src.storage.s3_storage import join_s3_key


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


def test_s3_storage_instantiates_with_fake_client_without_credentials():
    client = _FakeS3Client()
    storage = S3StorageBackend(bucket="test-bucket", prefix="mekong/v2/", client=client)

    assert storage.bucket == "test-bucket"
    assert storage.prefix == "mekong/v2"


def test_s3_key_joining_is_deterministic_and_posix():
    assert join_s3_key("/mekong/v2/", "/runtime//", Path("014501/status.json")) == (
        "mekong/v2/runtime/014501/status.json"
    )
    assert join_s3_key("", "assets", "norm_stats.json") == "assets/norm_stats.json"
    assert join_s3_key("prefix", Path("C:/tmp/model.ckpt")) == "prefix/tmp/model.ckpt"


def test_s3_storage_resolves_logical_namespaces():
    storage = S3StorageBackend(bucket="test-bucket", prefix="mekong/v2", client=_FakeS3Client())

    assert storage.resolve_model_path("weights", "model.keras", model_id="fno_v1") == S3ObjectRef(
        "test-bucket",
        "mekong/v2/models/fno_v1/weights/model.keras",
    )
    assert storage.resolve_asset_path("norm_stats.json", version="default").key == (
        "mekong/v2/assets/default/norm_stats.json"
    )
    assert storage.resolve_data_path("raw.csv").key == "mekong/v2/data/raw.csv"
    assert storage.runtime_path("status.json", station="014501", area="cache").key == (
        "mekong/v2/runtime/014501/cache/status.json"
    )
    assert storage.runtime_path("live_backfill.parquet", station="014501", area="artifacts").key == (
        "mekong/v2/runtime/014501/artifacts/live_backfill.parquet"
    )
    assert storage.backtest_path("summary.json", station="014501", model_id="fno_v1").key == (
        "mekong/v2/backtests/014501/fno_v1/summary.json"
    )
    assert storage.snapshot_path("status.json", snapshot_id="2026-05-05").key == (
        "mekong/v2/snapshots/2026-05-05/status.json"
    )


def test_s3_storage_text_and_json_roundtrip_and_exists():
    storage = S3StorageBackend(bucket="test-bucket", prefix="mekong/v2", client=_FakeS3Client())

    text_ref = storage.runtime_path("status.txt", station="014501", area="cache")
    assert storage.exists(text_ref) is False
    storage.write_text(text_ref, "ready")
    assert storage.exists(text_ref) is True
    assert storage.read_text(text_ref) == "ready"

    json_ref = storage.snapshot_path("status.json", snapshot_id="2026-05-05")
    payload = {"ok": True, "rows": [1, 2, 3]}
    storage.write_json(json_ref, payload)
    assert storage.read_json(json_ref) == payload


def test_s3_storage_accepts_raw_keys_and_matching_uris():
    storage = S3StorageBackend(bucket="test-bucket", prefix="mekong/v2", client=_FakeS3Client())

    storage.write_text("runtime/014501/status.txt", "ok")
    assert storage.read_text("mekong/v2/runtime/014501/status.txt") == "ok"
    assert storage.read_text("s3://test-bucket/mekong/v2/runtime/014501/status.txt") == "ok"


def test_s3_storage_import_has_no_heavy_app_dependencies():
    source = Path("src/storage/s3_storage.py").read_text(encoding="utf-8").lower()
    blocked = (
        "tensorflow",
        "gr" + "adio",
        "app.app",
        "_load_service",
        "event" + "bridge",
        "app" + " runner",
    )
    for token in blocked:
        assert token not in source
