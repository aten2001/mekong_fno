from pathlib import Path

from src.storage import S3KeyBuilder, S3StorageBackend
from src.storage.s3_keys import join_key, safe_token

from tests.test_s3_storage_boundaries import _FakeS3Client


def test_prefix_normalization_and_no_duplicate_slashes():
    builder = S3KeyBuilder(prefix="/mekong//v2/")

    key = builder.runtime_status_key("014501")

    assert key == "mekong/v2/runtime/014501/status.json"
    assert not key.startswith("/")
    assert "//" not in key


def test_windows_style_path_parts_are_posix_key_parts():
    builder = S3KeyBuilder(prefix="mekong/v2")

    assert builder.runtime_cache_key("014501", r"nested\live_cache.json") == (
        "mekong/v2/runtime/014501/cache/nested/live_cache.json"
    )


def test_safe_token_sanitizes_identity_tokens():
    assert safe_token("Stung Treng") == "Stung_Treng"
    assert safe_token("seasonal/fno:v1") == "seasonal_fno_v1"
    assert safe_token("default v1") == "default_v1"


def test_model_manifest_and_model_keys_are_deterministic():
    builder = S3KeyBuilder(prefix="mekong/v2")

    assert builder.model_manifest_key() == "mekong/v2/manifests/model_manifest.json"
    assert builder.model_root_key("seasonal/fno:v1") == "mekong/v2/models/seasonal_fno_v1"
    assert builder.model_manifest_key_for_model("seasonal/fno:v1") == (
        "mekong/v2/models/seasonal_fno_v1/manifest.json"
    )
    assert builder.model_weights_key("seasonal/fno:v1", "model.keras") == (
        "mekong/v2/models/seasonal_fno_v1/weights/model.keras"
    )
    assert builder.model_asset_key("seasonal/fno:v1", "norm_stats.json") == (
        "mekong/v2/models/seasonal_fno_v1/assets/norm_stats.json"
    )


def test_asset_keys_follow_assets_version_namespace():
    builder = S3KeyBuilder(prefix="/mekong/v2/")

    assert builder.asset_key("default", "norm_stats.json") == "mekong/v2/assets/default/norm_stats.json"
    assert builder.asset_key("default", Path("reports/phase_report.json")) == (
        "mekong/v2/assets/default/reports/phase_report.json"
    )


def test_runtime_keys_follow_runtime_station_namespace():
    builder = S3KeyBuilder(prefix="mekong//v2")

    assert builder.runtime_status_key("014501") == "mekong/v2/runtime/014501/status.json"
    assert builder.runtime_latest_inputs_key("014501") == "mekong/v2/runtime/014501/latest_inputs.json"
    assert builder.runtime_cache_key("014501", "live_cache.json") == (
        "mekong/v2/runtime/014501/cache/live_cache.json"
    )
    assert builder.runtime_artifact_key("014501", "live_backfill.parquet") == (
        "mekong/v2/runtime/014501/artifacts/live_backfill.parquet"
    )


def test_backtest_keys_support_station_only_and_model_specific_summaries():
    builder = S3KeyBuilder()

    assert builder.backtest_summary_key("Stung Treng") == "backtests/Stung_Treng/summary.json"
    assert builder.backtest_summary_key("Stung Treng", model_id="seasonal/fno:v1") == (
        "backtests/Stung_Treng/seasonal_fno_v1/summary.json"
    )
    assert builder.backtest_artifact_key("Stung Treng", "horizon_metrics.parquet", model_id="seasonal/fno:v1") == (
        "backtests/Stung_Treng/seasonal_fno_v1/horizon_metrics.parquet"
    )


def test_snapshot_keys_follow_snapshot_namespace():
    builder = S3KeyBuilder(prefix="mekong/v2")

    assert builder.snapshot_key("2026-05-05", "merged_daily.parquet") == (
        "mekong/v2/snapshots/2026-05-05/merged_daily.parquet"
    )
    assert builder.snapshot_status_key("2026-05-05") == "mekong/v2/snapshots/2026-05-05/status.json"
    assert builder.snapshot_runtime_status_key("2026-05-05", "014501") == (
        "mekong/v2/snapshots/2026-05-05/runtime/014501/status.json"
    )
    assert builder.snapshot_backtest_summary_key("2026-05-05", "014501", model_id="seasonal/fno:v1") == (
        "mekong/v2/snapshots/2026-05-05/backtests/014501/seasonal_fno_v1/summary.json"
    )


def test_path_traversal_is_rejected():
    builder = S3KeyBuilder(prefix="mekong/v2")

    try:
        builder.runtime_cache_key("014501", "../status.json")
    except ValueError:
        return
    raise AssertionError("path traversal with '..' should be rejected")


def test_absolute_local_paths_are_normalized_without_drive_or_root_leakage():
    key = join_key("prefix", Path("C:/tmp/model.ckpt"))

    assert key == "prefix/tmp/model.ckpt"
    assert "C:" not in key
    assert not key.startswith("/")


def test_s3_storage_backend_uses_key_builder_for_namespaces():
    storage = S3StorageBackend(bucket="test-bucket", prefix="/mekong//v2/", client=_FakeS3Client())

    assert isinstance(storage.key_builder, S3KeyBuilder)
    assert storage.resolve_asset_path("norm_stats.json", version="default").key == (
        "mekong/v2/assets/default/norm_stats.json"
    )
    assert storage.runtime_path("status.json", station="Stung Treng", area="cache").key == (
        "mekong/v2/runtime/Stung_Treng/cache/status.json"
    )
    assert storage.backtest_path("summary.json", station="Stung Treng", model_id="seasonal/fno:v1").key == (
        "mekong/v2/backtests/Stung_Treng/seasonal_fno_v1/summary.json"
    )
