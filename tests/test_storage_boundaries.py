from pathlib import Path
from types import SimpleNamespace

from src.storage import LocalStorageBackend
import src.storage.base as storage_base
import src.storage.local_storage as local_storage


def _layout(tmp_path: Path):
    runtime = tmp_path / ".runtime"
    return SimpleNamespace(
        root=runtime,
        cache=runtime / "cache",
        artifacts=runtime / "artifacts",
    )


def test_local_storage_resolves_current_project_and_runtime_paths(tmp_path):
    layout = _layout(tmp_path)
    storage = LocalStorageBackend.from_runtime_layout(
        project_root=tmp_path,
        runtime_layout=layout,
    )

    assert storage.resolve_model_path("stung_treng_fno.ckpt") == tmp_path / "weights" / "stung_treng_fno.ckpt"
    assert storage.resolve_asset_path("norm_stats.json") == tmp_path / "assets" / "norm_stats.json"
    assert storage.resolve_data_path("Water Level.csv") == tmp_path / "data" / "Water Level.csv"
    assert storage.runtime_path("live_recent_daily.json", area="cache") == layout.cache / "live_recent_daily.json"
    assert storage.runtime_path("live_backfill.parquet", area="artifacts") == layout.artifacts / "live_backfill.parquet"
    assert storage.runtime_path("raw.txt") == layout.root / "raw.txt"


def test_local_storage_resolves_backtest_and_snapshot_paths_without_changing_layout(tmp_path):
    layout = _layout(tmp_path)
    storage = LocalStorageBackend.from_runtime_layout(
        project_root=tmp_path,
        runtime_layout=layout,
    )

    assert storage.backtest_path(
        "backtest_2025_ytd_k1.parquet",
        station="014501",
        model_id="model/v1",
    ) == layout.cache / "backtests" / "014501" / "model_v1" / "backtest_2025_ytd_k1.parquet"
    assert storage.snapshot_path("2026-05-03", "status.json") == (
        layout.artifacts / "snapshots" / "2026-05-03" / "status.json"
    )


def test_local_storage_json_and_text_read_write(tmp_path):
    storage = LocalStorageBackend(project_root=tmp_path, runtime_root=tmp_path / ".runtime")

    status_path = storage.runtime_path("status.json", area="artifacts")
    storage.write_json(status_path, {"ok": True, "latest": "2026-05-03"})
    assert storage.read_json(status_path) == {"ok": True, "latest": "2026-05-03"}

    note_path = storage.runtime_path("notes", "ready.txt", area="cache")
    storage.write_text(note_path, "ready")
    assert storage.read_text(note_path) == "ready"


def test_storage_boundary_has_no_cloud_sdk_dependency():
    source = "\n".join(
        [
            Path(storage_base.__file__).read_text(encoding="utf-8"),
            Path(local_storage.__file__).read_text(encoding="utf-8"),
        ]
    )

    blocked_modules = ("bo" + "to3", "boto" + "core")
    for module_name in blocked_modules:
        assert module_name not in source
