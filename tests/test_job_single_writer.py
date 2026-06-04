import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

from src.jobs.refresh_backtest import refresh_backtest_job
from src.jobs.refresh_live import refresh_live_job
from src.storage import LocalStorageBackend


def _workspace_storage():
    root = Path.cwd() / "test_job_single_writer_workspace" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    storage = LocalStorageBackend(project_root=root, runtime_root=root / "runtime")
    return root, storage


def _cleanup(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    try:
        root.parent.rmdir()
    except OSError:
        pass


def test_refresh_live_job_writes_runtime_artifacts_through_storage():
    root, storage = _workspace_storage()
    try:
        live_daily = pd.Series(
            [10.0, 11.5],
            index=pd.to_datetime(["2020-05-01", "2020-05-02"]).date,
        )

        result = refresh_live_job(
            station_code="014501",
            live_daily=live_daily,
            storage=storage,
            download_existing=False,
            active_model_id="seasonal_fno_v1",
        )

        status_path = storage.runtime_path("014501", "status.json", area="artifacts")
        latest_path = storage.runtime_path("014501", "latest_inputs.json", area="artifacts")
        cache_path = storage.runtime_path("014501", "live_cache.json", area="cache")

        assert result["ok"] is True
        assert result["written"] == [str(status_path), str(latest_path), str(cache_path)]
        assert status_path.exists()
        assert latest_path.exists()
        assert cache_path.exists()

        status = storage.read_json(status_path)
        latest = storage.read_json(latest_path)
        cache = storage.read_json(cache_path)

        assert status["writer"] == "refresh_live_job"
        assert status["station_code"] == "014501"
        assert status["active_model_id"] == "seasonal_fno_v1"
        assert latest["latest_data_date"] == result["latest_data_date"]
        assert latest["latest_value"] == 11.5
        assert cache["rows"] == 2
        assert cache["records"][-1]["h"] == 11.5
    finally:
        _cleanup(root)


def test_refresh_live_job_keeps_cumulative_local_backfill(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    existing = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-05-01", "2020-05-02", "2020-05-03"]),
            "h": [10.0, 11.0, 12.0],
        }
    )
    existing.to_parquet(out_dir / "live_backfill.parquet", index=False)

    live_daily = pd.Series(
        [13.5, 14.0, 15.0],
        index=pd.to_datetime(["2020-05-03", "2020-05-04", "2020-05-05"]).date,
    )

    result = refresh_live_job(
        out_dir=out_dir,
        station_code="014501",
        live_daily=live_daily,
        download_existing=False,
    )

    saved = pd.read_parquet(out_dir / "live_backfill.parquet")

    assert result["ok"] is True
    assert saved["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-05-01",
        "2020-05-02",
        "2020-05-03",
        "2020-05-04",
        "2020-05-05",
    ]
    assert saved["h"].tolist() == [10.0, 11.0, 13.5, 14.0, 15.0]
    assert result["range"] == ["2020-05-01", "2020-05-05"]


def test_refresh_live_job_keeps_cumulative_storage_cache():
    root, storage = _workspace_storage()
    try:
        cache_path = storage.runtime_path("014501", "live_cache.json", area="cache")
        storage.write_json(
            cache_path,
            {
                "records": [
                    {"date": "2020-05-01", "h": 10.0},
                    {"date": "2020-05-02", "h": 11.0},
                ]
            },
        )
        live_daily = pd.Series(
            [12.0],
            index=pd.to_datetime(["2020-05-03"]).date,
        )

        result = refresh_live_job(
            station_code="014501",
            live_daily=live_daily,
            storage=storage,
            download_existing=False,
        )

        cache = storage.read_json(cache_path)
        assert result["range"] == ["2020-05-01", "2020-05-03"]
        assert cache["rows"] == 3
        assert [r["date"] for r in cache["records"]] == ["2020-05-01", "2020-05-02", "2020-05-03"]
    finally:
        _cleanup(root)


def test_refresh_backtest_job_writes_summary_through_storage():
    root, storage = _workspace_storage()
    try:
        summary = {
            "available": True,
            "samples": 2,
            "rmse_model": 0.42,
            "period_start": "2026-01-01",
            "period_end": "2026-01-02",
        }

        result = refresh_backtest_job(
            storage=storage,
            station="014501",
            model_id="seasonal_fno_v1",
            active_model_id="seasonal_fno_v1",
            summary=summary,
            horizons=(1, 7, 1),
        )

        summary_path = storage.backtest_path(
            "summary.json",
            station="014501",
            model_id="seasonal_fno_v1",
        )

        assert result["ok"] is True
        assert result["horizons"] == [1, 7]
        assert result["written"] == [str(summary_path)]
        assert summary_path.exists()

        saved = storage.read_json(summary_path)
        assert saved["writer"] == "refresh_backtest_job"
        assert saved["station"] == "014501"
        assert saved["model_id"] == "seasonal_fno_v1"
        assert saved["rmse_model"] == 0.42
        assert saved["period_end"] == "2026-01-02"
    finally:
        _cleanup(root)


def test_job_only_local_simulation_produces_runtime_and_backtest_state():
    root, storage = _workspace_storage()
    try:
        live_daily = pd.Series(
            [1.0, 2.0],
            index=pd.to_datetime(["2020-05-01", "2020-05-02"]).date,
        )

        live_result = refresh_live_job(
            station_code="014501",
            live_daily=live_daily,
            storage=storage,
            download_existing=False,
        )
        backtest_result = refresh_backtest_job(
            storage=storage,
            station="014501",
            model_id="seasonal_fno_v1",
            summary={"available": False, "samples": 0},
        )

        assert live_result["written"]
        assert backtest_result["written"]
        assert storage.read_json(storage.runtime_path("014501", "status.json", area="artifacts"))["writer"] == (
            "refresh_live_job"
        )
        assert storage.read_json(
            storage.backtest_path("summary.json", station="014501", model_id="seasonal_fno_v1")
        )["writer"] == "refresh_backtest_job"
    finally:
        _cleanup(root)


def test_job_modules_do_not_import_ui_api_or_model_runtime_at_import_time():
    before = set(sys.modules)

    import src.jobs.refresh_backtest
    import src.jobs.refresh_live

    newly_imported = set(sys.modules) - before
    for module_name in ("gradio", "app.app", "tensorflow"):
        assert module_name not in newly_imported


def test_job_and_api_source_keep_single_writer_boundary():
    files = [
        Path("src/jobs/refresh_live.py"),
        Path("src/jobs/refresh_backtest.py"),
        Path("app/fastapi_app.py"),
    ]
    source_by_file = {path.as_posix(): path.read_text(encoding="utf-8") for path in files}

    assert "refresh_live_job" not in source_by_file["app/fastapi_app.py"]
    assert "refresh_backtest_job" not in source_by_file["app/fastapi_app.py"]
    assert "write_json" not in source_by_file["app/fastapi_app.py"]
    assert "_load_service" not in source_by_file["src/jobs/refresh_live.py"]
    assert "_load_service" not in source_by_file["src/jobs/refresh_backtest.py"]
    assert "gradio" not in source_by_file["src/jobs/refresh_live.py"].lower()
    assert "gradio" not in source_by_file["src/jobs/refresh_backtest.py"].lower()
    assert "app.app" not in source_by_file["src/jobs/refresh_live.py"]
    assert "app.app" not in source_by_file["src/jobs/refresh_backtest.py"]
    assert "tensorflow" not in source_by_file["src/jobs/refresh_live.py"].lower()
    assert "tensorflow" not in source_by_file["src/jobs/refresh_backtest.py"].lower()
    assert "load_weights" not in source_by_file["src/jobs/refresh_live.py"]
    assert "load_weights" not in source_by_file["src/jobs/refresh_backtest.py"]
