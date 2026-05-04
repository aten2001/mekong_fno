from pathlib import Path

import pandas as pd

import src.jobs
import src.jobs.refresh_backtest as refresh_backtest
import src.jobs.refresh_live as refresh_live


def test_job_package_exports_callable_entrypoints():
    assert callable(src.jobs.refresh_live_job)
    assert callable(src.jobs.refresh_backtest_job)


def test_refresh_live_job_dry_run_is_local_and_side_effect_free(tmp_path):
    result = refresh_live.refresh_live_job(out_dir=tmp_path, station_code="014501", dry_run=True)

    assert result["dry_run"] is True
    assert result["station_code"] == "014501"
    assert result["files"] == ["live_backfill.parquet", "status.json"]
    assert not (tmp_path / "live_backfill.parquet").exists()
    assert not (tmp_path / "status.json").exists()


def test_refresh_backtest_job_dry_run_declares_required_inputs(tmp_path):
    result = refresh_backtest.refresh_backtest_job(
        year=2025,
        horizons=(1, 7, 1),
        output_dir=tmp_path,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["horizons"] == [1, 7]
    assert result["requires"] == ["runner", "water_daily"]


def test_refresh_live_merge_helpers_keep_stable_rows_and_prefer_newer_values():
    existing = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-01", "2026-05-02"]),
            "h": [1.0, 2.0],
        }
    )
    live = pd.Series(
        [2.5, 3.0],
        index=pd.to_datetime(["2026-05-02", "2026-05-03"]).date,
    )

    stable = refresh_live.stable_live_rows(live, cutoff=pd.Timestamp("2026-05-02").date())
    merged = refresh_live.merge_backfill_frames(existing, stable)

    assert merged["date"].dt.date.tolist() == [
        pd.Timestamp("2026-05-01").date(),
        pd.Timestamp("2026-05-02").date(),
    ]
    assert merged["h"].tolist() == [1.0, 2.5]


def test_job_modules_avoid_ui_and_cloud_scheduler_dependencies():
    source = "\n".join(
        [
            Path(refresh_live.__file__).read_text(encoding="utf-8"),
            Path(refresh_backtest.__file__).read_text(encoding="utf-8"),
        ]
    ).lower()

    blocked = (
        "gr" + "adio",
        "bo" + "to3",
        "boto" + "core",
        "event" + "bridge",
        "app" + " runner",
        "ecs",
    )
    for token in blocked:
        assert token not in source
