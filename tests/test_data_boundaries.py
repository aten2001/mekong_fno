import numpy as np
import pandas as pd

from src.data.historical import daily_from_runner_rows
from src.data.merge import build_target_daily_series, merge_hist_and_live_no_gaps, merge_upstream_daily_series
from src.data.upstream import upstream_daily_from_frame
from src.data.validation import recent_missing_dates, stable_live_daily_until


def test_merge_target_series_prefers_newer_inputs_and_fills_small_gaps():
    hist = pd.Series(
        [1.0, 3.0],
        index=pd.to_datetime(["2025-01-01", "2025-01-03"]).date,
    )
    backfill = pd.Series(
        [3.5],
        index=pd.to_datetime(["2025-01-03"]).date,
    )
    live = pd.Series(
        [4.0],
        index=pd.to_datetime(["2025-01-04"]).date,
    )

    merged = build_target_daily_series(hist, backfill, live)

    assert list(merged.index) == list(pd.date_range("2025-01-01", "2025-01-04", freq="D").date)
    assert np.isclose(merged.loc[pd.Timestamp("2025-01-02").date()], 2.25)
    assert np.isclose(merged.loc[pd.Timestamp("2025-01-03").date()], 3.5)
    assert np.isclose(merged.loc[pd.Timestamp("2025-01-04").date()], 4.0)


def test_upstream_csv_reader_aggregates_local_daily_values_and_interpolates_gap():
    df = pd.DataFrame(
        {
            "Timestamp (UTC+07:00)": [
                "2025-01-01 00:00:00",
                "2025-01-01 12:00:00",
                "2025-01-03 00:00:00",
            ],
            "Value": [10.0, 12.0, 16.0],
        }
    )
    s = upstream_daily_from_frame(df, log_label="[test]")

    assert list(s.index) == list(pd.date_range("2025-01-01", "2025-01-03", freq="D").date)
    assert np.isclose(s.loc[pd.Timestamp("2025-01-01").date()], 11.0)
    assert np.isclose(s.loc[pd.Timestamp("2025-01-02").date()], 13.5)
    assert np.isclose(s.loc[pd.Timestamp("2025-01-03").date()], 16.0)


def test_validation_helpers_keep_stable_live_values_and_report_recent_missing_dates():
    live = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]).date,
    )
    stable = stable_live_daily_until(live, pd.Timestamp("2025-01-02"))

    assert list(stable.index) == list(pd.to_datetime(["2025-01-01", "2025-01-02"]).date)
    assert stable.tolist() == [1.0, 2.0]

    sparse = pd.Series(
        [1.0, 3.0, 5.0],
        index=pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-05"]).date,
    )
    assert recent_missing_dates(sparse, days=3) == [
        pd.Timestamp("2025-01-02").date(),
        pd.Timestamp("2025-01-04").date(),
    ]


def test_historical_rows_and_upstream_merge_helpers():
    rows = [
        [0, 0.0, np.nan, 1.0, pd.Timestamp("2025-01-01 00:00:00")],
        [1, 0.0, np.nan, 3.0, pd.Timestamp("2025-01-01 12:00:00")],
        [2, 0.0, np.nan, 5.0, pd.Timestamp("2025-01-02 00:00:00")],
    ]
    daily = daily_from_runner_rows(rows)

    assert np.isclose(daily.loc[pd.Timestamp("2025-01-01").date()], 2.0)
    assert np.isclose(daily.loc[pd.Timestamp("2025-01-02").date()], 5.0)

    live = pd.Series([6.0], index=pd.to_datetime(["2025-01-03"]).date)
    merged = merge_upstream_daily_series(daily, live)
    direct = merge_hist_and_live_no_gaps(daily, live)
    assert merged.equals(direct)
