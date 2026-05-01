from __future__ import annotations

import pandas as pd


def daily_from_runner_rows(rows) -> pd.Series:
    """
    Convert TenYearUnifiedRunner rows into a target-station daily mean series.
    """
    df = pd.DataFrame(rows, columns=["time_idx", "x_pos", "u", "h", "ts"])
    df["date"] = pd.to_datetime(df["ts"]).dt.date
    return df.groupby("date")["h"].mean()


def load_runner_history_daily(
    runner,
    start_year: int = 2015,
    end_year: int = 2025,
    *,
    allow_missing_u: bool = True,
) -> pd.Series:
    """
    Load historical target-station rows through the existing runner and return daily means.
    """
    rows = runner.load_range_data(start_year, end_year, allow_missing_u=allow_missing_u)
    return daily_from_runner_rows(rows)

