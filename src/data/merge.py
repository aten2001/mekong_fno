from __future__ import annotations

from typing import Optional

import pandas as pd

from src.backfill import series_from_any


def _to_dt_index(s: Optional[pd.Series]) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(list(s.index))
    return pd.Series(s.values, index=idx, dtype=float).sort_index()


def merge_hist_and_live_no_gaps(
    water_daily_hist: pd.Series,
    live_daily: Optional[pd.Series],
    fill_small_holes: bool = True,
) -> pd.Series:
    """
    Merge two daily series with live-over-history precedence and a full daily calendar.
    """
    wd = _to_dt_index(water_daily_hist)
    if live_daily is not None and len(live_daily) > 0:
        live = _to_dt_index(live_daily)
        wd = pd.concat([wd, live]).groupby(level=0).last().sort_index()

    if len(wd) == 0:
        return pd.Series(dtype=float)

    full = pd.date_range(wd.index.min(), wd.index.max(), freq="D")
    wd = wd.reindex(full)

    if fill_small_holes:
        wd = wd.interpolate(limit=1, limit_area="inside")

    return pd.Series(wd.values, index=wd.index.date)


def build_target_daily_series(
    water_daily_hist: pd.Series,
    backfill_daily: Optional[pd.Series],
    live_daily: Optional[pd.Series],
) -> pd.Series:
    """
    Build the target-station series from history, backfill, then live values.
    """
    backfill_daily = series_from_any(backfill_daily) if backfill_daily is not None else None
    live_daily = series_from_any(live_daily)
    wd = merge_hist_and_live_no_gaps(water_daily_hist, backfill_daily, fill_small_holes=True)
    return merge_hist_and_live_no_gaps(wd, live_daily, fill_small_holes=True)


def merge_upstream_daily_series(
    upstream_hist: Optional[pd.Series],
    live_daily: Optional[pd.Series],
) -> Optional[pd.Series]:
    """
    Merge an upstream station's historical CSV series with its live tail.
    """
    live_daily = series_from_any(live_daily)
    if upstream_hist is not None and len(upstream_hist) > 0:
        return merge_hist_and_live_no_gaps(upstream_hist, live_daily, fill_small_holes=True)
    return series_from_any(live_daily)

