from __future__ import annotations

from typing import Optional

import pandas as pd

from src.backfill import series_from_any


def stable_live_daily_until(live_daily: Optional[pd.Series], cutoff) -> Optional[pd.Series]:
    """
    Return live daily values at or before cutoff, preserving the app's stable-backfill rule.
    """
    live_daily = series_from_any(live_daily)
    if live_daily is None or len(live_daily) == 0:
        return None

    cutoff = pd.Timestamp(cutoff)
    stable_idx = [
        d for d, v in live_daily.items()
        if pd.notna(v) and pd.Timestamp(d) <= cutoff
    ]
    if len(stable_idx) == 0:
        return None

    return pd.Series(
        [live_daily[d] for d in stable_idx],
        index=pd.Index(stable_idx, dtype="object"),
        dtype=float,
    ).sort_index()


def recent_missing_dates(water_daily: pd.Series, *, days: int = 14):
    """
    Find missing dates in the recent tail of a daily series.
    """
    if water_daily is None or len(water_daily) == 0:
        return []
    full = pd.date_range(min(water_daily.index), max(water_daily.index), freq="D")
    missing = set(full.date) - set(water_daily.index)
    cutoff = (pd.Timestamp(max(water_daily.index)) - pd.Timedelta(days=int(days))).date()
    return sorted([d for d in missing if d >= cutoff])
