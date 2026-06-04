from __future__ import annotations

from typing import Optional

import numpy as np
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


def latest_finite_date(water_daily: pd.Series):
    """
    Return the latest date with a finite water-level value.
    """
    water_daily = series_from_any(water_daily)
    if water_daily is None or len(water_daily) == 0:
        return None

    finite = water_daily[pd.notna(water_daily) & np.isfinite(water_daily.astype(float))]
    if len(finite) == 0:
        return None
    return max(finite.index)


def input_window_missing_dates(water_daily: pd.Series, anchor, need: int):
    """
    Report missing/NaN dates in the model input window ending at ``anchor``.

    The window mirrors the forecast feature builder's no-Feb-29 behavior.
    """
    water_daily = series_from_any(water_daily)
    if water_daily is None or len(water_daily) == 0 or anchor is None:
        return []

    d = pd.to_datetime(anchor).normalize()
    days = []
    while len(days) < int(need):
        if not (d.month == 2 and d.day == 29):
            days.append(d.date())
        d -= pd.Timedelta(days=1)
    days = days[::-1]

    missing = []
    for day in days:
        if day not in water_daily.index:
            missing.append(day)
            continue
        value = water_daily.loc[day]
        if pd.isna(value) or not np.isfinite(float(value)):
            missing.append(day)
    return missing


def anchor_fallback_reason(
    water_daily: pd.Series,
    *,
    selected_anchor,
    need: int,
    stale_threshold_days: int = 3,
):
    """
    Explain why a selected forecast anchor is older than the latest finite date.
    """
    latest = latest_finite_date(water_daily)
    if latest is None or selected_anchor is None:
        return {
            "latest_finite_date": latest,
            "selected_anchor": selected_anchor,
            "stale_days": None,
            "latest_window_missing_dates": [],
            "latest_window_missing_count": 0,
            "is_stale": False,
            "reason": None,
        }

    selected = pd.to_datetime(selected_anchor).date()
    latest = pd.to_datetime(latest).date()
    stale_days = int((pd.Timestamp(latest) - pd.Timestamp(selected)).days)
    missing = input_window_missing_dates(water_daily, latest, need)
    is_stale = stale_days > int(stale_threshold_days)

    reason = None
    if is_stale:
        if missing:
            reason = (
                f"current Stung Treng input window contains {len(missing)} missing "
                "water-level values"
            )
        else:
            reason = "latest usable contiguous input window is older than the latest merged date"

    return {
        "latest_finite_date": latest,
        "selected_anchor": selected,
        "stale_days": stale_days,
        "latest_window_missing_dates": missing,
        "latest_window_missing_count": len(missing),
        "is_stale": is_stale,
        "reason": reason,
    }
