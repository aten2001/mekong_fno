from __future__ import annotations

from typing import Optional

import pandas as pd

from src.backfill import series_from_any


def fetch_live_daily_series(
    *,
    station_code: str,
    cache_path: str,
    ttl_seconds: int = 900,
    error_label: Optional[str] = None,
) -> Optional[pd.Series]:
    """
    Fetch recent station daily means through the existing MRC cache helper.
    """
    from src.live_mrc import get_recent_daily_cached

    live_daily = None
    try:
        live_daily = get_recent_daily_cached(
            station_code=station_code,
            cache_path=cache_path,
            ttl_seconds=ttl_seconds,
        )
    except Exception as e:
        if error_label:
            print(f"{error_label}:", e)
    return series_from_any(live_daily)
