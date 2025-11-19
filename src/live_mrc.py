# src/live_mrc.py
import os
import json
import time
from typing import Optional
import requests
import pandas as pd

DEFAULT_BASE = "https://api.mrcmekong.org/api/v1/time-series/telemetry/recent/measurement"
DEFAULT_UA = "mekong-fno-app/1.0 (+https://example.org)"

def fetch_recent_measurements(
    station_code: str = "014501",
    base_url: str = DEFAULT_BASE,
    timeout: float = 10.0,
    headers: Optional[dict] = None,
) -> pd.DataFrame:
    """Fetch recent telemetry measurements from the MRC API.

    Args:
      station_code: MRC station code (e.g., ``"014501"``).
      base_url: Base endpoint for the recent measurement API.
      timeout: Request timeout in seconds.
      headers: Optional extra HTTP headers to merge into the default headers.

    Returns:
      pandas.DataFrame: Columns:
        - ``ts_utc`` (datetime64[ns, UTC]): Telemetry timestamp (UTC).
        - ``w`` (float): Water level (meters). Non-numeric values coerced to NaN.
        - ``r`` (float): Rainfall (mm). Non-numeric values coerced to NaN.

    Raises:
      requests.HTTPError: If the HTTP response status is an error.
      requests.RequestException: For network/connection issues.
    """
    url = f"{base_url}/{station_code}"
    hdr = {"Accept": "application/json", "User-Agent": DEFAULT_UA}
    if headers:
        hdr.update(headers)
    r = requests.get(url, timeout=timeout, headers=hdr)
    r.raise_for_status()
    js = r.json()
    meas = js.get("measurements", [])
    if not meas:
        return pd.DataFrame(columns=["ts_utc", "w", "r"])
    df = pd.DataFrame(meas)
    # columns: d (ISO8601 UTC), w (water level, m), r (rain, mm)
    df["ts_utc"] = pd.to_datetime(df["d"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc"])
    df = df.sort_values("ts_utc").reset_index(drop=True)
    df["w"] = pd.to_numeric(df.get("w", None), errors="coerce")
    df["r"] = pd.to_numeric(df.get("r", None), errors="coerce")
    return df[["ts_utc", "w", "r"]]

def recent_to_daily_mean(
    df_recent: pd.DataFrame,
    tz: str = "Asia/Bangkok",
    min_samples_per_day: int = 24,
) -> pd.Series:
    """
    Aggregate recent UTC observations into local-day daily means.

    Args:
      df_recent: DataFrame from :func:`fetch_recent_measurements` containing
        at least columns ``ts_utc`` (tz-aware UTC) and ``w`` (float).
      tz: IANA timezone string for local-day aggregation.
      min_samples_per_day: Minimum number of samples required to keep a day.

    Returns:
      pandas.Series: Daily mean water levels indexed by **python date** (not
      Timestamp). Series is sorted ascending; dtype is ``float``. Empty series
      if input is empty or no day meets the minimum sample count.
    """
    if df_recent is None or df_recent.empty:
        return pd.Series(dtype=float)
    ts_local = df_recent["ts_utc"].dt.tz_convert(tz)
    dates = ts_local.dt.date
    grp = df_recent.assign(date=dates).groupby("date")["w"]
    means = grp.mean()
    counts = grp.count()
    valid = counts[counts >= int(min_samples_per_day)].index
    means = means.loc[valid].astype(float)
    means.index = pd.Index(means.index, dtype="object")  # ensure python date objects
    return means

def merge_into_water_daily(
    water_daily: pd.Series,
    recent_daily: pd.Series,
) -> pd.Series:
    """
    Merge recent daily means into an existing daily series.

    Args:
      water_daily: Historical baseline Series (index: python ``date`` → float).
      recent_daily: Recent daily means to merge (index: python ``date`` → float).

    Returns:
      pandas.Series: Merged, sorted daily series (index: python ``date`` → float).
      If ``recent_daily`` is empty/None, returns ``water_daily`` unchanged.

    Raises:
      ValueError: If either input cannot be cast to ``float`` values.
    """
    if recent_daily is None or recent_daily.empty:
        return water_daily
    s = pd.concat([water_daily.astype(float), recent_daily.astype(float)])
    s = s.groupby(level=0).last().sort_index()
    return s

# -------- tiny file cache to avoid hammering the API --------
def get_recent_daily_cached(
    station_code: str = "014501",
    cache_path: str = "artifacts/live_recent_daily.json",
    ttl_seconds: int = 900,
    base_url: str = DEFAULT_BASE,
) -> pd.Series:
    """
    Return recent daily means using a small on-disk cache.

    Args:
      station_code: MRC station code to query.
      cache_path: JSON cache file path (created if missing).
      ttl_seconds: Cache time-to-live in seconds.
      base_url: Base endpoint for the MRC recent measurement API.

    Returns:
      pandas.Series: Daily mean water levels (index: python ``date`` → float).
      Empty series if the API returns no usable data.
    """
    now = time.time()
    # read cache
    if os.path.exists(cache_path):
        try:
            obj = json.load(open(cache_path, "r", encoding="utf-8"))
            if now - float(obj.get("fetched_at", 0)) <= float(ttl_seconds):
                data = obj.get("daily", {})
                if data:
                    ser = pd.Series({pd.to_datetime(k).date(): float(v) for k, v in data.items()})
                    return ser
        except Exception:
            pass

    # fetch live and write cache
    df = fetch_recent_measurements(station_code=station_code, base_url=base_url)
    daily = recent_to_daily_mean(df)
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        json.dump(
            {
                "station": station_code,
                "fetched_at": now,
                "daily": {str(k): float(v) for k, v in daily.items()},
            },
            open(cache_path, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )
    except Exception:
        pass
    return daily
