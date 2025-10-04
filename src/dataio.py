# src/dataio.py
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

def today_in_station_tz():
    """
    Return 'today' as a naive Timestamp (00:00) in the station's local timezone (UTC+07).
    """
    try:
        tz = ZoneInfo("Asia/Bangkok")
        return pd.Timestamp(pd.Timestamp.now(tz=tz).date())
    except Exception:
        return pd.Timestamp.utcnow().normalize()

def doy_no_leap(ts: pd.Timestamp) -> int:
    """
    Map any date to the same month/day in a non-leap base year and return day-of-year (1..365).
    Note: Feb 29 has been removed from train/inference pipeline.

    Args:
        ts (pandas.Timestamp): Input date/time. Naive or tz-aware; only month and day
            are used.

    Returns:
        int: Day-of-year in the inclusive range [1, 365] for the mapped date
        (using base year 2001, which is non-leap).
    """
    base_year = 2001    # pick a non-leap year
    return pd.Timestamp(base_year, ts.month, ts.day).dayofyear

def doy_no_leap_vec(ts_array) -> np.ndarray:
    """
    Vectorized version: map a sequence of timestamps to DOY (1..365) with leap day removed.

    Args:
        ts_array (array-like): Sequence of datetime-like objects (Timestamp, datetime,
            date, or ISO date strings). Timezone information, if present, is ignored.

    Returns:
        numpy.ndarray: 1-D int32 array of length N with values in [1, 365], where each
        element is the DOY of the same month/day in the non-leap base year 2001.

    Raises:
        ValueError: If any element corresponds to Feb 29 (2/29), which is invalid in
        the base year.
    """
    ts_array = pd.to_datetime(ts_array)
    base_year = 2001
    return np.array([pd.Timestamp(base_year, t.month, t.day).dayofyear for t in ts_array], dtype=np.int32)

def doy_sin_cos_series(ts_series):
    """
    Compute seasonal encodings (sin/cos of day-of-year) with leap day removed.

    Args:
        ts_series: array-like of pandas.Timestamp (or convertible)
    Returns:
        sin/cos(2π·DOY/365) arrays (float32).
    """
    ts_series = pd.to_datetime(ts_series)
    doy = np.array([doy_no_leap(pd.Timestamp(t)) for t in ts_series], dtype=np.int32)
    theta = 2.0 * np.pi * (doy - 1) / 365.0
    return np.sin(theta).astype(np.float32), np.cos(theta).astype(np.float32)
