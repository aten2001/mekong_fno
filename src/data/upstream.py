from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def load_upstream_daily_csv(path: str, *, log_label: str = "[3S]") -> Optional[pd.Series]:
    """
    Load an upstream-station CSV and produce a daily-mean series.
    """
    if not os.path.exists(path):
        print(f"{log_label} file not found: {path}")
        return None

    df = pd.read_csv(path)
    return upstream_daily_from_frame(df, log_label=log_label)


def upstream_daily_from_frame(df: pd.DataFrame, *, log_label: str = "[3S]") -> Optional[pd.Series]:
    """
    Convert an upstream-station dataframe into a daily-mean series.
    """
    cols = {c.lower(): c for c in df.columns}

    for k in ["timestamp (utc+07:00)", "timestamp", "ts", "datetime", "date", "time"]:
        if k in cols:
            tcol = cols[k]
            break
    else:
        tcol = df.columns[0]

    for k in ["value", "h", "w", "water_level", "level"]:
        if k in cols:
            vcol = cols[k]
            break
    else:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) == 0:
            print(f"{log_label} no numeric column found")
            return None
        vcol = num_cols[0]

    raw = df[tcol].astype(str)

    has_tz_token = raw.str.contains(r"Z|[+-]\d{2}:\d{2}$", regex=True, na=False).any()
    if has_tz_token:
        ts = pd.to_datetime(raw, errors="coerce", utc=True).dt.tz_convert("Asia/Bangkok")
    else:
        ts = pd.to_datetime(raw, errors="coerce")
        ts = ts.dt.tz_localize("Asia/Bangkok")

    df = df.loc[ts.notna()].copy()
    df["_ts_local"] = ts.dropna()
    df["_date_local"] = df["_ts_local"].dt.date

    s = df.groupby("_date_local")[vcol].mean().astype(float)
    s.index = pd.Index(s.index, dtype="object")
    s = s.sort_index()

    print(
        f"{log_label} daily series ready: len={len(s)}, "
        f"range={min(s.index) if len(s) else None}->{max(s.index) if len(s) else None}"
    )

    if len(s):
        full = pd.date_range(min(s.index), max(s.index), freq="D").date
        s = s.reindex(full)
        s = s.interpolate(limit=1, limit_area="inside")
        s = s.astype(float)

    return s
