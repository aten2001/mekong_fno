# src/backfill.py
import os
from typing import Optional
import pandas as pd

BACKFILL_PATH = os.path.join("artifacts", "live_backfill.parquet")

def _ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

def series_from_any(obj) -> Optional[pd.Series]:
    """
    normalize DataFrame/Series/None into a Series (index = Python date, value = float);
    do not change the daily bucketing/aggregation
    allow column names ['date','h'] or ['ts','h'/'w']
    """
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        idx = pd.to_datetime(list(obj.index)).date
        s = pd.Series(obj.values, index=idx, dtype=float)
        return s.sort_index()
    if isinstance(obj, pd.DataFrame):
        cols = {c.lower(): c for c in obj.columns}
        if "date" in cols and ("h" in cols or "w" in cols):
            valcol = cols.get("h", cols.get("w"))
            idx = pd.to_datetime(obj[cols["date"]]).dt.date
            s = pd.Series(obj[valcol].astype(float).values, index=idx)
            return s.sort_index()
        if "ts" in cols and ("h" in cols or "w" in cols):
            valcol = cols.get("h", cols.get("w"))
            idx = pd.to_datetime(obj[cols["ts"]]).dt.date
            s = pd.Series(obj[valcol].astype(float).values, index=idx)
            return s.sort_index()
    return None

def read_backfill(path: str = BACKFILL_PATH) -> Optional[pd.Series]:
    """
    read the backfill Parquet file as a Series (index = Python date, value = float)
    """
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if "date" in df.columns and "h" in df.columns:
            return pd.Series(
                df["h"].astype(float).values,
                index=pd.to_datetime(df["date"]).dt.date
            ).sort_index()
        if df.shape[1] == 1:
            s = df.iloc[:, 0]
            return pd.Series(
                s.astype(float).values,
                index=pd.to_datetime(df.index).dt.date
            ).sort_index()
    except Exception:
        return None
    return None

def write_backfill(s: pd.Series, path: str = BACKFILL_PATH) -> None:
    """
    write the Series (index = Python date, value = float) back to Parquet
    """
    if s is None or len(s) == 0:
        return
    _ensure_parent_dir(path)
    df = pd.DataFrame({"date": pd.to_datetime(list(s.index)), "h": s.values})
    df.to_parquet(path, index=False)
