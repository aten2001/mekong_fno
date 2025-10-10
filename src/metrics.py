# src/metrics.py
import numpy as np
import pandas as pd

def _to_1d(a):
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1)

def rmse(y_true, y_pred):
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("Shapes mismatch.")
    return float(np.sqrt(np.mean((yp - yt) ** 2)))

def mae(y_true, y_pred):
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("Shapes mismatch.")
    return float(np.mean(np.abs(yp - yt)))

def nse(y_true, y_pred):
    """
    Nash–Sutcliffe Efficiency
    1 - SSE / SST, If SST ≈ 0, return NaN
    """
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("Shapes mismatch.")
    denom = np.sum((yt - np.mean(yt)) ** 2)
    if denom <= 1e-12:
        return float('nan')
    return float(1.0 - np.sum((yp - yt) ** 2) / denom)

def pearson_r(y_true, y_pred):
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("Shapes mismatch.")
    if np.std(yt) < 1e-12 or np.std(yp) < 1e-12:
        return float('nan')
    R = np.corrcoef(yt, yp)
    return float(R[0, 1])

def peak_timing_error_days(dates, y_true, y_pred):
    """
    peak timing error (days): pred_peak_day - true_peak_day
    dates: a date sequence with the same length as y_true/y_pred (convertible to a DatetimeIndex)
    """
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("Shapes mismatch.")
    di = pd.to_datetime(dates)
    if di.size != yt.size:
        raise ValueError("dates length mismatch.")
    i_true = int(np.nanargmax(yt))
    i_pred = int(np.nanargmax(yp))
    d_true = pd.to_datetime(di[i_true]).normalize()
    d_pred = pd.to_datetime(di[i_pred]).normalize()
    return int((d_pred - d_true).days)
