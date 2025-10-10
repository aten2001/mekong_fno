# src/baselines.py
import numpy as np
from .dataio import doy_no_leap_vec

def _to_1d(a):
    a = np.asarray(a)
    return a.reshape(-1)

def predict_persistence_7th(X):
    """
    Persistence baseline: predict H=7 as the observed h on the last day of the window
    X: (N, L, 6), ch2 = h
    return: (N,) float array
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[2] < 3:
        raise ValueError("X must be (N,L,6) with h at channel=2")
    return X[:, -1, 2].astype(np.float32)

def predict_climatology_7th(tgt_dates, clim_vec):
    """
    Climatology baseline: predict H=7 as the DOY climatological mean for the target date
    tgt_dates: (N,) target dates (pd.Timestamp/str/date)

    clim_vec: length = 366; indices 1..365 are valid; index 0 is padding
    return: (N,) float array
    """
    doy = doy_no_leap_vec(tgt_dates)              # [N] in 1..365
    clim_vec = np.asarray(clim_vec, dtype=np.float32)
    if clim_vec.shape[0] != 366:
        raise ValueError("clim_vec should have length 366 with indices 1..365 valid.")
    return clim_vec[doy].astype(np.float32)

def predict_linear_7th(X, lookback=14):
    """
    Linear extrapolation baseline: fit a line to h over the last
    lookback days of the window and extrapolate to +7 days

    X: (N,L,6)，ch2 = h
    return: (N,) prediction at the target day
    """
    X = np.asarray(X, dtype=np.float32)
    N, L, C = X.shape
    if C < 3:
        raise ValueError("X must be (N,L,6) with h at channel=2")
    lb = int(max(2, min(lookback, L)))  # At least 2 points
    out = np.empty((N,), dtype=np.float32)
    # x-axis: 0..lb-1; the prediction point is at (lb-1)+7
    x = np.arange(lb, dtype=np.float32)
    x_hat = (lb - 1) + 7.0
    for i in range(N):
        h = X[i, -lb:, 2]
        # If all values are constant or NaN, use the last day (persistence)
        if not np.all(np.isfinite(h)) or np.nanstd(h) < 1e-8:
            out[i] = X[i, -1, 2]
            continue
        # First-order (linear) fit
        k, b = np.polyfit(x, h, deg=1)
        out[i] = float(k * x_hat + b)
    return out
