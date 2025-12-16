# src/baselines.py
import numpy as np
from .time_features import doy_no_leap_vec

def _to_1d(a):
    """Return a 1-D view of the input as a NumPy array."""
    a = np.asarray(a)
    return a.reshape(-1)

def predict_persistence_7th(X):
    """
    Persistence baseline for Day-7: use the last observed ``h``.

    Args:
      X (array-like): Input tensor of shape ``(N, L, 6)``; channel ``2`` must be ``h``.

    Returns:
      numpy.ndarray: Array of shape ``(N,)`` (``float32``) with the persistence prediction.

    Raises:
      ValueError: If ``X`` does not have 3 dims or ``X.shape[2] < 3``.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[2] < 3:
        raise ValueError("X must be (N,L,6) with h at channel=2")
    return X[:, -1, 2].astype(np.float32)

def predict_climatology_7th(tgt_dates, clim_vec):
    """
    Climatology baseline for Day-7 using DOY means.

    Args:
      tgt_dates: Sequence of target dates (``pd.Timestamp``/``datetime.date``/str), shape ``(N,)``.
      clim_vec (array-like): Climatology vector of length ``366`` where indices
        ``1..365`` are valid and index ``0`` is padding.

    Returns:
      numpy.ndarray: Array of shape ``(N,)`` (``float32``) with climatology-based predictions.

    Raises:
      ValueError: If ``clim_vec`` does not have length ``366``.
    """
    doy = doy_no_leap_vec(tgt_dates)              # [N] in 1..365
    clim_vec = np.asarray(clim_vec, dtype=np.float32)
    if clim_vec.shape[0] != 366:
        raise ValueError("clim_vec should have length 366 with indices 1..365 valid.")
    return clim_vec[doy].astype(np.float32)

def predict_linear_7th(X, lookback=14):
    """Linear extrapolation baseline for Day-7 over the recent history.

    Args:
      X (array-like): Input tensor of shape ``(N, L, 6)``; channel ``2`` must be ``h``.
      lookback (int): Number of trailing days used for the linear fit.
        Clamped to ``[2, L]``. Defaults to ``14``.

    Returns:
      numpy.ndarray: Array of shape ``(N,)`` (``float32``) with the linear-extrapolated predictions.

    Raises:
      ValueError: If the last dimension of ``X`` is smaller than 3 (missing ``h`` channel).
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
