# tests/test_baselines_metrics.py
import numpy as np, pandas as pd
from src.baselines import predict_persistence_7th, predict_climatology_7th, predict_linear_7th
from src.metrics import rmse, mae, nse, pearson_r, peak_timing_error_days

def test_metrics_small_arrays():
    yt = np.array([0., 1.])
    yp = np.array([0., 2.])
    assert np.isclose(rmse(yt, yp), np.sqrt(0.5))
    assert np.isclose(mae(yt, yp), 0.5)
    # Pearson
    assert np.isclose(pearson_r([0,1,2],[0,2,4]), 1.0)
    assert np.isclose(pearson_r([0,1,2],[2,1,0]), -1.0)
    # NSE：using the mean as the prediction ≈ 0
    y = np.array([1., 2., 3.])
    assert np.isclose(nse(y, y), 1.0)
    assert np.isclose(nse(y, np.full_like(y, y.mean())), 0.0)
    # Peak timing error (days)
    dates = pd.date_range("2020-01-01", periods=7, freq="D")
    yt = np.array([1,2,3,5,4,3,2], dtype=float)  # Peak at index = 3 (2020-01-04)
    yp = np.array([1,2,3,4,5,6,7], dtype=float)  # Peak at index = 6 (2020-01-07)
    assert peak_timing_error_days(dates, yt, yp) == 3

def test_baselines_shapes_and_values():
    # Construct simple sample: N = 2, L = 5, ch2 = h increasing
    X = np.zeros((2,5,6), dtype=np.float32)
    X[0,:,2] = [1,2,3,4,5]
    X[1,:,2] = [2,2,2,2,2]
    # persistence
    p = predict_persistence_7th(X)
    assert p.shape == (2,)
    assert np.allclose(p, [5,2])
    # climatology
    clim = np.zeros(366, dtype=np.float32)
    clim[10] = 7.5; clim[20] = 8.0
    dates = [pd.Timestamp("2023-01-10"), pd.Timestamp("2023-01-20")]
    c = predict_climatology_7th(dates, clim)
    assert c.shape == (2,) and np.allclose(c, [7.5, 8.0])
    # linear extrapolation: the first sequence has slope ≈ 1;
    # extrapolate to +7 ⇒ last value 5 + 17 = 12
    l = predict_linear_7th(X, lookback=5)
    assert l.shape == (2,)
    assert np.isclose(l[0], 12.0, atol=1e-4)
    # the second constant sequence → falls back to the persistence baseline
    assert np.isclose(l[1], 2.0, atol=1e-4)
