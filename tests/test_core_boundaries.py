import numpy as np
import pandas as pd

from src.core.assist import apply_future_correction
from src.core.backtest import attach_persistence_backtest, rmse_against_truth
from src.core.forecast import build_window_Xn, latest_contiguous_anchor


class _DummyRunner:
    train_years = (2025, 2025)
    norm_stats = {
        "t_mean": 0.0,
        "t_std": 1.0,
        "h_in_mean": 0.0,
        "h_in_std": 1.0,
        "dh_in_mean": 0.0,
        "dh_in_std": 1.0,
        "h_mean": 0.0,
        "h_std": 1.0,
    }


def test_latest_anchor_and_window_builder_are_app_independent():
    dates = pd.date_range("2025-01-01", periods=12, freq="D").date
    water_daily = pd.Series(np.arange(12, dtype=float), index=dates)

    assert latest_contiguous_anchor(water_daily, need=5) == pd.Timestamp("2025-01-12").date()

    Xn, fut_dates = build_window_Xn(
        _DummyRunner(),
        water_daily,
        pd.Timestamp("2025-01-12"),
        seq_length=5,
        pred_length=7,
    )

    assert Xn.shape == (1, 5, 6)
    assert len(fut_dates) == 7
    assert list(pd.to_datetime(fut_dates).date)[:2] == [
        pd.Timestamp("2025-01-13").date(),
        pd.Timestamp("2025-01-14").date(),
    ]
    assert np.allclose(Xn[0, :, 2], water_daily.iloc[-5:].values)


def test_persistence_backtest_and_rmse_ignore_missing_overlap():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-03", "2025-01-04"]),
            "h_true": [3.0, 5.0],
            "h_pred": [2.5, 4.5],
        }
    )
    water_daily = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]).date,
    )

    out, rmse_pers = attach_persistence_backtest(df, water_daily, horizon=1)

    assert out["h_pred_Pers"].tolist() == [2.0, 3.0]
    assert np.isclose(rmse_pers, np.sqrt(((2.0 - 3.0) ** 2 + (3.0 - 5.0) ** 2) / 2))
    assert rmse_against_truth([1.0, np.nan], [2.0, 4.0]) == 1.0


def test_future_assist_correction_keeps_future_upstream_values_unavailable():
    fut_dates = pd.date_range("2025-06-10", periods=3, freq="D")
    y_pred = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    upstream = pd.Series(
        [5.0, 6.0, 7.0],
        index=pd.to_datetime(["2025-06-09", "2025-06-10", "2025-06-11"]).date,
    )
    params = {
        "a": 1.0,
        "b1": 0.0,
        "b2": 0.0,
        "mu": [0.0, 0.0],
        "sd": [1.0, 1.0],
        "k": 1,
        "months": [6, 7, 8, 9, 10, 11],
    }

    y_out, used, avail, k = apply_future_correction(
        y_pred,
        fut_dates,
        upstream,
        params,
        today=pd.Timestamp("2025-06-10").date(),
    )

    assert k == 1
    assert avail == 1
    assert used.tolist() == [False, True, False]
    assert np.isnan(y_out[0])
    assert np.isclose(y_out[1], 11.0)
    assert np.isnan(y_out[2])
