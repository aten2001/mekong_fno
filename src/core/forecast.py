from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.time_features import doy_no_leap, doy_sin_cos_series


DEFAULT_SEQ_LENGTH = 150
DEFAULT_PRED_LENGTH = 7


def today_utc7_date():
    """
    Return today's calendar date in UTC+07 (Asia/Bangkok), with a safe fallback.
    """
    try:
        tz = ZoneInfo("Asia/Bangkok")
        return pd.Timestamp.now(tz=tz).date()
    except Exception:
        return pd.Timestamp.utcnow().date()


def latest_contiguous_anchor(water_daily: pd.Series, need: int = DEFAULT_SEQ_LENGTH):
    """
    Find the most recent contiguous run of valid daily observations and return the anchor date.
    """
    if len(water_daily) < need:
        raise ValueError(f"Not enough data for a {need}-day window (currently {len(water_daily)} days).")

    valid_dates = {d for d, v in water_daily.items() if pd.notna(v)}
    idx = pd.to_datetime(list(water_daily.index))
    d = idx.max().normalize()

    run = 0
    while d.date() >= idx.min().date():
        if d.date() in valid_dates:
            run += 1
            if run >= need:
                return (d + pd.Timedelta(days=need - 1)).date()
        else:
            run = 0
        d -= pd.Timedelta(days=1)

    raise ValueError(f"Not enough contiguous data for {need} days.")


def norm_inputs_like_train(X, st):
    """
    Normalize input features using the same statistics as training.
    """
    Xn = X.copy()
    Xn[:, :, 0] = (Xn[:, :, 0] - st["t_mean"]) / (st["t_std"] + 1e-8)
    Xn[:, :, 2] = (Xn[:, :, 2] - st["h_in_mean"]) / (st["h_in_std"] + 1e-8)
    Xn[:, :, 3] = (Xn[:, :, 3] - st["dh_in_mean"]) / (st["dh_in_std"] + 1e-8)
    return Xn


def build_window_Xn(
    runner,
    water_daily: pd.Series,
    date_anchor: pd.Timestamp,
    *,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    pred_length: int = DEFAULT_PRED_LENGTH,
):
    """
    Construct the model input window ending at `date_anchor` and normalize it like training.
    """
    date_anchor = pd.to_datetime(date_anchor).normalize()
    L = int(seq_length)

    days = []
    d = date_anchor
    while len(days) < L:
        if not (d.month == 2 and d.day == 29):
            days.append(d)
        d -= pd.Timedelta(days=1)
    days = days[::-1]

    def _time_idx_for_date(dt: pd.Timestamp) -> int:
        base = pd.Timestamp(f"{runner.train_years[0]}-01-01")
        all_days = pd.date_range(base, dt, freq="D")
        all_days = all_days[~((all_days.month == 2) & (all_days.day == 29))]
        return len(all_days) - 1

    h_vals = []
    for dt in days:
        key = getattr(dt, "date", lambda: dt)()
        if (key not in water_daily.index) or pd.isna(water_daily[key]):
            raise ValueError(
                f"Missing water level for {dt.date()} (NaN or absent), need continuous daily series with valid values."
            )
        h_vals.append(float(water_daily[key]))
    h_vals = np.asarray(h_vals, np.float32)
    dh1 = np.concatenate([[0.0], np.diff(h_vals)]).astype(np.float32)

    t_idx = np.asarray([_time_idx_for_date(dt) for dt in days], np.float32)
    x_pos = np.zeros_like(t_idx, np.float32)
    doy_sin, doy_cos = doy_sin_cos_series(days)

    feats6 = np.stack([t_idx, x_pos, h_vals, dh1, doy_sin, doy_cos], axis=1).astype(np.float32)
    Xn = norm_inputs_like_train(feats6.copy()[None, :, :], runner.norm_stats)

    fut_dates = pd.date_range(date_anchor + pd.Timedelta(days=1), periods=pred_length, freq="D")
    fut_dates = fut_dates[~((fut_dates.month == 2) & (fut_dates.day == 29))]
    while len(fut_dates) < pred_length:
        fut_dates = fut_dates.append(fut_dates[-1:] + pd.Timedelta(days=1))
        fut_dates = fut_dates[~((fut_dates.month == 2) & (fut_dates.day == 29))]
    return Xn, fut_dates


def predict_7_abs(runner, Xn, fut_dates, training: bool = False):
    """
    Run the model forward to obtain absolute water levels for the future horizon.
    """
    y_pred_n = runner.model(Xn, training=training).numpy()
    st = runner.norm_stats
    y_pred_anom = (y_pred_n * st["h_std"] + st["h_mean"])[0, :, 0]

    doys = [doy_no_leap(pd.to_datetime(d).normalize()) for d in fut_dates]
    clim_add = np.array([float(runner.clim[d]) for d in doys], dtype=np.float32)
    return (y_pred_anom + clim_add).astype(np.float32)

