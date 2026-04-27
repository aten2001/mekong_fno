from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.forecast import (
    DEFAULT_PRED_LENGTH,
    build_window_Xn,
    predict_7_abs,
    today_utc7_date,
)


def backtest_ytd_1day(
    runner,
    water_daily,
    start="2025-01-01",
    end=None,
    horizon=1,
    *,
    pred_length: int = DEFAULT_PRED_LENGTH,
):
    """
    Run a k-day-ahead backtest over a date range.
    """
    if end is None:
        end = today_utc7_date()
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end).date()

    k = int(horizon)
    if not (1 <= k <= pred_length):
        raise ValueError(f"horizon must be in [1,{pred_length}], got {horizon}")

    dates = pd.date_range(start, end, freq="D").date
    preds, trues, out_dates = [], [], []

    for T in dates:
        if T.month == 2 and T.day == 29:
            continue
        anchor = pd.Timestamp(T) - pd.Timedelta(days=k)

        if (anchor.date() not in water_daily.index) or (T not in water_daily.index):
            continue
        if pd.isna(water_daily[anchor.date()]) or pd.isna(water_daily[T]):
            continue
        try:
            Xn, fut_dates = build_window_Xn(runner, water_daily, anchor, pred_length=pred_length)
        except Exception:
            continue

        y_abs = predict_7_abs(runner, Xn, fut_dates, training=False)
        predk = float(y_abs[k - 1])
        true = float(water_daily[T])

        out_dates.append(pd.to_datetime(T))
        preds.append(predk)
        trues.append(true)

    df = pd.DataFrame({"date": out_dates, "h_true": trues, "h_pred": preds})
    if len(df):
        df["err"] = df["h_pred"] - df["h_true"]
        rmse = float(np.sqrt(np.mean(df["err"] ** 2)))
    else:
        rmse = None
    return df, rmse


def attach_persistence_backtest(df_backtest: pd.DataFrame, water_daily: pd.Series, horizon: int):
    """
    Attach a k-day-ahead persistence baseline to an existing backtest dataframe.
    """
    if df_backtest is None or len(df_backtest) == 0:
        return df_backtest, None

    df = df_backtest.copy()
    k = int(horizon)

    pers_vals = []
    for d in pd.to_datetime(df["date"]).dt.date:
        anchor = pd.Timestamp(d) - pd.Timedelta(days=k)
        key = anchor.date()
        v = water_daily.get(key, np.nan)
        pers_vals.append(float(v) if pd.notna(v) else np.nan)

    df["h_pred_Pers"] = np.asarray(pers_vals, dtype=np.float32)

    mask = np.isfinite(df["h_pred_Pers"].values) & np.isfinite(df["h_true"].values)
    if mask.sum() > 0:
        rmse_pers = float(np.sqrt(np.mean((df.loc[mask, "h_pred_Pers"].values - df.loc[mask, "h_true"].values) ** 2)))
    else:
        rmse_pers = None

    return df, rmse_pers


def rmse_against_truth(y_true, y_pred):
    """
    Compute RMSE on rows where both truth and prediction are finite.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() == 0:
        return None
    return float(np.sqrt(np.mean((yp[mask] - yt[mask]) ** 2)))


def mae_against_truth(y_true, y_pred):
    """
    Compute MAE on rows where both truth and prediction are finite.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs(yp[mask] - yt[mask])))

