from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.core.backtest import backtest_ytd_1day
from src.core.forecast import today_utc7_date


DEFAULT_WET_MONTHS = (6, 7, 8, 9, 10, 11)
DEFAULT_DRY_SHRINK = 0.4


def fit_upstream_residual_model(
    df_backtest: pd.DataFrame,
    upstream_daily: pd.Series,
    k_grid=(0, 1, 2, 3, 4, 5),
    wet_months=None,
):
    """
    Fit a wet-season residual correction using upstream daily level and first difference.
    """
    if wet_months is None:
        wet_months = DEFAULT_WET_MONTHS

    df = df_backtest.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["err"] = df["h_true"] - df["h_pred"]
    df = df[df["date"].map(lambda d: d.month in set(wet_months))].copy()

    if df.empty or upstream_daily is None or len(upstream_daily) == 0:
        return None

    upstream_lvl = upstream_daily
    upstream_d1 = upstream_daily.diff()

    best = None
    for k in k_grid:
        X_list, y_list = [], []
        for d, e in zip(df["date"], df["err"]):
            lag = d - pd.Timedelta(days=k)
            if (lag in upstream_lvl.index) and (lag in upstream_d1.index):
                x1 = upstream_lvl.get(lag)
                x2 = upstream_d1.get(lag)
                if pd.notna(e) and pd.notna(x1) and pd.notna(x2):
                    X_list.append([float(x1), float(x2)])
                    y_list.append(float(e))
        if len(X_list) < 40:
            continue

        X = np.asarray(X_list, np.float64)
        y = np.asarray(y_list, np.float64)

        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sd
        Xd = np.c_[np.ones(len(Xn)), Xn]

        lam = 1e-3
        A = Xd.T @ Xd + lam * np.eye(Xd.shape[1])
        coef = np.linalg.solve(A, Xd.T @ y)

        yhat = Xd @ coef
        rmse_resid = float(np.sqrt(np.mean((y - yhat) ** 2)))

        cand = dict(
            a=float(coef[0]),
            b1=float(coef[1]),
            b2=float(coef[2]),
            mu=mu.tolist(),
            sd=sd.tolist(),
            k=int(k),
            n=len(y),
            rmse_resid=rmse_resid,
            months=list(wet_months),
        )
        if (best is None) or (rmse_resid < best["rmse_resid"]):
            best = cand

    if best:
        print(
            f"[3S-fit] k={best['k']}, n={best['n']}, rmse_resid={best['rmse_resid']:.4f}, "
            f"coef a={best['a']:.4f}, b1={best['b1']:.4f}, b2={best['b2']:.4f}"
        )
    else:
        print("[3S-fit] not enough samples to fit")
    return best


def apply_backtest_correction(df_backtest: pd.DataFrame, upstream_daily: pd.Series, params: dict):
    """
    Apply an upstream residual correction to a backtest dataframe.
    """
    if not params:
        return np.full(len(df_backtest), np.nan), np.full(len(df_backtest), np.nan)

    a = float(params["a"])
    b1 = float(params["b1"])
    b2 = float(params["b2"])
    mu = np.asarray(params["mu"], dtype=np.float64)
    sd = np.asarray(params["sd"], dtype=np.float64)
    k = int(params["k"])
    wet_months = set(params.get("months", list(DEFAULT_WET_MONTHS)))

    dates = pd.to_datetime(df_backtest["date"]).dt.date
    lag_dates = [d - pd.Timedelta(days=k) for d in dates]

    s_lvl = upstream_daily.reindex(lag_dates)
    s_lvl = s_lvl.interpolate(limit=1, limit_area="inside").astype(float)

    s_d1 = s_lvl.diff()
    if len(s_d1) > 0:
        first_valid = np.where(~pd.isna(s_d1))[0]
        if len(first_valid):
            s_d1.iloc[first_valid[0]] = 0.0

    y_corr = []
    deltas = []
    for d, hp, x1, x2 in zip(dates, df_backtest["h_pred"], s_lvl.values, s_d1.values):
        if pd.isna(hp):
            y_corr.append(np.nan)
            deltas.append(np.nan)
            continue

        if d.month not in wet_months:
            y_corr.append(float(hp))
            deltas.append(0.0)
            continue

        if pd.isna(x1) or pd.isna(x2):
            delta = 0.0
        else:
            z = (np.array([float(x1), float(x2)]) - mu) / (sd + 1e-8)
            delta = float(a + b1 * z[0] + b2 * z[1])

        y_corr.append(float(hp + delta))
        deltas.append(delta)

    return np.array(y_corr, np.float32), np.array(deltas, np.float32)


def fit_pakse_params_for_tab1(runner, water_daily, pakse_daily, horizon_for_fit=1) -> Optional[dict]:
    """
    Fit a residual-correction model using upstream Pakse daily series.
    """
    if pakse_daily is None or len(pakse_daily) == 0:
        return None
    df_fit, _ = backtest_ytd_1day(runner, water_daily, start="2025-01-01", horizon=horizon_for_fit)
    if df_fit is None or len(df_fit) == 0:
        return None
    return fit_upstream_residual_model(df_fit, pakse_daily, k_grid=(0, 1, 2, 3, 4, 5), wet_months=DEFAULT_WET_MONTHS)


def fit_w3s_params_for_tab1(runner, water_daily, w3s_daily, horizon_for_fit=1):
    """
    Fit a residual-correction model using upstream 3S daily series.
    """
    if w3s_daily is None or len(w3s_daily) == 0:
        return None
    df_fit, _ = backtest_ytd_1day(runner, water_daily, start="2025-01-01", horizon=horizon_for_fit)
    if df_fit is None or len(df_fit) == 0:
        return None
    return fit_upstream_residual_model(df_fit, w3s_daily, k_grid=(0, 1, 2, 3), wet_months=DEFAULT_WET_MONTHS)


def apply_future_correction(
    y_pred_7,
    fut_dates,
    upstream_daily,
    params,
    shrink_dry=DEFAULT_DRY_SHRINK,
    allow_interp=True,
    *,
    today=None,
):
    """
    Apply an upstream residual-correction model to a future forecast window.
    """
    n = len(fut_dates)
    if not params or upstream_daily is None or len(upstream_daily) == 0:
        return np.full(n, np.nan, np.float32), np.zeros(n, dtype=bool), 0, None

    a = float(params["a"])
    b1 = float(params["b1"])
    b2 = float(params["b2"])
    mu = np.asarray(params["mu"], dtype=np.float64)
    sd = np.asarray(params["sd"], dtype=np.float64)
    k = int(params["k"])
    wet_months = set(params.get("months", list(DEFAULT_WET_MONTHS)))

    dates = pd.to_datetime(fut_dates).normalize().date
    lag_dates = np.array([d - pd.Timedelta(days=k) for d in dates], dtype="object")

    s_lvl = upstream_daily.reindex(lag_dates)
    if allow_interp and len(s_lvl) > 0:
        s_lvl = s_lvl.interpolate(limit=1, limit_area="inside")

    today = today_utc7_date() if today is None else pd.to_datetime(today).date()
    for i, ld in enumerate(lag_dates):
        if ld is None:
            s_lvl.iloc[i] = np.nan
            continue
        d_ld = pd.to_datetime(ld)
        if pd.isna(d_ld):
            s_lvl.iloc[i] = np.nan
            continue
        if d_ld.date() > today:
            s_lvl.iloc[i] = np.nan

    s_d1 = s_lvl.diff()
    if len(s_d1) > 0:
        first_valid = np.where(~pd.isna(s_d1))[0]
        if len(first_valid):
            s_d1.iloc[first_valid[0]] = 0.0

    y_out = np.array(y_pred_7, dtype=np.float32).copy()
    used = np.zeros(n, dtype=bool)

    for i, (d, hp, x1, x2) in enumerate(zip(dates, y_pred_7, s_lvl.values, s_d1.values)):
        if pd.isna(hp) or pd.isna(x1) or pd.isna(x2):
            y_out[i] = np.nan
            continue

        z = (np.array([float(x1), float(x2)]) - mu) / (sd + 1e-8)
        delta = float(a + b1 * z[0] + b2 * z[1])
        if d.month not in wet_months:
            delta *= float(shrink_dry)
        y_out[i] = float(hp + delta)
        used[i] = True

    avail = int(used.sum())
    return y_out, used, avail, k


def upstream_raw_available_mask(dates, upstream_daily: pd.Series, k: int):
    """
    Return availability for the lag-aligned raw upstream series.
    """
    lag_dates = np.array([d - pd.Timedelta(days=int(k)) for d in dates], dtype="object")
    return np.array(
        [(ld in upstream_daily.index) and pd.notna(upstream_daily.get(ld)) for ld in lag_dates],
        dtype=bool,
    )

