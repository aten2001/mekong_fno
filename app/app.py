# app/app.py
import os, json, glob, io, time
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.ioff()  # turn off interactive mode to avoid accidentally launching a GUI

import gradio as gr

from zoneinfo import ZoneInfo

from src.runner import TenYearUnifiedRunner, doy_sin_cos_series
from src.model_fno import SeasonalFNO1D

from src.runner import doy_no_leap

# --- live daily means from MRC API ---
from src.live_mrc import get_recent_daily_cached
from src.backfill import BACKFILL_PATH, read_backfill, write_backfill, series_from_any

SEQ_LENGTH = 120
PRED_LENGTH = 7
ART_DIR = "artifacts"
WEIGHTS_DIR = "weights"
CSV_DIR = os.environ.get("CSV_DIR", "data")

# 3S at Sekong bridge (014500) CSV
W3S_CSV = os.path.join(CSV_DIR, "Water Level.TelemetryKH_014500_3S at Sekong bridge.csv")

# --- NEW: Pakse (013901) paths & live cache ---
PAKSE_CSV = os.path.join(CSV_DIR, "Water Level.ManualLA_013901_Pakse.csv")
PAKSE_CODE = os.environ.get("PAKSE_CODE", "013901")
LIVE_CACHE_PAKSE = os.path.join(ART_DIR, "live_recent_pakse.json")

CLIM_PATH = os.path.join(ART_DIR, "clim_vec.npy")
NORM_PATH = os.path.join(ART_DIR, "norm_stats.json")
PHASE_JSON = os.path.join(ART_DIR, "phase_report.json")
RESID_PATH = os.path.join(ART_DIR, "residual_sigma.json")  # historical residual band
# --- risk thresholds shown on YTD backtest plot ---
ALARM_LEVEL = 10.7   # Alarm level: 10.7 m
FLOOD_LEVEL = 12.0   # Flood level: 12 m

# === merge historical & live data into a continuous daily series (auto-fill “internal 1-day gaps”) ===
from typing import Optional

def _merge_hist_and_live_no_gaps(water_daily_hist: pd.Series,
                                 live_daily: Optional[pd.Series],
                                 fill_small_holes: bool = True) -> pd.Series:
    """
    Return a continuous daily water_daily series (index = Python date, value = h)
    - use live data to overwrite/extend the tail of the historical series
    - rebuild a complete daily calendar index
    - interpolate (or forward-fill) internal gaps of ≤1 day
    """
    def _to_dt_index(s: Optional[pd.Series]) -> pd.Series:
        if s is None or len(s) == 0:
            return pd.Series(dtype=float)
        idx = pd.to_datetime(list(s.index))
        return pd.Series(s.values, index=idx, dtype=float).sort_index()

    wd = _to_dt_index(water_daily_hist)
    if live_daily is not None and len(live_daily) > 0:
        live = _to_dt_index(live_daily)
        # for overlapping days, prefer live values (overwrite same-day entries)
        wd = pd.concat([wd, live]).groupby(level=0).last().sort_index()

    if len(wd) == 0:
        return pd.Series(dtype=float)

    # reindex to a full daily calendar (from earliest to latest)
    full = pd.date_range(wd.index.min(), wd.index.max(), freq="D")
    wd = wd.reindex(full)

    # fill with linear interpolation only internal small gaps;
    # leave endpoints untouched.
    if fill_small_holes:
        wd = wd.interpolate(limit=1, limit_area="inside")

    # convert the index back to Python date to stay consistent with existing code
    return pd.Series(wd.values, index=wd.index.date)

#  read the upstream-station CSV → aggregate by local (UTC+7) calendar days
#  into a daily-mean Series (index = Python date → float)
def _load_upstream_daily_csv(path: str):
    if not os.path.exists(path):
        print(f"[3S] file not found: {path}")
        return None

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    # column matching
    for k in ["timestamp (utc+07:00)", "timestamp", "ts", "datetime", "date", "time"]:
        if k in cols:
            tcol = cols[k]; break
    else:
        tcol = df.columns[0]

    for k in ["value", "h", "w", "water_level", "level"]:
        if k in cols:
            vcol = cols[k]; break
    else:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) == 0:
            print("[3S] no numeric column found")
            return None
        vcol = num_cols[0]

    raw = df[tcol].astype(str)

    # check whether timestamps include a timezone marker
    has_tz_token = raw.str.contains(r'Z|[+-]\d{2}:\d{2}$', regex=True, na=False).any()
    if has_tz_token:
        ts = pd.to_datetime(raw, errors="coerce", utc=True).dt.tz_convert("Asia/Bangkok")
    else:
        ts = pd.to_datetime(raw, errors="coerce")
        ts = ts.dt.tz_localize("Asia/Bangkok")  # already in local time

    df = df.loc[ts.notna()].copy()
    df["_ts_local"] = ts.dropna()
    # aggregate by local calendar days
    df["_date_local"] = df["_ts_local"].dt.date
    s = df.groupby("_date_local")[vcol].mean().astype(float)

    # use python date as the index
    s.index = pd.Index(s.index, dtype="object")
    s = s.sort_index()

    print(f"[3S] daily series ready: len={len(s)}, range={min(s.index) if len(s) else None}→{max(s.index) if len(s) else None}")

    # ---- interpolation across gaps up to 1 day to avoid breaks caused by small gaps ----
    if len(s):
        full = pd.date_range(min(s.index), max(s.index), freq="D").date
        s = s.reindex(full)  # generate the complete daily calendar
        s = s.interpolate(limit=1, limit_area="inside")  # fill at most 1 day for internal gaps
        s = s.astype(float)

    return s

# fit FNO residuals using only the wet season (default Jun–Oct)
# features = [3S water level, 3S first difference], with lag k (0..3)
def _fit_3s_residual_model(
    df_backtest: pd.DataFrame,
    w3s_daily: pd.Series,
    k_grid=(0, 1, 2, 3),
    wet_months=(6, 7, 8, 9, 10),
):
    df = df_backtest.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["err"] = df["h_true"] - df["h_pred"]

    # keep only wet-season samples
    df = df[df["date"].map(lambda d: d.month in set(wet_months))].copy()

    if df.empty or w3s_daily is None or len(w3s_daily) == 0:
        return None

    w3s_lvl = w3s_daily
    w3s_d1  = w3s_daily.diff()

    best = None
    for k in k_grid:
        X_list, y_list = [], []
        for d, e in zip(df["date"], df["err"]):
            lag = d - pd.Timedelta(days=k)
            if (lag in w3s_lvl.index) and (lag in w3s_d1.index):
                x1 = w3s_lvl.get(lag)
                x2 = w3s_d1.get(lag)
                if pd.notna(e) and pd.notna(x1) and pd.notna(x2):
                    X_list.append([float(x1), float(x2)])
                    y_list.append(float(e))
        if len(X_list) < 40:
            continue

        X = np.asarray(X_list, np.float64)   # [N,2] -> [level, diff]
        y = np.asarray(y_list, np.float64)   # [N]

        # standardization + bias column
        mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sd
        Xd = np.c_[np.ones(len(Xn)), Xn]     # [1, z1, z2]

        # ridge regression with small L2
        lam = 1e-3
        A = Xd.T @ Xd + lam * np.eye(Xd.shape[1])
        coef = np.linalg.solve(A, Xd.T @ y)   # [a, b1, b2]

        yhat = Xd @ coef
        rmse_resid = float(np.sqrt(np.mean((y - yhat) ** 2)))

        cand = dict(
            a=float(coef[0]),
            b1=float(coef[1]),  # coefficient for the standardized level
            b2=float(coef[2]),  # coefficient for the standardized difference
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
        print(f"[3S-fit] k={best['k']}, n={best['n']}, rmse_resid={best['rmse_resid']:.4f}, "
              f"coef a={best['a']:.4f}, b1={best['b1']:.4f}, b2={best['b2']:.4f}")
    else:
        print("[3S-fit] not enough samples to fit")
    return best

# apply 3S station correction: y_corr = h_pred + (a + b1*z1 + b2*z2)，where z* are [level, diff] after standardized
def _apply_3s_correction(df_backtest: pd.DataFrame, w3s_daily: pd.Series, params: dict):
    if not params:
        return np.full(len(df_backtest), np.nan), np.full(len(df_backtest), np.nan)

    a  = float(params["a"])
    b1 = float(params["b1"]); b2 = float(params["b2"])
    mu = np.asarray(params["mu"], dtype=np.float64)   # [2] for [level, diff]
    sd = np.asarray(params["sd"], dtype=np.float64)   # [2]
    k  = int(params["k"])
    wet_months = set(params.get("months", [6,7,8,9,10]))

    # align to the date index lagged by k days
    # allowing interpolation over gaps up to 1 day
    dates = pd.to_datetime(df_backtest["date"]).dt.date
    lag_dates = [d - pd.Timedelta(days=k) for d in dates]
    s_lvl = w3s_daily.reindex(lag_dates)
    s_lvl = s_lvl.interpolate(limit=1, limit_area="inside").astype(float)

    # take the first difference and set the first NaN to 0
    s_d1 = s_lvl.diff()
    if len(s_d1) > 0:
        first_valid = np.where(~pd.isna(s_d1))[0]
        if len(first_valid):
            s_d1.iloc[first_valid[0]] = 0.0

    y_corr = []; deltas = []
    for d, hp, x1, x2 in zip(dates, df_backtest["h_pred"], s_lvl.values, s_d1.values):
        if pd.isna(hp):
            y_corr.append(np.nan); deltas.append(np.nan); continue

        # for dry seaon do not apply correction
        if d.month not in wet_months:
            y_corr.append(float(hp)); deltas.append(0.0); continue

        # for wet season apply correction only when both 3S features are valid
        if pd.isna(x1) or pd.isna(x2):
            delta = 0.0
        else:
            z = (np.array([float(x1), float(x2)]) - mu) / (sd + 1e-8)
            delta = float(a + b1 * z[0] + b2 * z[1])

        y_corr.append(float(hp + delta))
        deltas.append(delta)

    return np.array(y_corr, np.float32), np.array(deltas, np.float32)


# === determine continuity using only “valid (non-NaN) values” and return the last day of the contiguous block as the anchor ===
def _latest_contiguous_anchor(water_daily: pd.Series, need: int = 120) -> pd.Timestamp.date:
    """
    Scan backward from the last day in water_daily to find the most recent block
    with “≥ need days” of continuity, and return its last day as the anchor;
    Count only dates with non-NaN values toward continuity;
    """
    if len(water_daily) < need:
        raise ValueError(f"Not enough data for a {need}-day window (currently {len(water_daily)} days).")

    # collect only dates with non-NaN values
    valid_dates = {d for d, v in water_daily.items() if pd.notna(v)}

    # use Timestamp to step back day by day conveniently
    idx = pd.to_datetime(list(water_daily.index))
    d = idx.max().normalize()

    run = 0
    while d.date() >= idx.min().date():
        if d.date() in valid_dates:
            run += 1
            if run >= need:
                # return the “last day of the contiguous block” as the anchor (not the start)
                return (d + pd.Timedelta(days=need - 1)).date()
        else:
            run = 0
        d -= pd.Timedelta(days=1)

    raise ValueError(f"Not enough contiguous data for {need} days.")

STATION_CODE = os.environ.get("STUNG_TRENG_CODE", "014501")
LIVE_CACHE   = os.path.join(ART_DIR, "live_recent_daily.json")

def _find_ckpt(weights_dir=WEIGHTS_DIR):
    ckpt = tf.train.latest_checkpoint(weights_dir)
    if ckpt: return ckpt
    idx = glob.glob(os.path.join(weights_dir, "*.ckpt.index"))
    if idx:  return idx[0].replace(".index","")
    raise FileNotFoundError("No TF checkpoint found in 'weights/'")

def _today_utc7_date():
    try:
        tz = ZoneInfo("Asia/Bangkok")
        return pd.Timestamp.now(tz=tz).date()
    except Exception:
        return pd.Timestamp.utcnow().date()

def _norm_inputs_like_train(X, st):
    Xn = X.copy()
    Xn[:, :, 0] = (Xn[:, :, 0] - st['t_mean']) / (st['t_std'] + 1e-8)
    Xn[:, :, 2] = (Xn[:, :, 2] - st['h_in_mean']) / (st['h_in_std'] + 1e-8)
    Xn[:, :, 3] = (Xn[:, :, 3] - st['dh_in_mean']) / (st['dh_in_std'] + 1e-8)
    return Xn

def _build_window_Xn(runner, water_daily: pd.Series, date_anchor: pd.Timestamp):
    """
    runner.predict_h_range’s window construction and normalization
     (build only Xn and the next 7 dates)
    """
    date_anchor = pd.to_datetime(date_anchor).normalize()
    L = int(SEQ_LENGTH)
    # collect L days backward from the end (skip Feb 29)
    days = []
    d = date_anchor
    while len(days) < L:
        if not (d.month == 2 and d.day == 29):
            days.append(d)
        d -= pd.Timedelta(days=1)
    days = days[::-1]  # ascending order

    # 6 channels: time_idx, x_pos(0), h, dh1, doy_sin, doy_cos
    def _time_idx_for_date(dt: pd.Timestamp) -> int:
        base = pd.Timestamp(f"{runner.train_years[0]}-01-01")
        all_days = pd.date_range(base, dt, freq='D')
        all_days = all_days[~((all_days.month==2) & (all_days.day==29))]
        return len(all_days) - 1

    # h and dh1
    h_vals = []
    for dt in days:
        key = getattr(dt, "date", lambda: dt)()
        # if a date is missing from the index or its value is NaN, treat it as missing
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
    st = runner.norm_stats
    Xn = feats6.copy()[None, :, :]
    Xn[:, :, 0] = (Xn[:, :, 0] - st['t_mean'])   / (st['t_std'] + 1e-8)
    Xn[:, :, 2] = (Xn[:, :, 2] - st['h_in_mean'])/ (st['h_in_std'] + 1e-8)
    Xn[:, :, 3] = (Xn[:, :, 3] - st['dh_in_mean'])/(st['dh_in_std'] + 1e-8)

    # Next 7 dates (remove Feb 29; pad if boundary is encountered)
    fut_dates = pd.date_range(date_anchor + pd.Timedelta(days=1), periods=PRED_LENGTH, freq='D')
    fut_dates = fut_dates[~((fut_dates.month==2) & (fut_dates.day==29))]
    while len(fut_dates) < PRED_LENGTH:
        fut_dates = fut_dates.append(fut_dates[-1:] + pd.Timedelta(days=1))
        fut_dates = fut_dates[~((fut_dates.month==2) & (fut_dates.day==29))]
    return Xn, fut_dates

def _predict_7_abs(runner, Xn, fut_dates, training=False):
    """
    forward: standardized anomaly → de-standardize → add back climatology
    (use the 7th-day DOY for all steps, consistent with training/evaluation)
    """
    y_pred_n = runner.model(Xn, training=training).numpy()  # [1,7,1]
    st = runner.norm_stats
    y_pred_anom = (y_pred_n * st['h_std'] + st['h_mean'])[0, :, 0]  # [7]

    # add the 7th-day DOY to the entire sequence (consistent with evaluation)
    tgt_day = pd.to_datetime(fut_dates[-1]).normalize()
    tgt_doy = doy_no_leap(tgt_day)
    clim_add = np.full((PRED_LENGTH,), float(runner.clim[tgt_doy]), dtype=np.float32)
    return (y_pred_anom + clim_add).astype(np.float32)

def _backtest_ytd_1day(runner, water_daily, start="2025-01-01", end=None, horizon=1):
    """
    k-day-ahead backtest from 2025-01-01 to today (or a specified end date)
    For each target day T: build a 120-day window with anchor = T - k, forecast the next 7 days,
    and compare day-k with the observation on T.
    Returns: df(date, h_true, h_pred, err), rmse
    """
    if end is None:
        end = _today_utc7_date()
    start = pd.to_datetime(start).date()
    end   = pd.to_datetime(end).date()

    k = int(horizon)
    if not (1 <= k <= PRED_LENGTH):
        raise ValueError(f"horizon must be in [1,{PRED_LENGTH}], got {horizon}")

    dates = pd.date_range(start, end, freq="D").date
    preds, trues, out_dates = [], [], []

    for T in dates:
        # skip Feb 29 (to be consistent with training/inference)
        if T.month == 2 and T.day == 29:
            continue
        anchor = pd.Timestamp(T) - pd.Timedelta(days=k)

        # skip if anchor/target missing or NaN
        if (anchor.date() not in water_daily.index) or (T not in water_daily.index):
            continue
        if pd.isna(water_daily[anchor.date()]) or pd.isna(water_daily[T]):
            continue
        try:
            Xn, fut_dates = _build_window_Xn(runner, water_daily, anchor)
        except Exception:
            continue

        y_abs = _predict_7_abs(runner, Xn, fut_dates, training=False)  # [7]
        predk = float(y_abs[k - 1])  # k-day ahead
        true  = float(water_daily[T])

        out_dates.append(pd.to_datetime(T))
        preds.append(predk)
        trues.append(true)

    df = pd.DataFrame({"date": out_dates, "h_true": trues, "h_pred": preds})
    if len(df):
        df["err"] = df["h_pred"] - df["h_true"]
        rmse = float(np.sqrt(np.mean(df["err"]**2)))
    else:
        rmse = None
    return df, rmse

def ui_eval_ytd(horizon=1):
    """
    Plot the 2025 YTD ‘Observed vs Predicted’ (k-day ahead, k=1..7) and provide an RMSE summary.
    Also plot “FNO + 3S assist” and “FNO + Pakse assist” for the same horizon when available.
    """
    S = _load_service()
    runner, water_daily = S["runner"], S["water_daily"]
    w3s_daily   = S.get("w3s_daily")
    pakse_daily = S.get("pakse_daily")

    df, rmse = _backtest_ytd_1day(runner, water_daily, start="2025-01-01", horizon=horizon)
    if df is None or len(df) == 0:
        return None, f"Not enough data to backtest 2025 YTD (h={horizon}).", pd.DataFrame()

    # ---- 3S assist on this horizon ----
    y_corr = None
    note_extra = ""
    if w3s_daily is not None and len(w3s_daily) > 0:
        params = _fit_3s_residual_model(df, w3s_daily, k_grid=(0, 1, 2, 3))
        if params:
            y_corr, deltas = _apply_3s_correction(df, w3s_daily, params)
            mask = ~np.isnan(y_corr) & ~np.isnan(df["h_true"].values)
            if mask.sum() >= 30:
                rmse_corr = float(np.sqrt(np.mean((y_corr[mask] - df["h_true"].values[mask]) ** 2)))
                note_extra += f" | 3S assist: k={params['k']}, N={int(mask.sum())}, RMSE_adj={rmse_corr:.3f} m (vs {rmse:.3f})"
            df["h_pred_3S"] = y_corr
            df["delta_3S"]  = deltas

    # ---- Pakse assist on this horizon ----
    y_corr_pk = None
    if pakse_daily is not None and len(pakse_daily) > 0:
        params_pk = _fit_3s_residual_model(df, pakse_daily, k_grid=(0, 1, 2, 3))
        if params_pk:
            y_corr_pk, deltas_pk = _apply_3s_correction(df, pakse_daily, params_pk)
            mask_pk = ~np.isnan(y_corr_pk) & ~np.isnan(df["h_true"].values)
            if mask_pk.sum() >= 30:
                rmse_pk = float(np.sqrt(np.mean((y_corr_pk[mask_pk] - df["h_true"].values[mask_pk]) ** 2)))
                note_extra += f" | Pakse assist: k={params_pk['k']}, N={int(mask_pk.sum())}, RMSE_adj={rmse_pk:.3f} m (vs {rmse:.3f})"
            df["h_pred_Pakse"] = y_corr_pk
            df["delta_Pakse"]  = deltas_pk

    # ---- plot ----
    fig = plt.figure(figsize=(10.5, 4.0))
    plt.plot(df["date"], df["h_true"], label="Observed", linewidth=1.8)
    plt.plot(df["date"], df["h_pred"], label=f"FNO ({horizon}-day ahead)", linewidth=1.8)
    if y_corr is not None:
        plt.plot(df["date"], y_corr, label=f"FNO + 3S ({horizon}-day)", linewidth=1.8)
    if y_corr_pk is not None:
        plt.plot(df["date"], y_corr_pk, label=f"FNO + Pakse ({horizon}-day)", linewidth=1.8)
    plt.axhline(ALARM_LEVEL, linestyle="--", linewidth=1, label=f"Alarm {ALARM_LEVEL:.1f} m")
    plt.axhline(FLOOD_LEVEL, linestyle="--", linewidth=1, label=f"Flood {FLOOD_LEVEL:.1f} m")
    plt.title(f"2025 YTD — Observed vs Predicted ({horizon}-day ahead)")
    plt.xlabel("Date"); plt.ylabel("Water Level (m)")
    plt.xticks(rotation=20); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

    note = (
        f"Backtest window: 2025-01-01 → {df['date'].iloc[-1].date()} "
        f"| Horizon={horizon} day(s) | N={len(df)} | RMSE={rmse:.3f} m "
        f"| Alarm={ALARM_LEVEL:.1f} m, Flood={FLOOD_LEVEL:.1f} m"
        f"{note_extra}"
    )

    cols = ["date", "h_true", "h_pred"]
    if "h_pred_3S" in df.columns:
        if "month" not in df.columns:
            df["month"] = pd.to_datetime(df["date"]).dt.month
        df["wet"] = df["month"].isin([6, 7, 8, 9, 10])
        df["changed"] = np.where(np.isfinite(df.get("delta_3S", np.nan)) & (np.abs(df.get("delta_3S", 0.0)) > 1e-6),
                                 True, False)
        cols += ["h_pred_3S", "delta_3S", "wet", "changed"]
    if "h_pred_Pakse" in df.columns:
        cols += ["h_pred_Pakse", "delta_Pakse"]

    return fig, note, df[cols]

# ----------------- run time Singleton Cache (avoid repeated loading) -----------------
_APP_CACHE = dict()

def _load_service():
    """model/runner/daily series; load only once"""
    if _APP_CACHE.get("ready"):
        return _APP_CACHE
    t0 = time.perf_counter()
    runner = TenYearUnifiedRunner(CSV_DIR, seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH)
    data = runner.load_range_data(2015, 2025, allow_missing_u=True)
    # build clim & norm
    runner.set_climatology(np.load(CLIM_PATH))
    st = json.load(open(NORM_PATH, "r", encoding="utf-8"))
    runner.norm_stats = st
    # model
    model = SeasonalFNO1D(modes=64, width=96, num_layers=4, input_features=6, dropout_rate=0.1, l2=1e-5)
    _ = model(np.zeros((1, SEQ_LENGTH, 6), dtype=np.float32), training=False)
    model.load_weights(_find_ckpt())
    runner.model = model

    # daily series: water_daily
    df = pd.DataFrame(data, columns=['time_idx','x_pos','u','h','ts'])
    df['date'] = pd.to_datetime(df['ts']).dt.date
    water_daily_hist = df.groupby('date')['h'].mean()  # pandas.Series: index=date -> value=h(m)

    # read historical backfill (Parquet)
    backfill = read_backfill(BACKFILL_PATH)

    # fetch live daily means (Series: index = Python date → value = float)
    live_daily = None
    try:
        live_daily = get_recent_daily_cached(
            station_code=STATION_CODE,
            cache_path=LIVE_CACHE,
            ttl_seconds=900,
        )
    except Exception as e:
        print("[app] live fetch skipped:", e)

    # normalize to a Series (also handle cases where upstream returns a DataFrame)
    live_daily = series_from_any(live_daily)
    if backfill is not None:
        backfill = series_from_any(backfill)

    # three-layer merge: CSV ⊕ backfill ⊕ live (later layers overwrite earlier ones)
    wd = _merge_hist_and_live_no_gaps(water_daily_hist, backfill, fill_small_holes=True)
    water_daily = _merge_hist_and_live_no_gaps(wd, live_daily, fill_small_holes=True)

    # treat live data for “today-1 and earlier” as stable and write it back to backfill
    try:
        if live_daily is not None and len(live_daily) > 0:
            cutoff = (pd.Timestamp(_today_utc7_date()) - pd.Timedelta(days=1))
            stable_idx = [d for d, v in live_daily.items() if pd.notna(v) and pd.Timestamp(d) <= cutoff]
            if len(stable_idx) > 0:
                stable = pd.Series([live_daily[d] for d in stable_idx], index=stable_idx, dtype=float).sort_index()
                if backfill is None or len(backfill) == 0:
                    new_backfill = stable
                else:
                    new_backfill = pd.concat([backfill, stable]).groupby(level=0).last().sort_index()
                write_backfill(new_backfill, BACKFILL_PATH)
    except Exception as e:
        print("[app] backfill write skipped:", e)

    print(f"[app] daily merged: days={len(water_daily)}, range={min(water_daily.index)}→{max(water_daily.index)}")

    w3s_daily = _load_upstream_daily_csv(W3S_CSV)
    if w3s_daily is not None and len(w3s_daily) > 0:
        # Keep only the valid range
        lo = pd.to_datetime("2024-04-06").date()
        hi = pd.to_datetime("2025-09-21").date()
        w3s_daily = w3s_daily.loc[lo:hi]

    # debug
    print(f"[3S] path={W3S_CSV} exists={os.path.exists(W3S_CSV)}")
    if w3s_daily is None or len(w3s_daily) == 0:
        print("[3S] empty after load/crop")
    else:
        print(f"[3S] len={len(w3s_daily)}, range={min(w3s_daily.index)}→{max(w3s_daily.index)}")

    # --- NEW: Pakse (013901) daily series: CSV ⊕ recent live ---
    pakse_hist = _load_upstream_daily_csv(PAKSE_CSV)

    live_pakse = None
    try:
        live_pakse = get_recent_daily_cached(
            station_code=PAKSE_CODE,
            cache_path=LIVE_CACHE_PAKSE,
            ttl_seconds=900,
        )
    except Exception as e:
        print("[pakse] live fetch skipped:", e)

    live_pakse = series_from_any(live_pakse)

    if pakse_hist is not None and len(pakse_hist) > 0:
        pakse_daily = _merge_hist_and_live_no_gaps(pakse_hist, live_pakse, fill_small_holes=True)
    else:
        pakse_daily = series_from_any(live_pakse)

    if pakse_daily is not None and len(pakse_daily) > 0:
        print(f"[PAKSE] len={len(pakse_daily)}, range={min(pakse_daily.index)}→{max(pakse_daily.index)}")
    else:
        print("[PAKSE] empty after load/merge")

    # try to load historical residual band
    resid_sigma = None
    if os.path.exists(RESID_PATH):
        resid_sigma = json.load(open(RESID_PATH, "r", encoding="utf-8"))

    _APP_CACHE.update(dict(
        runner=runner,
        water_daily=water_daily,
        resid_sigma=resid_sigma,  # residual band statistics
        mc_cache={},  # MC Dropout results cache
        w3s_daily=w3s_daily,
        pakse_daily=pakse_daily,
        ready=True
    ))

    # debug
    full = pd.date_range(min(water_daily.index), max(water_daily.index), freq="D")
    missing = set(full.date) - set(water_daily.index)
    tail_missing = sorted([d for d in missing if d >= (max(water_daily.index) - pd.Timedelta(days=14)).date()])
    if tail_missing:
        print(f"[app][warn] recent missing days auto-handled: {tail_missing}")

    print(f"[app] loaded in {time.perf_counter()-t0:.2f}s, days={len(water_daily)}")
    return _APP_CACHE

# ----------------- Tab 1: Today → 7 Days -----------------
def ui_predict_today(show_uncertainty=False, src_choice="Historical residuals (fast)", mc_samples=30):
    S = _load_service()
    runner, water_daily = S["runner"], S["water_daily"]
    resid_sigma = S.get("resid_sigma")
    mc_cache = S.get("mc_cache", {})

    # choose the last day of the most recent block with “≥ SEQ_LENGTH days” of continuity as the anchor (avoid tail-end gaps)
    if len(water_daily) < SEQ_LENGTH:
        return None, f"Not enough data for a {SEQ_LENGTH}-day window (currently {len(water_daily)} days).", None
    try:
        anchor = _latest_contiguous_anchor(water_daily, SEQ_LENGTH)
    except Exception as e:
        return None, f"Not enough contiguous data: {e}", None

    # build Xn and the future dates
    try:
        Xn, fut_dates = _build_window_Xn(runner, water_daily, pd.Timestamp(anchor))
    except Exception as e:
        return None, f"Failed to build input window: {e}", None

    # central (deterministic) prediction
    t0 = time.perf_counter()
    y_abs = _predict_7_abs(runner, Xn, fut_dates, training=False)  # [7]
    latency_ms = (time.perf_counter() - t0) * 1000

    # default to historical residual band
    # optional MC Dropout with caching
    lo = hi = None
    band_note = ""
    if show_uncertainty:
        if src_choice.startswith("Historical residuals"):
            if resid_sigma and "by_horizon" in resid_sigma:
                sigma = np.array(resid_sigma["by_horizon"], dtype=np.float32)  # [7]
                lo = y_abs - 1.96 * sigma
                hi = y_abs + 1.96 * sigma
                band_note = f"Historical residual band ±1.96σ (n={resid_sigma.get('n', '?')})"
            else:
                # if residual_sigma.json is missing, fall back to MC
                src_choice = "MC Dropout (slow)"
                band_note = "residual_sigma.json not found; fell back to MC Dropout."

        if src_choice.startswith("MC Dropout"):
            key = (str(anchor), int(mc_samples))
            if key in mc_cache:
                lo, hi = mc_cache[key]
                band_note = f"MC Dropout p10–90 (cache hit, N={mc_samples})"
            else:
                N = int(mc_samples)
                Ys = [_predict_7_abs(runner, Xn, fut_dates, training=True) for _ in range(N)]
                Ys = np.stack(Ys, axis=0)  # [N,7]
                lo, hi = np.percentile(Ys, [10, 90], axis=0)
                mc_cache[key] = (lo, hi)
                band_note = f"MC Dropout p10–90 (N={mc_samples})"

    # plot
    fig = plt.figure(figsize=(9.5, 4.2))
    plt.plot(fut_dates, y_abs, label="FNO (mean)", linewidth=2)
    if lo is not None:
        plt.fill_between(fut_dates, lo, hi, alpha=0.18, label=band_note or "Uncertainty band")
    plt.title("Next 7-day absolute water level (UTC+07)")
    plt.xlabel("Date"); plt.ylabel("Water Level (m)")
    plt.xticks(rotation=20); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

    # output table
    df_out = pd.DataFrame({"date": pd.to_datetime(fut_dates).date, "h_pred": y_abs})
    if lo is not None:
        df_out["p10"] = lo; df_out["p90"] = hi

    note = "Next 7-day absolute water level (UTC+07)"
    if band_note:
        note += f"; Uncertainty: {band_note}"
    return fig, note, df_out

# ----------------- Tab2：ΔRMSE Table -----------------
def _load_phase_json_or_fallback():
    """Prefer reading artifacts/phase_report.json; if missing, return None"""
    if os.path.exists(PHASE_JSON):
        with open(PHASE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def ui_phase_table(scope):
    """
    scope: 'Merged', 'Dry', 'Wet'
    Return a DataFrame showing RMSE before/after alignment and ΔRMSE (based on 2024 test_applied)
    """
    mapping = {"Merged":"all", "Dry":"dry", "Wet":"wet"}
    key = mapping.get(scope, "all")
    rep = _load_phase_json_or_fallback()
    if rep is None:
        msg = "Missing artifacts/phase_report.json. Please run: python -m scripts.make_phase_report"
        return pd.DataFrame({"message":[msg]})
    row = rep["test_applied"][key]  # {k, rmse_before, rmse_after, gain}
    df = pd.DataFrame([{
        "Window": scope,
        "k* (days)": row["k"],
        "RMSE (raw)": round(row["rmse_before"], 3),
        "RMSE (aligned)": round(row["rmse_after"], 3),
        "ΔRMSE": round(row["gain"], 3),
    }])
    return df

def ui_compare_fno_vs_3s_window(horizon=1):
    """
    Compare FNO vs FNO+3S assist on dates where 3S data are available (with lag k) for a given horizon.
    Returns a single-row DataFrame with RMSE/MAE for both methods within that window.
    """
    S = _load_service()
    runner, water_daily = S["runner"], S["water_daily"]
    w3s_daily = S.get("w3s_daily")

    df, _ = _backtest_ytd_1day(runner, water_daily, start="2025-01-01", horizon=horizon)
    if df is None or len(df) == 0:
        return pd.DataFrame({"message": [f"Not enough data (h={horizon})."]})
    if w3s_daily is None or len(w3s_daily) == 0:
        return pd.DataFrame({"message": ["3S daily series (014500) is empty."]})

    params = _fit_3s_residual_model(df, w3s_daily, k_grid=(0, 1, 2, 3))
    if not params:
        return pd.DataFrame({"message": ["Not enough samples to fit 3S assist parameters."]})

    y_corr, _ = _apply_3s_correction(df, w3s_daily, params)

    k = int(params["k"])
    dates = pd.to_datetime(df["date"]).dt.date.values
    lag_dates = np.array([d - pd.Timedelta(days=k) for d in dates], dtype="object")
    has_3s = np.array([(d in w3s_daily.index) and pd.notna(w3s_daily.get(d)) for d in lag_dates], dtype=bool)

    h_true = df["h_true"].values.astype(float)
    h_pred = df["h_pred"].values.astype(float)
    mask = has_3s & np.isfinite(h_true) & np.isfinite(h_pred) & np.isfinite(y_corr)
    if mask.sum() == 0:
        return pd.DataFrame({"message": ["No overlapping dates where 3S (with lag k) is available."]})

    idx = np.where(mask)[0]; d_sub = dates[idx]
    rmse_fno = float(np.sqrt(np.mean((h_pred[idx] - h_true[idx]) ** 2)))
    rmse_3s  = float(np.sqrt(np.mean((y_corr[idx]  - h_true[idx]) ** 2)))
    mae_fno  = float(np.mean(np.abs(h_pred[idx] - h_true[idx])))
    mae_3s   = float(np.mean(np.abs(y_corr[idx]  - h_true[idx])))

    return pd.DataFrame([{
        "Window":        f"3S-available (with lag k, h={horizon})",
        "k (days)":      k,
        "N":             int(len(idx)),
        "From":          str(d_sub.min()),
        "To":            str(d_sub.max()),
        "RMSE (FNO)":    round(rmse_fno, 3),
        "RMSE (FNO+3S)": round(rmse_3s,  3),
        "ΔRMSE":         round(rmse_3s - rmse_fno, 3),
        "MAE (FNO)":     round(mae_fno,  3),
        "MAE (FNO+3S)":  round(mae_3s,   3),
        "ΔMAE":          round(mae_3s - mae_fno, 3),
    }])

def ui_compare_fno_vs_pakse_window(horizon=1):
    """
    Compare FNO vs FNO+Pakse assist on dates where Pakse data are available (with lag k) for a given horizon.
    Returns a single-row DataFrame with RMSE/MAE for both methods within that window.
    """
    S = _load_service()
    runner, water_daily = S["runner"], S["water_daily"]
    pakse_daily = S.get("pakse_daily")

    df, _ = _backtest_ytd_1day(runner, water_daily, start="2025-01-01", horizon=horizon)
    if df is None or len(df) == 0:
        return pd.DataFrame({"message": [f"Not enough data (h={horizon})."]})
    if pakse_daily is None or len(pakse_daily) == 0:
        return pd.DataFrame({"message": ["Pakse daily series (013901) is empty."]})

    params_pk = _fit_3s_residual_model(df, pakse_daily, k_grid=(0, 1, 2, 3))
    if not params_pk:
        return pd.DataFrame({"message": ["Not enough samples to fit Pakse assist parameters."]})

    y_corr_pk, _ = _apply_3s_correction(df, pakse_daily, params_pk)

    k = int(params_pk["k"])
    dates = pd.to_datetime(df["date"]).dt.date.values
    lag_dates = np.array([d - pd.Timedelta(days=k) for d in dates], dtype="object")
    has_pk = np.array([(d in pakse_daily.index) and pd.notna(pakse_daily.get(d)) for d in lag_dates], dtype=bool)

    h_true = df["h_true"].values.astype(float)
    h_pred = df["h_pred"].values.astype(float)
    mask = has_pk & np.isfinite(h_true) & np.isfinite(h_pred) & np.isfinite(y_corr_pk)
    if mask.sum() == 0:
        return pd.DataFrame({"message": ["No overlapping dates where Pakse (with lag k) is available."]})

    idx = np.where(mask)[0]; d_sub = dates[idx]
    rmse_fno = float(np.sqrt(np.mean((h_pred[idx] - h_true[idx]) ** 2)))
    rmse_pk  = float(np.sqrt(np.mean((y_corr_pk[idx] - h_true[idx]) ** 2)))
    mae_fno  = float(np.mean(np.abs(h_pred[idx] - h_true[idx])))
    mae_pk   = float(np.mean(np.abs(y_corr_pk[idx] - h_true[idx])))

    return pd.DataFrame([{
        "Window":            f"Pakse-available (with lag k, h={horizon})",
        "k (days)":          k,
        "N":                 int(len(idx)),
        "From":              str(d_sub.min()),
        "To":                str(d_sub.max()),
        "RMSE (FNO)":        round(rmse_fno, 3),
        "RMSE (FNO+Pakse)":  round(rmse_pk,  3),
        "ΔRMSE":             round(rmse_pk - rmse_fno, 3),
        "MAE (FNO)":         round(mae_fno,  3),
        "MAE (FNO+Pakse)":   round(mae_pk,   3),
        "ΔMAE":              round(mae_pk - mae_fno, 3),
    }])

# ----------------- Gradio UI -----------------
def build_app():
    with gr.Blocks(title="Mekong FNO Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("### Mekong Water Level Forecast (Stung Treng) — FNO\n- Tab1: **Forecast Today → +7 days** (optional uncertainty)\n- Tab2: **ΔRMSE alignment evaluation** (reads `artifacts/phase_report.json`)")

        with gr.Tabs():
            with gr.Tab("Forecast (Today → +7 days)"):
                with gr.Row():
                    btn = gr.Button("Forecast +7 Days (UTC+07)", variant="primary")
                    ck = gr.Checkbox(value=False, label="Show uncertainty (Residuals/MC)")
                    src = gr.Radio(choices=["Historical residuals (fast)", "MC Dropout (slow)"],
                                   value="Historical residuals (fast)", label="Uncertainty source")
                    samp = gr.Slider(10, 100, value=30, step=5, label="MC samples", interactive=True)
                out_plot = gr.Plot()
                out_note = gr.Markdown()
                out_df   = gr.Dataframe(headers=["date","h_pred","p10","p90"], interactive=False)

                btn.click(fn=ui_predict_today, inputs=[ck, src, samp], outputs=[out_plot, out_note, out_df])

            with gr.Tab("Evaluation (2025 YTD & ΔRMSE)"):
                # --- shared horizon selector for backtest/compare ---
                with gr.Row():
                    h_sel = gr.Slider(1, 7, value=1, step=1, label="Backtest horizon (days ahead)", interactive=True)

                # --- YTD backtest (k-day ahead) ---
                with gr.Row():
                    btn_bt = gr.Button("Run 2025 YTD backtest (k-day ahead)", variant="primary")
                ytd_plot = gr.Plot()
                ytd_note = gr.Markdown()
                ytd_df = gr.Dataframe(interactive=False)

                # pass horizon to ui_eval_ytd
                bt_evt = btn_bt.click(fn=ui_eval_ytd, inputs=h_sel, outputs=[ytd_plot, ytd_note, ytd_df])

                # FNO vs FNO+3S comparison within the 3S-available window
                gr.Markdown("### FNO vs FNO + 3S (Only where 3S is available)")
                with gr.Row():
                    btn_cmp = gr.Button("Compare on 3S-available dates (RMSE & MAE)", variant="secondary")
                cmp_tbl = gr.Dataframe(interactive=False)

                # manual refresh
                btn_cmp.click(fn=ui_compare_fno_vs_3s_window, inputs=h_sel, outputs=cmp_tbl)
                # automatically refresh once after the backtest completes
                bt_evt.then(fn=ui_compare_fno_vs_3s_window, inputs=h_sel, outputs=cmp_tbl)

                # --- NEW: FNO vs FNO+Pakse comparison within the Pakse-available window ---
                gr.Markdown("### FNO vs FNO + Pakse (Only where Pakse is available)")
                with gr.Row():
                    btn_cmp_pk = gr.Button("Compare on Pakse-available dates (RMSE & MAE)", variant="secondary")
                pk_tbl = gr.Dataframe(interactive=False)

                # manual refresh
                btn_cmp_pk.click(fn=ui_compare_fno_vs_pakse_window, inputs=h_sel, outputs=pk_tbl)
                # automatically refresh once after the backtest completes
                bt_evt.then(fn=ui_compare_fno_vs_pakse_window, inputs=h_sel, outputs=pk_tbl)

                gr.Markdown("---")

                # --- ΔRMSE table ---
                scope = gr.Radio(choices=["Merged", "Dry", "Wet"], value="Merged", label="Select window")
                tbl = gr.Dataframe(interactive=False)
                scope.change(fn=ui_phase_table, inputs=scope, outputs=tbl)
                gr.Markdown(
                    "> Note: scan the optimal phase shift k* on 2023, then fix it on the corresponding 2024 windows. "
                    "Shows RMSE before/after alignment and ΔRMSE."
                )
    return demo

if __name__ == "__main__":
    # warm-up (reduce latency for load model and data for the first time)
    _load_service()
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)