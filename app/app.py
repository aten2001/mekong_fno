# app/app.py
import os, json, glob, io, time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import gradio as gr
from zoneinfo import ZoneInfo

from src.runner import TenYearUnifiedRunner, doy_sin_cos_series
from src.model_fno import SeasonalFNO1D

from src.runner import doy_no_leap

# --- live daily means from MRC API ---
from src.live_mrc import get_recent_daily_cached, merge_into_water_daily

SEQ_LENGTH = 120
PRED_LENGTH = 7
ART_DIR = "artifacts"
WEIGHTS_DIR = "weights"
CSV_DIR = os.environ.get("CSV_DIR", "data")
CLIM_PATH = os.path.join(ART_DIR, "clim_vec.npy")
NORM_PATH = os.path.join(ART_DIR, "norm_stats.json")
PHASE_JSON = os.path.join(ART_DIR, "phase_report.json")
RESID_PATH = os.path.join(ART_DIR, "residual_sigma.json")  # day5: historical residual band
# --- risk thresholds shown on YTD backtest plot ---
ALARM_LEVEL = 10.7   # Alarm level: 10.7 m
FLOOD_LEVEL = 12.0   # Flood level: 12 m

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
        if key not in water_daily.index:
            raise ValueError(f"Missing water level for {dt.date()}, need continuous daily series.")
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

def _backtest_ytd_1day(runner, water_daily, start="2025-01-01", end=None):
    """
    1-day-ahead backtest from 2025-01-01 to today (or a specified end date)
    For each target day T: build a 120-day window with anchor = T-1, forecast the next 7 days,
    and compare day-1 with the observation.
    Returns: df(date, h_true, h_pred, err), rmse
    """
    if end is None:
        end = _today_utc7_date()
    start = pd.to_datetime(start).date()
    end   = pd.to_datetime(end).date()

    dates = pd.date_range(start, end, freq="D").date
    preds, trues, out_dates = [], [], []

    for T in dates:
        # skip Feb 29 (to be consistent with training/inference)
        if T.month == 2 and T.day == 29:
            continue
        anchor = pd.Timestamp(T) - pd.Timedelta(days=1)

        # skip if the anchor or target day is missing (requires a continuous daily series)
        if anchor.date() not in water_daily.index or T not in water_daily.index:
            continue

        try:
            Xn, fut_dates = _build_window_Xn(runner, water_daily, anchor)
        except Exception:
            continue

        y_abs = _predict_7_abs(runner, Xn, fut_dates, training=False)  # [7]
        pred1 = float(y_abs[0])  # 1-day ahead
        true  = float(water_daily[T])

        out_dates.append(pd.to_datetime(T))
        preds.append(pred1)
        trues.append(true)

    df = pd.DataFrame({"date": out_dates, "h_true": trues, "h_pred": preds})
    if len(df):
        df["err"] = df["h_pred"] - df["h_true"]
        rmse = float(np.sqrt(np.mean(df["err"]**2)))
    else:
        rmse = None
    return df, rmse

def ui_eval_ytd():
    """
    Gradio callback: plot the 2025 YTD ‘Observed vs Predicted’ (1-day ahead) and provide an RMSE summary.
    Overlay threshold lines on the plot: Alarm 10.7 m, Flood 12 m.
    """
    S = _load_service()
    runner, water_daily = S["runner"], S["water_daily"]

    df, rmse = _backtest_ytd_1day(runner, water_daily, start="2025-01-01")
    if df is None or len(df) == 0:
        return None, "Not enough data to backtest 2025 YTD.", pd.DataFrame()

    # plot
    fig = plt.figure(figsize=(10.5, 4.0))
    plt.plot(df["date"], df["h_true"], label="Observed", linewidth=1.8)
    plt.plot(df["date"], df["h_pred"], label="FNO (1-day ahead)", linewidth=1.8)
    # threshold lines
    plt.axhline(ALARM_LEVEL, linestyle="--", linewidth=1, label=f"Alarm {ALARM_LEVEL:.1f} m")
    plt.axhline(FLOOD_LEVEL, linestyle="--", linewidth=1, label=f"Flood {FLOOD_LEVEL:.1f} m")

    plt.title("2025 YTD — Observed vs Predicted (1-day ahead)")
    plt.xlabel("Date"); plt.ylabel("Water Level (m)")
    plt.xticks(rotation=20); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

    note = (
        f"Backtest window: 2025-01-01 → {df['date'].iloc[-1].date()} "
        f"| N={len(df)} | RMSE={rmse:.3f} m "
        f"| Alarm={ALARM_LEVEL:.1f} m, Flood={FLOOD_LEVEL:.1f} m"
    )
    # return the df for download/viewing (including err)
    return fig, note, df[["date", "h_true", "h_pred", "err"]]

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
    water_daily = df.groupby('date')['h'].mean()  # pandas.Series: index=date -> value=h(m)

    try:
        live_daily = get_recent_daily_cached(
            station_code=STATION_CODE,
            cache_path=LIVE_CACHE,
            ttl_seconds=900,  # 15 minutes cache
        )
        if live_daily is not None and not live_daily.empty:
            water_daily = merge_into_water_daily(water_daily, live_daily)
            print(f"[app] merged live daily: +{len(live_daily)} day(s)")
    except Exception as e:
        print("[app] live merge skipped:", e)

    # day5: try to load historical residual band
    resid_sigma = None
    if os.path.exists(RESID_PATH):
        resid_sigma = json.load(open(RESID_PATH, "r", encoding="utf-8"))

    _APP_CACHE.update(dict(
        runner=runner,
        water_daily=water_daily,
        resid_sigma=resid_sigma,  # residual band statistics
        mc_cache={},  # MC Dropout results cache
        ready=True
    ))
    print(f"[app] loaded in {time.perf_counter()-t0:.2f}s, days={len(water_daily)}")
    return _APP_CACHE

# ----------------- Tab 1: Today → 7 Days -----------------
def ui_predict_today(show_uncertainty=False, src_choice="Historical residuals (fast)", mc_samples=30):
    S = _load_service()
    runner, water_daily = S["runner"], S["water_daily"]
    resid_sigma = S.get("resid_sigma")
    mc_cache = S.get("mc_cache", {})

    # Choose the window end: prefer “today (UTC+07)”
    # if data is missing, fall back to the last observed day
    today = _today_utc7_date()
    anchor = today if today in water_daily.index else max(water_daily.index)
    if len(water_daily) < SEQ_LENGTH:
        return None, f"Not enough data for a {SEQ_LENGTH}-day window (currently {len(water_daily)} days).", None

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
                # --- YTD backtest (1-day ahead) ---
                with gr.Row():
                    btn_bt = gr.Button("Run 2025 YTD backtest (1-day ahead)", variant="primary")
                ytd_plot = gr.Plot()
                ytd_note = gr.Markdown()
                ytd_df = gr.Dataframe(interactive=False)
                btn_bt.click(fn=ui_eval_ytd, inputs=None, outputs=[ytd_plot, ytd_note, ytd_df])

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