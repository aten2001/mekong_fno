# app/app.py
"""
Mekong FNO - Upstream-Assisted Water Level Forecasting System (Gradio app)

This module serves two purposes:
1) Provide Gradio UI callbacks for forecasting and evaluation.
2) Host a process-level singleton cache that loads the model + data once per process.

Key design constraints:
- This file may be run as a script: `python app/app.py`.
  Therefore we bootstrap `sys.path` early so absolute imports work regardless of CWD.
- Hugging Face Spaces typically provides a persistent `/data` volume. Runtime artifacts
  are routed to `RUNTIME_ROOT` (default: `/data/runtime` on HF, otherwise `<repo>/.runtime`).
- Some globals are initialized at import-time (e.g. `LAYOUT = get_layout()`).
  Keep import-time side effects minimal and deterministic.
"""

# =============================================================================
# Bootstrap import path (so `python app/app.py` works from any CWD)
# =============================================================================
# NOTE:
# - When running as a script, Python sets sys.path based on the working directory,
#   which can break absolute imports such as `from src.runner import ...`.
# - Add the repo root to sys.path to make imports stable.
import os, sys

# Force TensorFlow to use legacy tf.keras (Keras 2) behavior on TF>=2.16
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

ROOT = os.path.dirname(os.path.dirname(__file__))

# Ensure repo root is the very first import root.
try:
    while ROOT in sys.path:
        sys.path.remove(ROOT)
except Exception:
    pass
sys.path.insert(0, ROOT)

# =============================================================================
# Imports
# =============================================================================
# Standard library
import json, glob, time, threading
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

# Third-party
import numpy as np
import pandas as pd
import tensorflow as tf

# Matplotlib: force non-interactive backend for server environments (HF/CI)
# - `MPLBACKEND` must be set *before* importing matplotlib in some setups.
# - `plt.ioff()` avoids accidental GUI usage.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.ioff()  # turn off interactive mode to avoid accidentally launching a GUI

import gradio as gr

# Project modules
from src.runner import TenYearUnifiedRunner
from src.model_fno import SeasonalFNO1D
from src.core.assist import (
    apply_backtest_correction as _apply_3s_correction,
    apply_future_correction as _apply_upstream_correction_future,
    fit_pakse_params_for_tab1 as _fit_pakse_params_for_tab1,
    fit_upstream_residual_model as _fit_3s_residual_model,
    fit_w3s_params_for_tab1 as _fit_w3s_params_for_tab1,
    upstream_raw_available_mask as _upstream_raw_available_mask,
)
from src.core.backtest import (
    attach_persistence_backtest as _attach_persistence_backtest,
    backtest_ytd_1day as _backtest_ytd_1day,
    mae_against_truth as _mae_against_truth,
    rmse_against_truth as _rmse_against_truth,
)
from src.core.forecast import (
    build_window_Xn as _build_window_Xn,
    latest_contiguous_anchor as _latest_contiguous_anchor,
    predict_7_abs as _predict_7_abs,
    today_utc7_date as _today_utc7_date,
)
from src.data.historical import load_runner_history_daily
from src.data.live import fetch_live_daily_series
from src.data.merge import build_target_daily_series, merge_upstream_daily_series
from src.data.upstream import load_upstream_daily_csv
from src.data.validation import recent_missing_dates, stable_live_daily_until

# Live daily means from MRC API (cached)
from src.backfill import read_backfill, series_from_any

# =============================================================================
# Global configuration / constants
# =============================================================================
# Model IO lengths: keep consistent with training
SEQ_LENGTH = 150
PRED_LENGTH = 7

# Repo root alias (historically named REPO_ROOT in this codebase)
REPO_ROOT = ROOT

# Input/asset locations (overridable via env)
ASSETS_DIR  = os.environ.get("ASSETS_DIR",  os.path.join(REPO_ROOT, "assets"))
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", os.path.join(REPO_ROOT, "weights"))
CSV_DIR     = os.environ.get("CSV_DIR",     os.path.join(REPO_ROOT, "data"))

# Runtime root:
# - Local:  <repo>/.runtime
# - HF:     /data/runtime  (persistent volume)
DEFAULT_RUNTIME = "/data/runtime" if os.path.isdir("/data") else os.path.join(REPO_ROOT, ".runtime")
os.environ.setdefault("RUNTIME_ROOT", DEFAULT_RUNTIME)

# =============================================================================
# Runtime layout & canonical runtime files
# =============================================================================
# The runtime layout encapsulates all run-time outputs (caches, artifacts, logs).
# IMPORTANT: `LAYOUT = get_layout()` runs at import-time and may create directories.
from app.runtime_paths import get_layout
from app.runtime_files import (
    backfill_path,
    live_cache_daily_default,
    live_cache_3s,
    live_cache_pakse,
    assist_params_path,
    backtest_cache_path,
    backtest_metrics_cache_path,
)
from app.runtime_lock import lock_for_path, atomic_write_json, atomic_write_parquet
from app.sync_artifacts import sync_backfill_from_dataset, sync_status_from_dataset

# NOTE: import-time side effect: may mkdir runtime folders.
LAYOUT = get_layout()

# Keep string versions for libraries that expect `str` paths (e.g. pandas read/write helpers).
RUNTIME_CACHE_DIR = str(LAYOUT.cache)
RUNTIME_ART_DIR   = str(LAYOUT.artifacts)

# Canonical runtime files (generated by runtime_files.py)
# - Prefer Path for locking/atomic writes
# - Provide str for legacy helpers/third-party APIs
BACKFILL_P: Path = backfill_path(LAYOUT)                  # Path for lock + atomic parquet writes
BACKFILL_PATH    = str(BACKFILL_P)                        # str for existing backfill helpers
DATASET_REPO = os.environ.get("HF_DATASET_REPO", "").strip()
HF_READ_TOKEN = os.environ.get("HF_READ_TOKEN", "").strip() or None
LIVE_CACHE       = str(live_cache_daily_default(LAYOUT))  # JSON cache: target station recent daily series
LIVE_CACHE_3S    = str(live_cache_3s(LAYOUT))             # JSON cache: 3S upstream recent daily series
LIVE_CACHE_PAKSE = str(live_cache_pakse(LAYOUT))          # JSON cache: Pakse upstream recent daily series

# =============================================================================
# Station configuration
# =============================================================================
# Target station (Stung Treng): code used by MRC API
STATION_CODE = os.environ.get("STUNG_TRENG_CODE", "014501")

# Pakse (013901) CSV path & station code
PAKSE_CSV  = os.path.join(CSV_DIR, "Water Level.ManualLA_013901_Pakse.csv")
PAKSE_CODE = os.environ.get("PAKSE_CODE", "013901")

# 3S (014500) CSV path & station code
W3S_CSV  = os.path.join(CSV_DIR, "Water Level.TelemetryKH_014500_3S at Sekong bridge.csv")
W3S_CODE = os.environ.get("W3S_CODE", "014500")

# Assets used for inference / evaluation
CLIM_PATH = os.path.join(ASSETS_DIR, "clim_vec.npy")
NORM_PATH = os.path.join(ASSETS_DIR, "norm_stats.json")
PHASE_JSON = os.path.join(ASSETS_DIR, "phase_report.json")
RESID_PATH = os.path.join(ASSETS_DIR, "residual_sigma.json")  # historical residual band (fast uncertainty)

# Risk thresholds shown on plots (domain-specific)
ALARM_LEVEL = 10.7   # meters
FLOOD_LEVEL = 12.0   # meters

# Season configuration for upstream-assist behavior
WET_MONTHS = (6, 7, 8, 9, 10, 11)
DRY_SHRINK = float(os.environ.get("PAKSE_DRY_SHRINK", "0.4"))  # λ ∈ [0,1], shrink delta outside wet season

# =============================================================================
# Evaluation plot display controls
# =============================================================================
EVAL_LAYER_CHOICES = [
    "Observed",
    "Persistence",
    "FNO",
    "FNO + 3S",
    "FNO + Pakse",
    "Alarm",
    "Flood",
]

DEFAULT_EVAL_LAYERS = [
    "Observed",
    "Persistence",
    "FNO + Pakse",
    "Alarm",
    "Flood",
]

EVAL_COMPARISON_MODE_TO_LAYERS = {
    "Observed vs Persistence": ["Observed", "Persistence", "Alarm", "Flood"],
    "Observed vs FNO": ["Observed", "FNO", "Alarm", "Flood"],
    "Observed vs FNO + 3S": ["Observed", "FNO + 3S", "Alarm", "Flood"],
    "Observed vs FNO + Pakse": ["Observed", "FNO + Pakse", "Alarm", "Flood"],
    "Observed vs FNO + Pakse vs Persistence": ["Observed", "Persistence", "FNO + Pakse", "Alarm", "Flood"],
    "All model variants": ["Observed", "Persistence", "FNO", "FNO + 3S", "FNO + Pakse", "Alarm", "Flood"],
}

def _resolve_eval_layers(comparison_mode: str, use_custom_layers: bool, selected_layers):
    """
    Resolve which layers should be drawn in Tab2.

    Priority:
    1) If custom-layer mode is enabled, use the CheckboxGroup selection.
    2) Otherwise use the predefined comparison-mode mapping.
    3) Always preserve the canonical order in EVAL_LAYER_CHOICES.
    """
    if use_custom_layers:
        layers = list(selected_layers or DEFAULT_EVAL_LAYERS)
    else:
        layers = list(EVAL_COMPARISON_MODE_TO_LAYERS.get(
            comparison_mode,
            DEFAULT_EVAL_LAYERS
        ))

    keep = set(layers)
    ordered = [x for x in EVAL_LAYER_CHOICES if x in keep]
    return ordered or DEFAULT_EVAL_LAYERS

# =============================================================================
# Data loading and merge helpers are imported from src.data.
# =============================================================================
# -----------------------------------------------------------------------------
# Upstream-assist core helpers are imported from src.core.assist.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Assist-param cache and backtest cache: concurrency-safe on-disk artifacts
# -----------------------------------------------------------------------------
def _load_or_fit_assist_params(
    station: str,
    upstream_name: str,
    upstream_daily: Optional[pd.Series],
    fit_fn,
    *,
    model_id: str,
    last_date,
):
    """
    Load (or fit) upstream-assist parameters with a concurrency-safe on-disk cache.

    Concurrency & consistency:
    - Fast path: unlocked read for performance.
    - Slow path: lock + double-check + fit + atomic write.
    - Atomic write prevents readers from seeing partially-written files.

    Cache key dimensions:
    - station, upstream_name, model_id, last_date

    Returns:
        params dict (truthy) or None if unavailable / insufficient data.
    """
    if upstream_daily is None or len(upstream_daily) == 0:
        return None

    p = assist_params_path(LAYOUT, station, upstream_name, model_id, last_date)

    # Fast path: unlocked read
    if p.exists():
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            params = payload.get("params")
            if params:
                print(f"[assist] cache hit: {p}")
                return params
        except Exception:
            pass

    # Locked path: double-check + fit + atomic write
    with lock_for_path(p):
        if p.exists():
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                params = payload.get("params")
                if params:
                    print(f"[assist] cache hit(after lock): {p}")
                    return params
            except Exception:
                pass

        print(f"[assist] cache miss -> fit: {p}")
        params = fit_fn()
        if params:
            payload = {
                "station": station,
                "upstream": upstream_name,
                "model_id": model_id,
                "last_date": str(last_date),
                "params": params,
            }
            atomic_write_json(p, payload)
            print(f"[assist] cached: {p}")
        return params

# =============================================================================
# Process-level singleton cache (model + data)
# =============================================================================
# These globals are created at import-time.
# - `_APP_LOCK` guards initialization to ensure only one thread builds the cache.
# - `_APP_CACHE` stores the ready-to-use service objects for all callbacks.
_APP_CACHE: dict = {}
_APP_LOCK = threading.Lock()

def _load_service(force_reload: bool = False):
    """
    Initialize and cache the end-to-end inference “service” (model, data, upstream series).

    Thread-safety:
    - Double-checked locking: unlocked fast path, then locked initialization path.
    - Build into `new_cache` and swap at the end to avoid exposing partial state.

    Args:
        force_reload:
            If True, rebuild the cache even if already ready.

    Returns:
        dict:
            A cache dict containing at least:
              - runner, water_daily, model_id, (optional) w3s_daily, pakse_daily, resid_sigma
    """
    # Fast path (unlocked)
    if _APP_CACHE.get("ready") and not force_reload:
        return _APP_CACHE

    # Locked path
    with _APP_LOCK:
        # double-check after acquiring the lock
        if _APP_CACHE.get("ready") and not force_reload:
            return _APP_CACHE

        t0 = time.perf_counter()

        # IMPORTANT:
        # - Do NOT clear the old cache first.
        # - Build `new_cache` fully, then atomically swap.
        new_cache: dict = {}

        runner = TenYearUnifiedRunner(CSV_DIR, seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH)
        water_daily_hist = load_runner_history_daily(runner, 2015, 2025, allow_missing_u=True)

        # Training-time artifacts: climatology + normalization stats
        runner.set_climatology(np.load(CLIM_PATH))
        st = json.load(open(NORM_PATH, "r", encoding="utf-8"))
        runner.norm_stats = st

        # Model build + weights load
        model = SeasonalFNO1D(modes=64, width=96, num_layers=4, input_features=6, dropout_rate=0.1, l2=1e-5)
        _ = model(np.zeros((1, SEQ_LENGTH, 6), dtype=np.float32), training=False)
        ckpt = _find_ckpt()
        model.load_weights(ckpt)
        runner.model = model

        model_id = Path(ckpt).name

        # ---------------------------------------------------------------------
        # Target station: build daily mean series from history + backfill + live
        # ---------------------------------------------------------------------
        # read historical backfill (Parquet)
        backfill = read_backfill(BACKFILL_PATH)

        # fetch live daily means
        live_daily = fetch_live_daily_series(
            station_code=STATION_CODE,
            cache_path=LIVE_CACHE,
            ttl_seconds=900,
            error_label="[app] live fetch skipped",
        )

        water_daily = build_target_daily_series(water_daily_hist, backfill, live_daily)

        # ---------------------------------------------------------------------------------------
        # Persist “stable” part of live data into backfill (<= today-1).
        # Rationale: avoid repeated API calls for old days; treat yesterday and earlier as stable.
        # ---------------------------------------------------------------------------------------
        try:
            cutoff = (pd.Timestamp(_today_utc7_date()) - pd.Timedelta(days=1))
            stable = stable_live_daily_until(live_daily, cutoff)

            if stable is not None and len(stable) > 0:
                # Lock on the canonical backfill parquet path to prevent concurrent writes.
                with lock_for_path(BACKFILL_P):
                    cur = series_from_any(read_backfill(BACKFILL_PATH))
                    cur = series_from_any(cur) if cur is not None else None

                    if cur is None or len(cur) == 0:
                        merged = stable
                    else:
                        merged = pd.concat([cur, stable]).groupby(level=0).last().sort_index()

                    _atomic_write_backfill_series(BACKFILL_P, merged)

        except Exception as e:
            print("[app] backfill write skipped:", e)

        print(f"[app] daily merged: days={len(water_daily)}, range={min(water_daily.index)}→{max(water_daily.index)}")

        # ---------------------------------------------------------------------
        # Upstream stations: 3S and Pakse (CSV history + live API tail)
        # ---------------------------------------------------------------------
        w3s_hist = load_upstream_daily_csv(W3S_CSV)
        live_3s = fetch_live_daily_series(
            station_code=W3S_CODE,
            cache_path=LIVE_CACHE_3S,
            ttl_seconds=900,
            error_label="[3S] live fetch skipped",
        )
        w3s_daily = merge_upstream_daily_series(w3s_hist, live_3s)

        print(f"[3S] path={W3S_CSV} exists={os.path.exists(W3S_CSV)}")
        if w3s_daily is None or len(w3s_daily) == 0:
            print("[3S] empty after load/merge")
        else:
            print(f"[3S] len={len(w3s_daily)}, range={min(w3s_daily.index)}→{max(w3s_daily.index)}")

        # Pakse daily series
        pakse_hist = load_upstream_daily_csv(PAKSE_CSV)
        live_pakse = fetch_live_daily_series(
            station_code=PAKSE_CODE,
            cache_path=LIVE_CACHE_PAKSE,
            ttl_seconds=900,
            error_label="[pakse] live fetch skipped",
        )
        pakse_daily = merge_upstream_daily_series(pakse_hist, live_pakse)

        if pakse_daily is not None and len(pakse_daily) > 0:
            print(f"[PAKSE] len={len(pakse_daily)}, range={min(pakse_daily.index)}→{max(pakse_daily.index)}")
        else:
            print("[PAKSE] empty after load/merge")

        # Historical residual band (fast uncertainty option)
        resid_sigma = None
        if os.path.exists(RESID_PATH):
            resid_sigma = json.load(open(RESID_PATH, "r", encoding="utf-8"))

        # Populate cache fully before swap
        new_cache.update(dict(
            runner=runner,
            water_daily=water_daily,
            resid_sigma=resid_sigma,
            mc_cache={},          # in-memory MC cache (keyed by (anchor, N))
            w3s_daily=w3s_daily,
            pakse_daily=pakse_daily,
            model_id=model_id,
            ready=True,
        ))

        # Debug: missing days in the recent tail (handled by interpolation/anchor selection)
        tail_missing = recent_missing_dates(water_daily, days=14)
        if tail_missing:
            print(f"[app][warn] recent missing days auto-handled: {tail_missing}")

        print(f"[app] loaded in {time.perf_counter()-t0:.2f}s, days={len(water_daily)}")

        # Atomic swap: old cache remains valid for in-flight requests
        _APP_CACHE.clear()
        _APP_CACHE.update(new_cache)
        return _APP_CACHE

# =============================================================================
# Upstream-assist fitting helpers are imported from src.core.assist.
# =============================================================================
# =============================================================================
# Forecast core helpers are imported from src.core.forecast.
# =============================================================================
# LIVE_CACHE   = os.path.join(RUNTIME_CACHE_DIR, "live_recent_daily.json")

def _find_ckpt(weights_dir=WEIGHTS_DIR):
    """
    Locate a TensorFlow checkpoint prefix under the given weights directory.

    Policy:
    - Prefer TensorFlow's latest_checkpoint() resolution when available.
    - Fall back to scanning "*.ckpt.index" (older layouts / incomplete metadata).

    Returns:
        str: Filesystem path to the checkpoint prefix (no extension).

    Raises:
        FileNotFoundError: If no checkpoint can be found in the directory.
    """
    ckpt = tf.train.latest_checkpoint(weights_dir)
    if ckpt: return ckpt
    idx = glob.glob(os.path.join(weights_dir, "*.ckpt.index"))
    if idx:  return idx[0].replace(".index","")
    raise FileNotFoundError("No TF checkpoint found in 'weights/'")

# =============================================================================
# Time utilities are imported from src.core.forecast.
# =============================================================================
# =============================================================================
# Input normalization is implemented in src.core.forecast.
# =============================================================================
# =============================================================================
# Window building is implemented in src.core.forecast.
# =============================================================================
# =============================================================================
# Forecast inference wrappers are implemented in src.core.forecast.
# =============================================================================
# =============================================================================
# Backtest core is implemented in src.core.backtest.
# =============================================================================
# =============================================================================
# Persistence baseline core is implemented in src.core.backtest.
# =============================================================================
# =============================================================================
# Tab2 note / metrics summary helpers
# =============================================================================
EVAL_METRIC_MODEL_ORDER = [
    "Persistence",
    "FNO",
    "FNO + 3S",
    "FNO + Pakse",
]

def _build_eval_metrics_summary(metrics_map: dict) -> pd.DataFrame:
    """
    Build the full metrics summary table shown under the plot.

    Only models with available RMSE values are included.
    """
    rows = []
    for model_name in EVAL_METRIC_MODEL_ORDER:
        rmse_val = metrics_map.get(model_name)
        if rmse_val is None:
            continue
        rows.append({
            "Model": model_name,
            "RMSE (m)": round(float(rmse_val), 3),
        })
    return pd.DataFrame(rows, columns=["Model", "RMSE (m)"])

def _build_eval_note(
    *,
    end_date,
    horizon: int,
    n_rows: int,
    visible_layers,
    metrics_map: dict,
):
    """
    Build the short main note for Tab2.

    Policy:
    - Show only context + currently visible model metrics.
    - Do not include hidden model metrics.
    """
    parts = [
        f"Backtest period: 2025-01-01 → {end_date}",
        f"Horizon={horizon} day(s)",
        f"N={n_rows}",
        f"Visible={', '.join(visible_layers)}",
    ]

    for model_name in EVAL_METRIC_MODEL_ORDER:
        if model_name in visible_layers:
            rmse_val = metrics_map.get(model_name)
            if rmse_val is not None:
                parts.append(f"{model_name} RMSE={rmse_val:.3f} m")

    if "Alarm" in visible_layers:
        parts.append(f"Alarm={ALARM_LEVEL:.1f} m")
    if "Flood" in visible_layers:
        parts.append(f"Flood={FLOOD_LEVEL:.1f} m")

    return " | ".join(parts)

# =============================================================================
# Tab2 advanced diagnostics (availability-aware comparisons & routing)
# =============================================================================
def _prepare_assist_eval_bundle(horizon=1):
    """
    Build a shared evaluation bundle for:
      - 3S-only comparison
      - Pakse-only comparison
      - common-overlap comparison
      - availability summary
      - routing summary

    Returns:
        dict with either:
          {"error": "..."}
        or:
          {
            "df": ...,
            "dates": ...,
            "h_true": ...,
            "h_pred": ...,
            "base_mask": ...,
            "params_3s": ...,
            "params_pk": ...,
            "y_corr_3s": ...,
            "y_corr_pk": ...,
            "has_3s": ...,
            "has_pk": ...,
            "k_3s": ...,
            "k_pk": ...,
            "horizon": ...,
          }
    """
    S = _load_service()
    runner, water_daily = S["runner"], S["water_daily"]
    w3s_daily = S.get("w3s_daily")
    pakse_daily = S.get("pakse_daily")

    year = 2025
    k_h = int(horizon)
    last_date = max(water_daily.index)
    model_id = S.get("model_id") or Path(_find_ckpt()).name

    df, rmse_fno_full = _load_or_run_backtest_ytd_cached(
        year=year,
        k=k_h,
        station=STATION_CODE,
        model_id=model_id,
        last_date=last_date,
        runner=runner,
        water_daily=water_daily,
    )

    if df is None or len(df) == 0:
        return {"error": f"Not enough data (h={horizon})."}

    df = df.copy()
    dates = pd.to_datetime(df["date"]).dt.date.values
    h_true = df["h_true"].values.astype(float)
    h_pred = df["h_pred"].values.astype(float)
    base_mask = np.isfinite(h_true) & np.isfinite(h_pred)

    out = {
        "df": df,
        "dates": dates,
        "h_true": h_true,
        "h_pred": h_pred,
        "base_mask": base_mask,
        "horizon": k_h,
        "rmse_fno_full": rmse_fno_full,
        "mae_fno_full": _mae_against_truth(h_true, h_pred),
        "params_3s": None,
        "params_pk": None,
        "y_corr_3s": None,
        "y_corr_pk": None,
        "has_3s": np.zeros(len(df), dtype=bool),
        "has_pk": np.zeros(len(df), dtype=bool),
        "k_3s": None,
        "k_pk": None,
    }

    # 3S
    if w3s_daily is not None and len(w3s_daily) > 0:
        params_3s = _fit_3s_residual_model(df, w3s_daily, k_grid=(0, 1, 2, 3))
        if params_3s:
            y_corr_3s, _ = _apply_3s_correction(df, w3s_daily, params_3s)
            k_3s = int(params_3s["k"])
            has_3s = _upstream_raw_available_mask(dates, w3s_daily, k_3s)

            out.update({
                "params_3s": params_3s,
                "y_corr_3s": y_corr_3s,
                "has_3s": has_3s,
                "k_3s": k_3s,
            })

    # Pakse
    if pakse_daily is not None and len(pakse_daily) > 0:
        params_pk = _fit_3s_residual_model(df, pakse_daily, k_grid=(0, 1, 2, 3))
        if params_pk:
            y_corr_pk, _ = _apply_3s_correction(df, pakse_daily, params_pk)
            k_pk = int(params_pk["k"])
            has_pk = _upstream_raw_available_mask(dates, pakse_daily, k_pk)

            out.update({
                "params_pk": params_pk,
                "y_corr_pk": y_corr_pk,
                "has_pk": has_pk,
                "k_pk": k_pk,
            })

    return out

# =============================================================================
# Tab2 callback: YTD backtest plot (Observed vs FNO + Pakse vs Persistence)
# =============================================================================
def ui_eval_ytd(
    horizon=1,
    comparison_mode="Observed vs FNO + Pakse vs Persistence",
    use_custom_layers=False,
    selected_layers=None,
):
    """
    Tab2 callback: run a k-day-ahead backtest from 2025-01-01 to the latest available date
    and render a configurable evaluation plot.

    Display logic:
    - Comparison mode controls the default visible lines.
    - If custom-layer mode is enabled, CheckboxGroup selections override comparison mode.

    Returns:
        (fig, note, metrics_summary, df):
          fig: matplotlib.figure.Figure | None
          note: short summary string for the current visible view only
          metrics_summary: full RMSE summary table for all evaluated variants
          df: raw backtest rows with prediction columns
    """
    S = _load_service()
    runner, water_daily = S["runner"], S["water_daily"]
    w3s_daily = S.get("w3s_daily")
    pakse_daily = S.get("pakse_daily")

    year = 2025
    k = int(horizon)
    last_date = max(water_daily.index)
    model_id = S.get("model_id") or Path(_find_ckpt()).name

    df, rmse = _load_or_run_backtest_ytd_cached(
        year=year,
        k=k,
        station=STATION_CODE,
        model_id=model_id,
        last_date=last_date,
        runner=runner,
        water_daily=water_daily,
    )

    empty_metrics = pd.DataFrame(columns=["Model", "RMSE (m)"])

    if df is None or len(df) == 0:
        return (
            None,
            f"Not enough data to backtest from 2025-01-01 to the latest available date (h={horizon}).",
            empty_metrics,
            pd.DataFrame(),
        )

    # -------------------------------------------------------------------------
    # Attach persistence baseline on the same backtest rows
    # -------------------------------------------------------------------------
    df, rmse_pers = _attach_persistence_backtest(df, water_daily, horizon=k)

    # -------------------------------------------------------------------------
    # Optional overlays: assist correction on the same backtest rows
    # -------------------------------------------------------------------------
    y_corr = None
    rmse_corr = None
    if w3s_daily is not None and len(w3s_daily) > 0:
        params = _fit_3s_residual_model(df, w3s_daily, k_grid=(0, 1, 2, 3))
        if params:
            y_corr, deltas = _apply_3s_correction(df, w3s_daily, params)
            df["h_pred_3S"] = y_corr
            df["delta_3S"] = deltas
            rmse_corr = _rmse_against_truth(df["h_true"].values, y_corr)

    y_corr_pk = None
    rmse_pk = None
    if pakse_daily is not None and len(pakse_daily) > 0:
        params_pk = _fit_3s_residual_model(df, pakse_daily, k_grid=(0, 1, 2, 3))
        if params_pk:
            y_corr_pk, deltas_pk = _apply_3s_correction(df, pakse_daily, params_pk)
            df["h_pred_Pakse"] = y_corr_pk
            df["delta_Pakse"] = deltas_pk
            rmse_pk = _rmse_against_truth(df["h_true"].values, y_corr_pk)

    visible_layers = _resolve_eval_layers(
        comparison_mode=comparison_mode,
        use_custom_layers=use_custom_layers,
        selected_layers=selected_layers,
    )

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(10.8, 4.4))

    if "Observed" in visible_layers:
        plt.plot(df["date"], df["h_true"], label="Observed", linewidth=2.0)

    if "Persistence" in visible_layers and "h_pred_Pers" in df.columns:
        plt.plot(df["date"], df["h_pred_Pers"], label=f"Persistence ({horizon}-day ahead)", linewidth=1.8)

    if "FNO" in visible_layers:
        plt.plot(df["date"], df["h_pred"], label=f"FNO ({horizon}-day ahead)", linewidth=1.8)

    if "FNO + 3S" in visible_layers and y_corr is not None:
        plt.plot(df["date"], y_corr, label=f"FNO + 3S ({horizon}-day)", linewidth=1.8)

    if "FNO + Pakse" in visible_layers and y_corr_pk is not None:
        plt.plot(df["date"], y_corr_pk, label=f"FNO + Pakse ({horizon}-day)", linewidth=1.8)

    if "Alarm" in visible_layers:
        plt.axhline(ALARM_LEVEL, linestyle="--", color="darkgoldenrod", linewidth=1, label=f"Alarm {ALARM_LEVEL:.1f} m")

    if "Flood" in visible_layers:
        plt.axhline(FLOOD_LEVEL, linestyle="--", color="red", linewidth=1, label=f"Flood {FLOOD_LEVEL:.1f} m")

    start_date = "2025-01-01"
    end_date = df["date"].iloc[-1].date()
    view_name = "Custom layers" if use_custom_layers else comparison_mode

    plt.title(
        f"Observed vs Predicted ({horizon}-day ahead)\n"
        f"Backtest: {start_date} to {end_date} | View: {view_name}"
    )
    plt.xlabel("Date")
    plt.ylabel("Water Level (m)")
    plt.xticks(rotation=20)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Metrics summary + short note
    # -------------------------------------------------------------------------
    metrics_map = {
        "Persistence": rmse_pers,
        "FNO": rmse,
        "FNO + 3S": rmse_corr,
        "FNO + Pakse": rmse_pk,
    }

    metrics_summary = _build_eval_metrics_summary(metrics_map)

    note = _build_eval_note(
        end_date=end_date,
        horizon=horizon,
        n_rows=len(df),
        visible_layers=visible_layers,
        metrics_map=metrics_map,
    )

    # -------------------------------------------------------------------------
    # Raw output table
    # -------------------------------------------------------------------------
    cols = ["date", "h_true", "h_pred", "h_pred_Pers"]

    if "h_pred_3S" in df.columns:
        if "month" not in df.columns:
            df["month"] = pd.to_datetime(df["date"]).dt.month
        df["wet"] = df["month"].isin(WET_MONTHS)
        df["changed"] = np.where(
            np.isfinite(df.get("delta_3S", np.nan)) & (np.abs(df.get("delta_3S", 0.0)) > 1e-6),
            True,
            False,
        )
        cols += ["h_pred_3S", "delta_3S", "wet", "changed"]

    if "h_pred_Pakse" in df.columns:
        cols += ["h_pred_Pakse", "delta_Pakse"]

    return fig, note, metrics_summary, df[cols]

# =============================================================================
# Backtest cache (parquet + metrics json): concurrency-safe on-disk artifacts
# =============================================================================
def _load_or_run_backtest_ytd_cached(
    *,
    year: int,
    k: int,
    station: str,
    model_id: str,
    last_date,   # python date (water_daily index uses date)
    runner,
    water_daily: pd.Series,
):
    """
    Load YTD backtest from runtime cache (parquet + metrics json) with concurrency safety.

    Cache key dimensions:
      - station, model_id, year, k

    Cache validity:
      - meta["last_date"] == str(last_date)

    Consistency guarantees:
      - Both files written under a lock.
      - Both writes are atomic (temp + replace).
      - Metrics JSON is written AFTER parquet to avoid "new meta + old parquet" mismatch.

    Returns:
      (df, rmse): df is pd.DataFrame; rmse is float|None
    """
    k = int(k)
    year = int(year)

    p = backtest_cache_path(LAYOUT, station, model_id, year, k)          # parquet (Path)
    m = backtest_metrics_cache_path(LAYOUT, station, model_id, year, k)  # json (Path)

    def _read_cached_pair_unlocked():
        """
        Best-effort cache read WITHOUT acquiring a lock.
        Returns:
          (df, rmse) on hit; None on miss/stale/broken.
        """
        if not (p.exists() and m.exists()):
            return None

        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[backtest] cache metrics read failed: {m} -> {repr(e)}")
            return None

        try:
            meta_last = meta.get("last_date")
            if meta_last != str(last_date):
                print(f"[backtest] cache stale: k={k}, year={year} -> meta.last_date={meta_last} != {str(last_date)}")
                return None

            df = pd.read_parquet(p)
            if df is None or len(df) == 0:
                # treat as miss to trigger recompute (defensive)
                print(f"[backtest] cache empty parquet treated as miss: {p}")
                return None

            rmse = meta.get("rmse", None)
            rmse = float(rmse) if rmse is not None else None

            print(f"[backtest] cache hit: k={k}, year={year} -> {p}")
            return df, rmse

        except Exception as e:
            print(f"[backtest] cache parquet read/parse failed: {p} -> {repr(e)}")
            return None

    # Fast path: unlocked read (performance).
    hit = _read_cached_pair_unlocked()
    if hit is not None:
        return hit

    # Slow path: lock + double-check + compute + atomic paired write.
    # Use parquet path as the lock target (one of p/m is enough)
    with lock_for_path(p):
        # double-check inside lock (another request may have filled it while we waited)
        hit = _read_cached_pair_unlocked()
        if hit is not None:
            print(f"[backtest] cache hit(after lock): k={k}, year={year} -> {p}")
            return hit

        # compute
        print(f"[backtest] cache miss -> compute: k={k}, year={year} -> {p}")
        try:
            df, rmse = _backtest_ytd_1day(runner, water_daily, start=f"{year}-01-01", horizon=k)
        except Exception as e:
            print(f"[backtest] compute failed: k={k}, year={year} -> {repr(e)}")
            return None, None

        # write only when we have rows
        if df is not None and len(df) > 0:
            meta = {
                "station": station,
                "model_id": model_id,
                "year": year,
                "k": k,
                "n": int(len(df)),
                "rmse": float(rmse) if rmse is not None else None,
                "last_date": str(last_date),
            }

            try:
                # Defensive mkdir (runtime_files generally ensures this, but keep safe)
                p.parent.mkdir(parents=True, exist_ok=True)
                m.parent.mkdir(parents=True, exist_ok=True)

                # Paired atomic write: parquet first, then metrics
                atomic_write_parquet(p, df, index=False)
                atomic_write_json(m, meta)

                print(f"[backtest] cached: k={k}, year={year} -> {p} (+ {m})")
            except Exception as e:
                # If caching fails, still return computed result
                print(f"[backtest] cache write failed (non-fatal): k={k}, year={year} -> {repr(e)}")

        return df, rmse

# =============================================================================
# Backfill persistence (atomic parquet writes)
# =============================================================================
def _atomic_write_backfill_series(p: Path, s: pd.Series) -> None:
    """
    Atomically persist backfill series to parquet at path p.

    Assumes the backfill parquet schema is:
      - date: datetime64[ns]
      - h: float
    (If your src/backfill.py uses a different column name, adjust here to match.)
    """
    if s is None or len(s) == 0:
        return

    # ensure parent exists
    p.parent.mkdir(parents=True, exist_ok=True)

    df_bf = pd.DataFrame({
        "date": pd.to_datetime(list(s.index)),
        "h": pd.to_numeric(np.asarray(s.values), errors="coerce"),
    }).dropna(subset=["date", "h"])

    # keep deterministic order
    df_bf = df_bf.sort_values("date").reset_index(drop=True)

    atomic_write_parquet(p, df_bf, index=False)

# =============================================================================
# Tab1 callback: Forecast Today → +7 days (optional uncertainty & assists)
# =============================================================================
def ui_predict_today(show_uncertainty=False, src_choice="Historical residuals (fast)", mc_samples=30):
    """
    Tab1 callback: generate the next 7-day absolute water-level forecast (UTC+07).

    Enhancements:
    - Optional uncertainty band:
        * Historical residual band (fast, uses assets/residual_sigma.json)
        * MC Dropout quantiles (slow; cached in-memory per process)
    - Optional upstream assist overlays:
        * Pakse
        * 3S
      Each assist is applied with guardrails: lag alignment, no future-leak, dry-season shrink.

    Returns:
        (fig, note, df_out):
          fig: matplotlib figure (or None on failure)
          note: summary string (data/uncertainty/assist availability)
          df_out: table with date, h_pred, (optional p10/p90), (optional assist columns)
    """
    S = _load_service()
    runner, water_daily = S["runner"], S["water_daily"]
    resid_sigma = S.get("resid_sigma")
    mc_cache = S.get("mc_cache", {})
    pakse_daily = S.get("pakse_daily")
    w3s_daily = S.get("w3s_daily")

    last_date = max(water_daily.index)
    model_id = S.get("model_id") or Path(_find_ckpt()).name

    # Choose an anchor that guarantees a contiguous, valid SEQ_LENGTH-day window.
    if len(water_daily) < SEQ_LENGTH:
        return None, f"Not enough data for a {SEQ_LENGTH}-day window (currently {len(water_daily)} days).", None
    try:
        anchor = _latest_contiguous_anchor(water_daily, SEQ_LENGTH)
    except Exception as e:
        return None, f"Not enough contiguous data: {e}", None

    # Build model input window.
    try:
        Xn, fut_dates = _build_window_Xn(runner, water_daily, pd.Timestamp(anchor))
    except Exception as e:
        return None, f"Failed to build input window: {e}", None

    # central (deterministic) prediction
    t0 = time.perf_counter()
    y_abs = _predict_7_abs(runner, Xn, fut_dates, training=False)  # [7]
    latency_ms = (time.perf_counter() - t0) * 1000

    # -------------------------------------------------------------------------
    # Uncertainty band
    # -------------------------------------------------------------------------
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
                # Fallback to MC if residual artifact is missing.
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

    # -------------------------------------------------------------------------
    # Plot: baseline + optional overlays
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(9.5, 4.2))
    plt.plot(fut_dates, y_abs, label="FNO (mean)", linewidth=2)

    # ---- Pakse assist overlay (with cache + guardrails) ----
    pk_note = ""
    params_pk = _load_or_fit_assist_params(
        STATION_CODE,
        "Pakse",
        pakse_daily,
        fit_fn=lambda: _fit_pakse_params_for_tab1(runner, water_daily, pakse_daily, horizon_for_fit=1),
        model_id=model_id,
        last_date=last_date,
    )

    if params_pk:
        y_pk, used_pk, avail_pk, k_pk = _apply_upstream_correction_future(
            y_abs, fut_dates, pakse_daily, params_pk, shrink_dry=DRY_SHRINK, allow_interp=True
        )
        y_pk_plot = y_pk.copy()
        y_pk_plot[~used_pk] = np.nan
        if avail_pk > 0:
            plt.plot(fut_dates, y_pk_plot, label=f"FNO + Pakse (k={k_pk}; avail={avail_pk}/{PRED_LENGTH})", linewidth=2)
            try:
                plt.plot(np.array(fut_dates, dtype="datetime64[ns]")[used_pk], y_pk[used_pk], "o", markersize=3)
            except Exception:
                pass
            pk_note = (
                f" | Pakse assist: k={k_pk}, avail={avail_pk}/{PRED_LENGTH}, "
                f"source=recent\\~29d incl.today; dry-shrink λ={DRY_SHRINK}"
            )
        else:
            pk_note = f" | Pakse not available for this window (k={k_pk}); source=recent\\~29d incl.today"
    else:
        pk_note = " | Pakse assist unavailable (insufficient data/fit)"

    # ---- 3S assist overlay (with cache + guardrails) ----
    s3_note = ""
    params_3s = _load_or_fit_assist_params(
        STATION_CODE,
        "3S",
        w3s_daily,
        fit_fn=lambda: _fit_w3s_params_for_tab1(runner, water_daily, w3s_daily, horizon_for_fit=1),
        model_id=model_id,
        last_date=last_date,
    )

    if params_3s:
        y_3s, used_3s, avail_3s, k_3s = _apply_upstream_correction_future(
            y_abs, fut_dates, w3s_daily, params_3s, shrink_dry=DRY_SHRINK, allow_interp=True
        )
        y_3s_plot = y_3s.copy()
        y_3s_plot[~used_3s] = np.nan
        if avail_3s > 0:
            plt.plot(fut_dates, y_3s_plot, label=f"FNO + 3S (k={k_3s}; avail={avail_3s}/{PRED_LENGTH})", linewidth=2)
            try:
                plt.plot(np.array(fut_dates, dtype="datetime64[ns]")[used_3s], y_3s[used_3s], "o", markersize=3)
            except Exception:
                pass
            s3_note = (
                f" | 3S assist: k={k_3s}, avail={avail_3s}/{PRED_LENGTH}, "
                f"source=CSV⊕recent\\~29d incl.today; dry-shrink λ={DRY_SHRINK}"
            )
        else:
            s3_note = f" | 3S not available for this window (k={k_3s}); source=CSV⊕recent\\~29d incl.today"
    else:
        s3_note = " | 3S assist unavailable (insufficient data/fit)"

    if lo is not None:
        plt.fill_between(fut_dates, lo, hi, alpha=0.18, label=band_note or "Uncertainty band")
    plt.title("Next 7-day absolute water level (UTC+07)")
    plt.xlabel("Date"); plt.ylabel("Water Level (m)")
    plt.xticks(rotation=20); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

    # -------------------------------------------------------------------------
    # Output table
    # -------------------------------------------------------------------------
    df_out = pd.DataFrame({"date": pd.to_datetime(fut_dates).date, "h_pred": y_abs})
    if lo is not None:
        df_out["p10"] = lo; df_out["p90"] = hi

    # output the Pakse-corrected column and the availability flag
    if params_pk:
        # If params exist, the above y_pk/used_pk have been computed (otherwise pk_note will indicate unavailable/not fitted)
        try:
            df_out["h_pred_Pakse"] = y_pk
            df_out["pakse_used"]   = used_pk
        except Exception:
            # If y_pk/used_pk are absent (e.g., fitting failed), skip
            pass

    # 3S-corrected column and availability flag
    if params_3s:
        try:
            df_out["h_pred_3S"] = y_3s
            df_out["s3_used"] = used_3s
        except Exception:
            pass

    note = "Next 7-day absolute water level (UTC+07)"
    if band_note:
        note += f"; Uncertainty: {band_note}"
    note += (s3_note or "")
    note += (pk_note or "")
    return fig, note, df_out

# =============================================================================
# Phase-report helpers (assets/phase_report.json)
# =============================================================================
def _load_phase_json_or_fallback():
    """
    Load the phase-alignment report from assets, if available.

    Returns:
        dict | None: Parsed JSON payload, or None if the artifact is missing.
    """
    if os.path.exists(PHASE_JSON):
        with open(PHASE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def ui_phase_table(scope):
    """
    Tab2 callback: render a one-row summary table from phase_report.json.

    Args:
        scope: One of {"Merged","Dry","Wet"} selecting which evaluation window to display.

    Returns:
        pd.DataFrame: A one-row table (or a one-row message when the artifact is missing).
    """
    mapping = {"Merged":"all", "Dry":"dry", "Wet":"wet"}
    key = mapping.get(scope, "all")
    rep = _load_phase_json_or_fallback()
    if rep is None:
        msg = "Missing assets/phase_report.json. Please run: python -m scripts.make_phase_report"
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

# =============================================================================
# Tab2 callbacks: windowed comparisons (FNO vs FNO + upstream assist)
# =============================================================================
def ui_compare_fno_vs_3s_window(horizon=1):
    """
    Compare baseline FNO vs FNO + 3S assist on the intersected window where 3S is usable.

    Policy:
    - Fit assist params on the same YTD backtest rows.
    - Evaluate only on dates where the lagged upstream features exist (no extrapolation).

    Returns:
        pd.DataFrame: One-row metrics table, or a message table if comparison cannot run.
    """
    S = _load_service()
    runner, water_daily = S["runner"], S["water_daily"]
    w3s_daily = S.get("w3s_daily")

    year = 2025
    k_h = int(horizon)
    last_date = max(water_daily.index)
    model_id = S.get("model_id") or Path(_find_ckpt()).name

    df, _ = _load_or_run_backtest_ytd_cached(
        year=year,
        k=k_h,
        station=STATION_CODE,
        model_id=model_id,
        last_date=last_date,
        runner=runner,
        water_daily=water_daily,
    )

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
    Compare baseline FNO vs FNO + Pakse assist on the intersected window where Pakse is usable.

    Returns:
        pd.DataFrame: One-row metrics table, or a message table if comparison cannot run.
    """
    S = _load_service()
    runner, water_daily = S["runner"], S["water_daily"]
    pakse_daily = S.get("pakse_daily")

    year = 2025
    k_h = int(horizon)
    last_date = max(water_daily.index)
    model_id = S.get("model_id") or Path(_find_ckpt()).name

    df, _ = _load_or_run_backtest_ytd_cached(
        year=year,
        k=k_h,
        station=STATION_CODE,
        model_id=model_id,
        last_date=last_date,
        runner=runner,
        water_daily=water_daily,
    )

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

def ui_compare_common_overlap_window(horizon=1):
    """
    Fair head-to-head comparison on the same dates where both 3S and Pakse
    are available under their fitted lag-k settings.
    """
    B = _prepare_assist_eval_bundle(horizon=horizon)
    if "error" in B:
        return pd.DataFrame({"message": [B["error"]]})

    if B["params_3s"] is None or B["params_pk"] is None:
        return pd.DataFrame({"message": ["Need both fitted 3S and Pakse assist parameters to compare on common dates."]})

    h_true = B["h_true"]
    h_pred = B["h_pred"]
    y3 = np.asarray(B["y_corr_3s"], dtype=float)
    ypk = np.asarray(B["y_corr_pk"], dtype=float)

    mask = (
        B["base_mask"]
        & B["has_3s"]
        & B["has_pk"]
        & np.isfinite(y3)
        & np.isfinite(ypk)
    )

    if mask.sum() == 0:
        return pd.DataFrame({"message": ["No common overlap dates where both 3S and Pakse are available."]})

    idx = np.where(mask)[0]
    d_sub = np.asarray(B["dates"])[idx]

    rmse_fno = _rmse_against_truth(h_true[idx], h_pred[idx])
    rmse_3s = _rmse_against_truth(h_true[idx], y3[idx])
    rmse_pk = _rmse_against_truth(h_true[idx], ypk[idx])

    mae_fno = _mae_against_truth(h_true[idx], h_pred[idx])
    mae_3s = _mae_against_truth(h_true[idx], y3[idx])
    mae_pk = _mae_against_truth(h_true[idx], ypk[idx])

    rmse_map = {
        "FNO": rmse_fno,
        "FNO + 3S": rmse_3s,
        "FNO + Pakse": rmse_pk,
    }
    mae_map = {
        "FNO": mae_fno,
        "FNO + 3S": mae_3s,
        "FNO + Pakse": mae_pk,
    }

    best_rmse_variant = min(rmse_map, key=rmse_map.get)
    best_mae_variant = min(mae_map, key=mae_map.get)

    return pd.DataFrame([{
        "Window": f"3S & Pakse overlap (same dates, h={horizon})",
        "k_3S (days)": B["k_3s"],
        "k_Pakse (days)": B["k_pk"],
        "N_overlap": int(len(idx)),
        "From": str(d_sub.min()),
        "To": str(d_sub.max()),
        "RMSE (FNO)": round(rmse_fno, 3),
        "RMSE (FNO+3S)": round(rmse_3s, 3),
        "RMSE (FNO+Pakse)": round(rmse_pk, 3),
        "MAE (FNO)": round(mae_fno, 3),
        "MAE (FNO+3S)": round(mae_3s, 3),
        "MAE (FNO+Pakse)": round(mae_pk, 3),
        "Best RMSE variant": best_rmse_variant,
        "Best MAE variant": best_mae_variant,
    }])

def ui_availability_summary(horizon=1):
    """
    Summarize source availability under the fitted lag-k settings.
    """
    B = _prepare_assist_eval_bundle(horizon=horizon)
    if "error" in B:
        return pd.DataFrame({"message": [B["error"]]})

    n_total = int(len(B["df"]))

    has_3s = B["has_3s"] if B["params_3s"] is not None else np.zeros(n_total, dtype=bool)
    has_pk = B["has_pk"] if B["params_pk"] is not None else np.zeros(n_total, dtype=bool)

    n_3s = int(has_3s.sum())
    n_pk = int(has_pk.sum())
    n_both = int((has_3s & has_pk).sum())
    n_3s_only = int((has_3s & ~has_pk).sum())
    n_pk_only = int((has_pk & ~has_3s).sum())
    n_neither = int((~has_3s & ~has_pk).sum())

    return pd.DataFrame([{
        "Window": f"Availability summary (h={horizon})",
        "Horizon": int(horizon),
        "Total backtest days": n_total,
        "k_3S (days)": B["k_3s"],
        "k_Pakse (days)": B["k_pk"],
        "3S available days": n_3s,
        "Pakse available days": n_pk,
        "Both available days": n_both,
        "3S only days": n_3s_only,
        "Pakse only days": n_pk_only,
        "Neither available days": n_neither,
        "3S availability rate (%)": round(100.0 * n_3s / n_total, 1) if n_total else None,
        "Pakse availability rate (%)": round(100.0 * n_pk / n_total, 1) if n_total else None,
        "Overlap rate (%)": round(100.0 * n_both / n_total, 1) if n_total else None,
    }])

def ui_operational_routing_summary(horizon=1):
    """
    Evaluate an availability-aware routing policy:

      Pakse available -> use FNO + Pakse
      else if 3S available -> use FNO + 3S
      else -> fallback to base FNO
    """
    B = _prepare_assist_eval_bundle(horizon=horizon)
    if "error" in B:
        return pd.DataFrame({"message": [B["error"]]})

    h_true = B["h_true"]
    h_pred = B["h_pred"]
    routed = h_pred.copy()

    n_total = int(len(B["df"]))
    has_3s = B["has_3s"] if B["params_3s"] is not None else np.zeros(n_total, dtype=bool)
    has_pk = B["has_pk"] if B["params_pk"] is not None else np.zeros(n_total, dtype=bool)

    # start with 3S fallback over base FNO
    use_3s = has_3s.copy()
    if B["y_corr_3s"] is not None:
        routed[use_3s] = np.asarray(B["y_corr_3s"], dtype=float)[use_3s]

    # Pakse takes precedence over 3S
    use_pk = has_pk.copy()
    if B["y_corr_pk"] is not None:
        routed[use_pk] = np.asarray(B["y_corr_pk"], dtype=float)[use_pk]

    # final routing source counts
    src = np.full(n_total, "FNO", dtype=object)
    src[use_3s] = "FNO + 3S"
    src[use_pk] = "FNO + Pakse"  # overwrite priority

    use_pk_n = int((src == "FNO + Pakse").sum())
    use_3s_n = int((src == "FNO + 3S").sum())
    fallback_n = int((src == "FNO").sum())

    rmse_fno = _rmse_against_truth(h_true, h_pred)
    rmse_routed = _rmse_against_truth(h_true, routed)
    mae_fno = _mae_against_truth(h_true, h_pred)
    mae_routed = _mae_against_truth(h_true, routed)

    return pd.DataFrame([{
        "Routing policy": "Pakse > 3S > FNO fallback",
        "Horizon": int(horizon),
        "N_total": n_total,
        "RMSE (FNO full)": round(rmse_fno, 3) if rmse_fno is not None else None,
        "RMSE (routed)": round(rmse_routed, 3) if rmse_routed is not None else None,
        "ΔRMSE": round(rmse_routed - rmse_fno, 3) if (rmse_fno is not None and rmse_routed is not None) else None,
        "MAE (FNO full)": round(mae_fno, 3) if mae_fno is not None else None,
        "MAE (routed)": round(mae_routed, 3) if mae_routed is not None else None,
        "ΔMAE": round(mae_routed - mae_fno, 3) if (mae_fno is not None and mae_routed is not None) else None,
        "Use Pakse days": use_pk_n,
        "Use 3S days": use_3s_n,
        "Fallback FNO days": fallback_n,
        "Use Pakse rate (%)": round(100.0 * use_pk_n / n_total, 1) if n_total else None,
        "Use 3S rate (%)": round(100.0 * use_3s_n / n_total, 1) if n_total else None,
        "Fallback FNO rate (%)": round(100.0 * fallback_n / n_total, 1) if n_total else None,
    }])

# =============================================================================
# Service lifecycle (manual reload)
# =============================================================================
def ui_reload_service():
    """
    Gradio callback: force reload model & data into the process-level cache.

    Returns:
        str: Status string including reload timestamp and latest available date.
    """
    t0 = time.perf_counter()

    # 1) sync dataset artifacts first
    if DATASET_REPO:
        try:
            sync_status_from_dataset(DATASET_REPO, LAYOUT.artifacts, token=HF_READ_TOKEN)
            sync_backfill_from_dataset(DATASET_REPO, BACKFILL_P, token=HF_READ_TOKEN)
        except Exception as e:
            print("[sync][warn] reload sync failed:", repr(e))

    # 2) then force reload so the new backfill is actually used
    S = _load_service(force_reload=True)

    water_daily = S.get("water_daily")
    try:
        last_day = max(water_daily.index) if water_daily is not None and len(water_daily) > 0 else None
    except Exception:
        last_day = None

    now = pd.Timestamp.now(tz=ZoneInfo("Asia/Bangkok"))
    dt_str = now.strftime("%Y-%m-%d %H:%M")

    msg = f"Reloaded at {dt_str} (UTC+07); latest water_daily date = {last_day}"
    elapsed = time.perf_counter() - t0
    msg += f" | reload took {elapsed:.2f} s"

    return msg

# =============================================================================
# Gradio app construction (layout + wiring)
# =============================================================================
def build_app():
    """
    Construct the Gradio Blocks UI and wire callbacks.

    Design:
    - Layout is declarative (components created in one place).
    - Business logic lives in ui_* callbacks.
    - Heavy loading happens inside `_load_service()` (explicit warm-up in __main__).

    Returns:
        gr.Blocks: Fully wired application.
    """
    with gr.Blocks(title="Mekong FNO Demo") as demo:
        gr.Markdown(
            "### Mekong Water Level Forecast (Stung Treng) — FNO\n"
            "- Tab1: **Forecast Today → +7 days** (optional uncertainty)\n"
            "- Tab2: **Backtest since 2025-01-01 and ΔRMSE alignment evaluation**"
        )

        # ---------------------------------------------------------------------
        # Global controls (shared by both tabs)
        # ---------------------------------------------------------------------
        with gr.Row():
            reload_btn = gr.Button("Reload data/model", variant="secondary")
            reload_note = gr.Markdown()
        reload_btn.click(fn=ui_reload_service, inputs=None, outputs=reload_note)

        # ---------------------------------------------------------------------
        # Tabs
        # ---------------------------------------------------------------------
        with gr.Tabs():
            # =================================================================
            # Tab1: Forecast (Today → +7 days)
            # =================================================================
            with gr.Tab("Forecast (Today → +7 days)"):
                with gr.Row():
                    btn = gr.Button("Forecast +7 Days (UTC+07)", variant="primary")
                    ck = gr.Checkbox(value=False, label="Show uncertainty (Residuals/MC)")
                    src = gr.Radio(
                        choices=["Historical residuals (fast)", "MC Dropout (slow)"],
                        value="Historical residuals (fast)",
                        label="Uncertainty source",
                    )
                    samp = gr.Slider(10, 100, value=30, step=5, label="MC samples", interactive=True)

                out_plot = gr.Plot()
                out_note = gr.Markdown()
                out_df = gr.Dataframe(headers=["date", "h_pred", "p10", "p90"], interactive=False)

                # Dataflow: (ck, src, samp) -> ui_predict_today -> (plot, note, table)
                btn.click(fn=ui_predict_today, inputs=[ck, src, samp], outputs=[out_plot, out_note, out_df])

            # =================================================================
            # Tab2: Evaluation (2025 YTD & ΔRMSE)
            # =================================================================
            with gr.Tab("Evaluation (Backtest since 2025-01-01 & ΔRMSE)"):
                # shared horizon selector for backtest/compare
                with gr.Row():
                    h_sel = gr.Slider(1, 7, value=1, step=1, label="Backtest horizon (days ahead)", interactive=True)

                cmp_mode = gr.Radio(
                    choices=[
                        "Observed vs Persistence",
                        "Observed vs FNO",
                        "Observed vs FNO + 3S",
                        "Observed vs FNO + Pakse",
                        "Observed vs FNO + Pakse vs Persistence",
                        "All model variants",
                    ],
                    value="Observed vs FNO + Pakse vs Persistence",
                    label="Comparison mode",
                )

                use_custom_layers = gr.Checkbox(
                    value=False,
                    label="Use custom layer selection",
                )

                with gr.Accordion("Advanced layers", open=False):
                    custom_layers = gr.CheckboxGroup(
                        choices=EVAL_LAYER_CHOICES,
                        value=DEFAULT_EVAL_LAYERS,
                        label="Visible curves / thresholds",
                    )
                    gr.Markdown(
                        "When **Use custom layer selection** is checked, the plot ignores "
                        "**Comparison mode** and uses the selected layers here."
                    )

                # ----------------- YTD backtest -----------------
                with gr.Row():
                    btn_bt = gr.Button("Run backtest from 2025-01-01 (k-day ahead)", variant="primary")

                ytd_plot = gr.Plot()
                ytd_note = gr.Markdown()

                gr.Markdown("### Metrics summary")
                ytd_metrics = gr.Dataframe(
                    headers=["Model", "RMSE (m)"],
                    interactive=False,
                )

                with gr.Accordion("Backtest rows", open=False):
                    ytd_df = gr.Dataframe(interactive=False)

                # Dataflow: h_sel -> ui_eval_ytd -> (plot, note, metrics_summary, raw_df)
                bt_evt = btn_bt.click(
                    fn=ui_eval_ytd,
                    inputs=[h_sel, cmp_mode, use_custom_layers, custom_layers],
                    outputs=[ytd_plot, ytd_note, ytd_metrics, ytd_df],
                )

                # ----------------- Advanced diagnostics: availability-aware evaluation -----------------
                with gr.Accordion("Advanced diagnostics (availability-aware)", open=False):
                    gr.Markdown(
                        "These tables separate **source-specific gain**, **fair same-date comparison**, "
                        "**data availability**, and **real routing performance**."
                    )

                    with gr.Tabs():
                        # ----------------- Existing: 3S-only subset -----------------
                        with gr.Tab("3S-only subset"):
                            gr.Markdown("### FNO vs FNO + 3S (only where 3S is available)")
                            with gr.Row():
                                btn_cmp = gr.Button(
                                    "Compare on 3S-available dates (RMSE & MAE)",
                                    variant="secondary",
                                )
                            cmp_tbl = gr.Dataframe(interactive=False)

                        # ----------------- Existing: Pakse-only subset -----------------
                        with gr.Tab("Pakse-only subset"):
                            gr.Markdown("### FNO vs FNO + Pakse (only where Pakse is available)")
                            with gr.Row():
                                btn_cmp_pk = gr.Button(
                                    "Compare on Pakse-available dates (RMSE & MAE)",
                                    variant="secondary",
                                )
                            pk_tbl = gr.Dataframe(interactive=False)

                        # ----------------- common overlap -----------------
                        with gr.Tab("Common overlap"):
                            gr.Markdown("### Fair comparison on dates where both 3S and Pakse are available")
                            with gr.Row():
                                btn_cmp_overlap = gr.Button(
                                    "Compare on common overlap dates",
                                    variant="secondary",
                                )
                            overlap_tbl = gr.Dataframe(interactive=False)

                        # ----------------- availability -----------------
                        with gr.Tab("Availability"):
                            gr.Markdown("### Source availability summary")
                            gr.Markdown(
                                "> Availability here means: under the fitted lag-k setting, the lag-aligned upstream date exists "
                                "in the source series and its raw value is finite."
                            )
                            with gr.Row():
                                btn_avail = gr.Button(
                                    "Summarize source availability",
                                    variant="secondary",
                                )
                            avail_tbl = gr.Dataframe(interactive=False)

                        # ----------------- routing -----------------
                        with gr.Tab("Routing"):
                            gr.Markdown("### Operational routing performance")
                            with gr.Row():
                                btn_route = gr.Button(
                                    "Evaluate routed operational performance",
                                    variant="secondary",
                                )
                            route_tbl = gr.Dataframe(interactive=False)

                # ---------- manual refresh ----------
                btn_cmp.click(fn=ui_compare_fno_vs_3s_window, inputs=h_sel, outputs=cmp_tbl)
                btn_cmp_pk.click(fn=ui_compare_fno_vs_pakse_window, inputs=h_sel, outputs=pk_tbl)
                btn_cmp_overlap.click(fn=ui_compare_common_overlap_window, inputs=h_sel, outputs=overlap_tbl)
                btn_avail.click(fn=ui_availability_summary, inputs=h_sel, outputs=avail_tbl)
                btn_route.click(fn=ui_operational_routing_summary, inputs=h_sel, outputs=route_tbl)

                # ---------- auto refresh after main backtest ----------
                bt_evt.then(fn=ui_compare_fno_vs_3s_window, inputs=h_sel, outputs=cmp_tbl)
                bt_evt.then(fn=ui_compare_fno_vs_pakse_window, inputs=h_sel, outputs=pk_tbl)
                bt_evt.then(fn=ui_compare_common_overlap_window, inputs=h_sel, outputs=overlap_tbl)
                bt_evt.then(fn=ui_availability_summary, inputs=h_sel, outputs=avail_tbl)
                bt_evt.then(fn=ui_operational_routing_summary, inputs=h_sel, outputs=route_tbl)

                # manual refresh
                btn_cmp_pk.click(fn=ui_compare_fno_vs_pakse_window, inputs=h_sel, outputs=pk_tbl)
                # automatically refresh once after the backtest completes
                bt_evt.then(fn=ui_compare_fno_vs_pakse_window, inputs=h_sel, outputs=pk_tbl)

                gr.Markdown("---")

                # ----------------- Phase report table -----------------
                scope = gr.Radio(choices=["Merged", "Dry", "Wet"], value="Merged", label="Select window")
                tbl = gr.Dataframe(interactive=False)
                scope.change(fn=ui_phase_table, inputs=scope, outputs=tbl)
                gr.Markdown(
                    "> Note: scan the optimal phase shift k* on 2023, then fix it on the corresponding 2024 windows. "
                    "Shows RMSE before/after alignment and ΔRMSE."
                )
    return demo

# =============================================================================
# Entrypoint (__main__): warmup + launch
# =============================================================================
if __name__ == "__main__":
    # Runtime layout visibility (useful in HF logs / debugging storage behavior).
    print(f"[runtime] root={LAYOUT.root}")
    print(f"[runtime] cache={LAYOUT.cache}")
    print(f"[runtime] artifacts={LAYOUT.artifacts}")

    if DATASET_REPO:
        try:
            sync_status_from_dataset(DATASET_REPO, LAYOUT.artifacts, token=HF_READ_TOKEN)
            sync_backfill_from_dataset(DATASET_REPO, BACKFILL_P, token=HF_READ_TOKEN)
        except Exception as e:
            print("[sync][warn] startup sync failed:", repr(e))

    # Warm-up: preload model + data to reduce first-request latency.
    _load_service()

    app = build_app()
    app.launch(server_name="0.0.0.0",
               server_port=7860,
               theme=gr.themes.Soft(),
               )
