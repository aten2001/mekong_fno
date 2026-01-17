# scripts/make_phase_report.py
import os, sys, json, glob
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# --- make paths stable regardless of current working directory ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runner import TenYearUnifiedRunner
from src.model_fno import SeasonalFNO1D
from src.time_features import doy_no_leap_vec

SEQ_LENGTH, PRED_LENGTH = 120, 7

# --- canonical dirs (align with app.py defaults) ---
ASSETS_DIR  = Path(os.environ.get("ASSETS_DIR",  str(REPO_ROOT / "assets")))
WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", str(REPO_ROOT / "weights")))
CSV_DIR     = os.environ.get("CSV_DIR", str(REPO_ROOT / "data"))

CLIM_PATH = ASSETS_DIR / "clim_vec.npy"
NORM_PATH = ASSETS_DIR / "norm_stats.json"


def _find_ckpt(weights_dir: Path = WEIGHTS_DIR) -> str:
    """Resolve the TensorFlow checkpoint to load from weights_dir."""
    ckpt = tf.train.latest_checkpoint(str(weights_dir))
    if ckpt:
        return ckpt
    idx = glob.glob(str(weights_dir / "*.ckpt.index"))
    if idx:
        return idx[0].replace(".index", "")
    raise FileNotFoundError(f"No TF checkpoint found under: {weights_dir}")


def _norm_inputs_like_train(X, st):
    """Apply training-time normalization to selected channels."""
    Xn = X.copy()
    Xn[:, :, 0] = (Xn[:, :, 0] - st["t_mean"]) / (st["t_std"] + 1e-8)
    Xn[:, :, 2] = (Xn[:, :, 2] - st["h_in_mean"]) / (st["h_in_std"] + 1e-8)
    Xn[:, :, 3] = (Xn[:, :, 3] - st["dh_in_mean"]) / (st["dh_in_std"] + 1e-8)
    return Xn


def _fno_predict7(model, Xn, dates, clim_vec, st):
    """Standardized anomaly → de-standardize → add target-day DOY climatology; return day-7 absolute WL."""
    y_pred_n = model.predict(Xn, verbose=0)            # (N,7,1) standardized anomaly
    y_anom   = y_pred_n * st["h_std"] + st["h_mean"]   # (N,7,1) anomaly in meters
    tgt_doy  = doy_no_leap_vec(pd.to_datetime(dates))
    clim_add = clim_vec[tgt_doy][:, None, None]        # (N,1,1)
    y_abs    = y_anom + clim_add
    return y_abs[:, -1, 0]                             # day-7


def _split_indices(runner, tgt_dates):
    """Build validation/test window indices for dry/wet seasons."""
    td = pd.to_datetime(tgt_dates).normalize()
    yd = lambda y, m, d: runner._year_window_indices(td, y, m, d, length_days=60)
    return dict(
        val_all=np.sort(np.concatenate([
            yd(runner.val_year, *runner.dry_start_month_day),
            yd(runner.val_year, *runner.wet_start_month_day),
        ])),
        val_dry=yd(runner.val_year, *runner.dry_start_month_day),
        val_wet=yd(runner.val_year, *runner.wet_start_month_day),
        tst_dry=yd(runner.test_year, *runner.dry_start_month_day),
        tst_wet=yd(runner.test_year, *runner.wet_start_month_day),
    )


def _rmse_with_shift(y_true, y_pred, k):
    """RMSE under integer time shift k; overlap < 10 samples returns NaN."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if k > 0:
        yt, yp = y_true[:-k], y_pred[k:]
    elif k < 0:
        s = -k
        yt, yp = y_true[s:], y_pred[:-s]
    else:
        yt, yp = y_true, y_pred

    if len(yt) < 10:
        return np.nan
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def main():
    # ---- sanity checks (fail fast) ----
    if not CLIM_PATH.exists():
        raise FileNotFoundError(f"Missing climatology: {CLIM_PATH}")
    if not NORM_PATH.exists():
        raise FileNotFoundError(f"Missing norm stats: {NORM_PATH}")

    runner = TenYearUnifiedRunner(CSV_DIR, seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH)
    data = runner.load_range_data(2015, 2025, allow_missing_u=True)
    X, Y, _, tgt_dates = runner.prepare_sequences_no_season(data)

    clim = np.load(str(CLIM_PATH))
    st = json.load(open(str(NORM_PATH), "r", encoding="utf-8"))

    model = SeasonalFNO1D(
        modes=64, width=96, num_layers=4,
        input_features=X.shape[2],
        dropout_rate=0.1, l2=1e-5
    )
    _ = model(np.zeros((1, SEQ_LENGTH, X.shape[2]), dtype=np.float32), training=False)
    model.load_weights(_find_ckpt())

    ids = _split_indices(runner, tgt_dates)

    def series(idx):
        d = pd.to_datetime(np.asarray(tgt_dates)[idx]).normalize()
        yt = Y[idx][:, -1, 0]
        yp = _fno_predict7(model, _norm_inputs_like_train(X[idx], st), d, clim, st)
        order = np.argsort(d.values)
        return yt[order], yp[order]

    report = {"val": {}, "test_applied": {}}

    # 1) Scan k on 2023 windows
    for key_src, key_dst in [("val_all", "all"), ("val_dry", "dry"), ("val_wet", "wet")]:
        yt, yp = series(ids[key_src])
        base = _rmse_with_shift(yt, yp, 0)
        best_k, best_rmse = 0, base
        for k in range(-10, 11):
            r = _rmse_with_shift(yt, yp, k)
            if np.isnan(r):
                continue
            if r < best_rmse:
                best_rmse, best_k = r, k
        report["val"][key_dst] = {
            "best_k": int(best_k),
            "base_rmse": float(base),
            "aligned_rmse": float(best_rmse),
            "gain": float(base - best_rmse),
        }

    # 2) Apply selected k* to 2024 windows
    for key_src, key_dst in [("tst_dry", "dry"), ("tst_wet", "wet")]:
        yt, yp = series(ids[key_src])
        k = report["val"][key_dst]["best_k"]
        base = _rmse_with_shift(yt, yp, 0)
        aligned = _rmse_with_shift(yt, yp, k)
        report["test_applied"][key_dst] = {
            "k": int(k),
            "rmse_before": float(base),
            "rmse_after": float(aligned),
            "gain": float(base - aligned),
        }

    # merged (dry+wet)
    yt_d, yp_d = series(ids["tst_dry"])
    yt_w, yp_w = series(ids["tst_wet"])
    yt_all = np.concatenate([yt_d, yt_w])
    yp_all = np.concatenate([yp_d, yp_w])
    k_all = report["val"]["all"]["best_k"]
    base = _rmse_with_shift(yt_all, yp_all, 0)
    aligned = _rmse_with_shift(yt_all, yp_all, k_all)
    report["test_applied"]["all"] = {
        "k": int(k_all),
        "rmse_before": float(base),
        "rmse_after": float(aligned),
        "gain": float(base - aligned),
    }

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out = ASSETS_DIR / "phase_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Saved:", str(out))


if __name__ == "__main__":
    main()
