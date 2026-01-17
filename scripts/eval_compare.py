# scripts/eval_compare.py

# --- bootstrap sys.path so running from any CWD works consistently ---
import os, sys, json, glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # <repo>/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

from src.runner import TenYearUnifiedRunner
from src.model_fno import SeasonalFNO1D
from src.baselines import predict_persistence_7th, predict_climatology_7th, predict_linear_7th
from src.metrics import rmse, mae, nse, pearson_r, peak_timing_error_days

# ---------------------------
# Stable, CWD-independent paths (match your assets structure)
# assets/
#   clim_vec.npy
#   norm_stats.json
#   reports/
#     eval_compare.csv/json ...
# ---------------------------
ASSETS_DIR  = Path(os.environ.get("ASSETS_DIR",  str(REPO_ROOT / "assets")))
REPORTS_DIR = ASSETS_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", str(REPO_ROOT / "weights")))
CSV_DIR     = Path(os.environ.get("CSV_DIR",     str(REPO_ROOT / "data")))

# IMPORTANT: per your screenshot, these live directly under assets/
CLIM_PATH = ASSETS_DIR / "clim_vec.npy"
NORM_PATH = ASSETS_DIR / "norm_stats.json"


def _find_ckpt(weights_dir=str(WEIGHTS_DIR), fallback_prefix="stung_treng_fno.ckpt"):
    """
    Find a TensorFlow checkpoint under the weights directory.

    Args:
      weights_dir (str): Directory containing checkpoint files.
      fallback_prefix: Filename prefix (without extension) to try as a last resort.

    Returns:
      str: Checkpoint prefix path (without ``.index`` suffix).

    Raises:
      FileNotFoundError: If no checkpoint can be resolved under ``weights_dir``.
    """
    ckpt = tf.train.latest_checkpoint(weights_dir)
    if ckpt:
        return ckpt

    cand = glob.glob(os.path.join(weights_dir, "*.ckpt.index"))
    if cand:
        return cand[0][:-len(".index")]

    prefix = os.path.join(weights_dir, fallback_prefix)
    if os.path.exists(prefix + ".index"):
        return prefix

    raise FileNotFoundError(f"No TF checkpoint found under: {weights_dir}")


def _fno_predict7(model, Xn, tgt_dates, clim_vec, st):
    """
    Predict day-7 absolute water level from normalized inputs.
    """
    y_pred_n = model.predict(Xn, verbose=0)                # (N,7,1)
    y_pred_anom = y_pred_n * st["h_std"] + st["h_mean"]    # (N,7,1)

    from src.time_features import doy_no_leap_vec
    tgt_doy = doy_no_leap_vec(pd.to_datetime(tgt_dates))
    clim_add = clim_vec[tgt_doy][:, None, None]            # (N,1,1)
    y_abs = y_pred_anom + clim_add                         # (N,7,1)
    return y_abs[:, -1, 0]


def _norm_inputs_like_train(X, st):
    """
    Apply training-time normalization to input tensor.
    """
    Xn = X.copy()
    Xn[:, :, 0] = (Xn[:, :, 0] - st["t_mean"]) / (st["t_std"] + 1e-8)
    Xn[:, :, 2] = (Xn[:, :, 2] - st["h_in_mean"]) / (st["h_in_std"] + 1e-8)
    Xn[:, :, 3] = (Xn[:, :, 3] - st["dh_in_mean"]) / (st["dh_in_std"] + 1e-8)
    return Xn


def _print_block(tag, yt):
    """
    Pretty-print helper for a metrics block.
    """
    print(f"\n[{tag}] n={len(yt)}")
    print("model   rmse   mae    nse    r      peak_dt")

    def row(name, a, b, dates):
        print(
            f"{name:<7} {rmse(a,b):.3f}  {mae(a,b):.3f}  {nse(a,b):.3f}  "
            f"{pearson_r(a,b):.3f}  {peak_timing_error_days(dates, a, b):+d}"
        )
    return row


def main():
    # ===== 0) sanity print (optional) =====
    print("[paths]")
    print("  REPO_ROOT  =", REPO_ROOT)
    print("  ASSETS_DIR =", ASSETS_DIR)
    print("  REPORTS_DIR=", REPORTS_DIR)
    print("  WEIGHTS_DIR=", WEIGHTS_DIR)
    print("  CSV_DIR    =", CSV_DIR)
    print("  CLIM_PATH  =", CLIM_PATH)
    print("  NORM_PATH  =", NORM_PATH)

    if not CLIM_PATH.exists():
        raise FileNotFoundError(f"Missing: {CLIM_PATH}")
    if not NORM_PATH.exists():
        raise FileNotFoundError(f"Missing: {NORM_PATH}")

    # ===== 1) Load data and build samples =====
    runner = TenYearUnifiedRunner(csv_files_path=str(CSV_DIR), seq_length=120, pred_length=7)
    data = runner.load_range_data(2015, 2025, allow_missing_u=True)
    X, Y, tgt_season, tgt_dates = runner.prepare_sequences_no_season(data)

    # ===== 2) Load clim/norm_stats/weights and build the model =====
    clim = np.load(str(CLIM_PATH))
    st = json.load(open(str(NORM_PATH), "r", encoding="utf-8"))

    model = SeasonalFNO1D(modes=64, width=96, num_layers=4, input_features=6, dropout_rate=0.1, l2=1e-5)
    _ = model(np.zeros((1, 120, 6), dtype=np.float32), training=False)
    ckpt = _find_ckpt(weights_dir=str(WEIGHTS_DIR))
    model.load_weights(ckpt)

    # ===== 3) Split indices for 2023/2024 dry/wet windows =====
    ids = {}
    dry_m1, dry_d1 = runner.dry_start_month_day
    wet_m1, wet_d1 = runner.wet_start_month_day
    ids["val_dry"] = runner._year_window_indices(tgt_dates, runner.val_year,  dry_m1, dry_d1, 60)
    ids["val_wet"] = runner._year_window_indices(tgt_dates, runner.val_year,  wet_m1, wet_d1, 60)
    ids["tst_dry"] = runner._year_window_indices(tgt_dates, runner.test_year, dry_m1, dry_d1, 60)
    ids["tst_wet"] = runner._year_window_indices(tgt_dates, runner.test_year, wet_m1, wet_d1, 60)

    # ===== 4) Evaluate comparisons per window =====
    rows = []
    for tag in ["val_dry", "val_wet", "tst_dry", "tst_wet"]:
        idx = ids[tag]
        if idx.size == 0:
            continue

        yt = Y[idx][:, -1, 0]
        Xn = _norm_inputs_like_train(X[idx], st)
        dates = pd.to_datetime(tgt_dates[idx])

        yp_fno   = _fno_predict7(model, Xn, dates, clim, st)
        yp_pers  = predict_persistence_7th(X[idx])
        yp_clim  = predict_climatology_7th(dates, clim)
        yp_linear= predict_linear_7th(X[idx], lookback=14)

        rprint = _print_block(tag, yt)  # FIX: _print_block only takes (tag, yt)
        rprint("FNO",    yt, yp_fno,   dates)
        rprint("Pers",   yt, yp_pers,  dates)
        rprint("Clim",   yt, yp_clim,  dates)
        rprint("Linear", yt, yp_linear, dates)

        def rec(model_name, yp):
            rows.append({
                "window": tag,
                "model": model_name,
                "rmse": rmse(yt, yp),
                "mae": mae(yt, yp),
                "nse": nse(yt, yp),
                "r": pearson_r(yt, yp),
                "peak_dt": int(peak_timing_error_days(dates, yt, yp)),
                "n": int(len(yt)),
            })

        rec("FNO", yp_fno)
        rec("Pers", yp_pers)
        rec("Clim", yp_clim)
        rec("Linear", yp_linear)

    # ===== 5) 2024 weighted RMSE/MAE (dry/wet weighted by sample counts) =====
    df = pd.DataFrame(rows)

    def w_rmse_mae(df_all, tag1="tst_dry", tag2="tst_wet"):
        def for_model(m):
            d1 = df_all[(df_all.window == tag1) & (df_all.model == m)].iloc[0]
            d2 = df_all[(df_all.window == tag2) & (df_all.model == m)].iloc[0]
            n1, n2 = d1["n"], d2["n"]
            mse_w = (n1 * (d1["rmse"] ** 2) + n2 * (d2["rmse"] ** 2)) / (n1 + n2)
            mae_w = (n1 * d1["mae"] + n2 * d2["mae"]) / (n1 + n2)
            return float(np.sqrt(mse_w)), float(mae_w)

        return {m: for_model(m) for m in ["Clim", "FNO", "Linear", "Pers"]}

    w = w_rmse_mae(df)
    print("\n[Test 2024 weighted RMSE]")
    for m, (wr, _) in w.items():
        print(f"{m:<6}: {wr:.3f}")
    print("\n[Test 2024 weighted MAE]")
    for m, (_, wa) in w.items():
        print(f"{m:<6}: {wa:.3f}")

    # ===== 6) Persist comparison results (to assets/reports) =====
    out_csv  = REPORTS_DIR / "eval_compare.csv"
    out_json = REPORTS_DIR / "eval_compare.json"

    df.to_csv(str(out_csv), index=False)
    with open(str(out_json), "w", encoding="utf-8") as f:
        json.dump({
            "by_window": rows,
            "test_2024_weighted": {m: {"rmse": w[m][0], "mae": w[m][1]} for m in w}
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()