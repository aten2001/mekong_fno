# scripts/make_residual_band.py
import os, json, glob
import numpy as np
import pandas as pd
import tensorflow as tf

from src.runner import TenYearUnifiedRunner
from src.model_fno import SeasonalFNO1D

ART_DIR = "artifacts"
SEQ_LENGTH, PRED_LENGTH = 120, 7
CSV_DIR = os.environ.get("CSV_DIR", ".")
CLIM_PATH = os.path.join(ART_DIR, "clim_vec.npy")
NORM_PATH = os.path.join(ART_DIR, "norm_stats.json")
WEIGHTS_DIR = "weights"
OUT_PATH = os.path.join(ART_DIR, "residual_sigma.json")

def _find_ckpt():
    ckpt = tf.train.latest_checkpoint(WEIGHTS_DIR)
    if ckpt: return ckpt
    idx = glob.glob(os.path.join(WEIGHTS_DIR, "*.ckpt.index"))
    if idx:  return idx[0].replace(".index","")
    raise FileNotFoundError("No TF checkpoint in 'weights/'")

def _norm_inputs_like_train(X, st):
    Xn = X.copy()
    Xn[:, :, 0] = (Xn[:, :, 0] - st['t_mean']) / (st['t_std'] + 1e-8)
    Xn[:, :, 2] = (Xn[:, :, 2] - st['h_in_mean']) / (st['h_in_std'] + 1e-8)
    Xn[:, :, 3] = (Xn[:, :, 3] - st['dh_in_mean']) / (st['dh_in_std'] + 1e-8)
    return Xn

def _split_indices(runner, tgt_dates):
    td = pd.to_datetime(tgt_dates).normalize()
    yd = lambda y, m, d: runner._year_window_indices(td, y, m, d, length_days=60)
    return dict(
        val_all=np.sort(np.concatenate([
            yd(runner.val_year, *runner.dry_start_month_day),
            yd(runner.val_year, *runner.wet_start_month_day),
        ])),
        tst_dry=yd(runner.test_year, *runner.dry_start_month_day),
        tst_wet=yd(runner.test_year, *runner.wet_start_month_day),
    )

def _predict_7_abs_target7(model, Xn, dates, clim_vec, st):
    """
    return an (N, 7) sequence of absolute water levels;
    add the 7th-day DOY climatology to all 7 steps (consistent with the app)
    """
    from src.dataio import doy_no_leap_vec
    y_pred_n = model.predict(Xn, verbose=0)          # (N,7,1) standardized anomaly
    y_anom   = (y_pred_n * st['h_std'] + st['h_mean'])[:, :, 0]  # (N,7)
    tgt_doy  = doy_no_leap_vec(pd.to_datetime(dates))            # (N,)
    clim_add = clim_vec[tgt_doy][:, None]                        # (N,1)
    y_abs    = y_anom + clim_add                                 # (N,7)
    return y_abs

def main():
    runner = TenYearUnifiedRunner(CSV_DIR, seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH)
    data = runner.load_range_data(2015, 2025, allow_missing_u=True)
    X, Y_abs, _, tgt_dates = runner.prepare_sequences_no_season(data)

    clim = np.load(CLIM_PATH)
    st = json.load(open(NORM_PATH, "r", encoding="utf-8"))
    model = SeasonalFNO1D(modes=64, width=96, num_layers=4, input_features=X.shape[2], dropout_rate=0.1, l2=1e-5)
    _ = model(np.zeros((1, SEQ_LENGTH, X.shape[2]), dtype=np.float32), training=False)
    model.load_weights(_find_ckpt())

    idxs = _split_indices(runner, tgt_dates)
    # use 2023 (merged) plus 2024 (dry/wet)
    # together for the statistics to make them more stable
    idx_all = np.sort(np.concatenate([idxs["val_all"], idxs["tst_dry"], idxs["tst_wet"]]))
    dates = pd.to_datetime(np.asarray(tgt_dates)[idx_all]).normalize()
    Xn = _norm_inputs_like_train(X[idx_all], st)
    y_pred_abs = _predict_7_abs_target7(model, Xn, dates, clim, st)   # (N,7)
    y_true_abs = Y_abs[idx_all][:, :, 0]                               # (N,7)

    resid = y_true_abs - y_pred_abs                                    # (N,7)
    sigma_step = np.std(resid, axis=0, ddof=1).astype(float).tolist()
    sigma_all  = float(np.std(resid, ddof=1))

    os.makedirs(ART_DIR, exist_ok=True)
    out = dict(
        by_horizon=sigma_step,
        overall=sigma_all,
        n=int(len(idx_all)),
        note="std of residuals using 2023(val_all)+2024(dry,wet); target7 climatology mode"
    )
    json.dump(out, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Saved:", OUT_PATH, "\nσ_step:", sigma_step, "overall:", sigma_all)

if __name__ == "__main__":
    main()
