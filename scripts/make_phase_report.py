# scripts/make_phase_report.py
import os, json, glob
import numpy as np
import pandas as pd
import tensorflow as tf

from src.runner import TenYearUnifiedRunner
from src.model_fno import SeasonalFNO1D
from src.time_features import doy_no_leap_vec

ART_DIR = "artifacts"
SEQ_LENGTH, PRED_LENGTH = 120, 7
CSV_DIR = os.environ.get("CSV_DIR", "../tests")
CLIM_PATH = os.path.join(ART_DIR, "clim_vec.npy")
NORM_PATH = os.path.join(ART_DIR, "norm_stats.json")
WEIGHTS_DIR = "weights"

def _find_ckpt():
    """Resolve the TensorFlow checkpoint to load from ``WEIGHTS_DIR``.

    Returns:
      str: The checkpoint prefix path (i.e., without the ``.index`` suffix).

    Raises:
      FileNotFoundError: If no checkpoint file can be found under ``WEIGHTS_DIR``.
    """
    ckpt = tf.train.latest_checkpoint(WEIGHTS_DIR)
    if ckpt: return ckpt
    idx = glob.glob(os.path.join(WEIGHTS_DIR, "*.ckpt.index"))
    if idx:  return idx[0].replace(".index","")
    raise FileNotFoundError("No TF checkpoint found in 'weights/'")

def _norm_inputs_like_train(X, st):
    """
    Apply training-time normalization to input features.

    Normalizes channels by z-score using training statistics:
      - channel 0 (time_idx) → (x - t_mean) / (t_std + 1e-8)
      - channel 2 (h)        → (x - h_in_mean) / (h_in_std + 1e-8)
      - channel 3 (dh1)      → (x - dh_in_mean) / (dh_in_std + 1e-8)

    Args:
      X (np.ndarray): shape (N, L, C). Raw input tensor.
      st (dict): contains the keys
          ``{'t_mean','t_std','h_in_mean','h_in_std','dh_in_mean','dh_in_std'}``.

    Returns:
      np.ndarray: Normalized copy of ``X`` with the same shape.
    """
    Xn = X.copy()
    Xn[:, :, 0] = (Xn[:, :, 0] - st['t_mean']) / (st['t_std'] + 1e-8)
    Xn[:, :, 2] = (Xn[:, :, 2] - st['h_in_mean']) / (st['h_in_std'] + 1e-8)
    Xn[:, :, 3] = (Xn[:, :, 3] - st['dh_in_mean']) / (st['dh_in_std'] + 1e-8)
    return Xn

# Forward pass: standardized anomaly → de-standardize → add target-day DOY climatology;
# return Day-7 absolute WL
def _fno_predict7(model, Xn, dates, clim_vec, st):
    """
    Compute day-7 absolute water level from normalized inputs.

    Args:
      model (tf.keras.Model): ``tf.keras.Model`` (FNO) with output shape (N, 7, 1) in anomaly units (standardized).
      Xn (np.ndarray):  shape (N, L, C). Normalized inputs.
      dates (date.datetime): Sequence of datetimes (length N), target dates per sample.
      clim_vec (np.ndarray): climatology vector indexed by DOY (no-leap indexing).
      st (dict): dict with keys ``{'h_mean','h_std'}`` for de-standardization.

    Returns:
      np.ndarray: Shape (N,), absolute water-level predictions at horizon=7.

    Notes:
      Uses ``doy_no_leap_vec`` so that Feb 29 is skipped/compacted consistently with training.
    """
    y_pred_n = model.predict(Xn, verbose=0)
    y_anom   = y_pred_n * st['h_std'] + st['h_mean']
    tgt_doy  = doy_no_leap_vec(pd.to_datetime(dates))
    clim_add = clim_vec[tgt_doy][:, None, None]
    y_abs    = y_anom + clim_add
    return y_abs[:, -1, 0]

def _split_indices(runner, tgt_dates):
    """
    Build validation/test window indices for dry/wet seasons.

    Args:
      runner (TenYearUnifiedRunner): ``TenYearUnifiedRunner`` that provides date utilities and season anchors.
      tgt_dates: Sequence-like of target dates for each sample.

    Returns:
      dict[str, np.ndarray]: Keys:
        - ``'val_all'``: merged validation (dry ∪ wet, 2023).
        - ``'val_dry'``: validation dry window (2023).
        - ``'val_wet'``: validation wet window (2023).
        - ``'tst_dry'``: test dry window (2024).
        - ``'tst_wet'``: test wet window (2024).
    """
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

# RMSE under an integer time shift k; overlap < 10 samples returns NaN
def _rmse_with_shift(y_true, y_pred, k):
    """Compute RMSE after applying an integer time shift.

    Args:
      y_true: Array-like, ground-truth sequence.
      y_pred: Array-like, predicted sequence (same length as ``y_true`` before shift).
      k: int, time shift in samples (days).

    Returns:
      float: RMSE computed on the overlapped portion, or ``np.nan`` if
      the overlap has fewer than 10 points.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    if k > 0:  yt, yp = y_true[:-k], y_pred[k:]
    elif k < 0: s=-k; yt, yp = y_true[s:], y_pred[:-s]
    else:      yt, yp = y_true, y_pred
    if len(yt) < 10: return np.nan
    return float(np.sqrt(np.mean((yp-yt)**2)))

def main():
    """
    Generate phase-alignment report by scanning time shifts on 2023 and applying to 2024.

    Workflow:
      1) Load samples (2015–2025), climatology, normalization stats, and FNO weights.
      2) Build validation windows (2023: dry/wet/merged) and test windows (2024: dry/wet).
      3) For each 2023 window, scan integer shift ``k ∈ [-10, 10]`` to minimize RMSE.
      4) Apply selected ``k*`` to the corresponding 2024 window(s) and compute RMSE gain.
      5) Save a JSON report to ``artifacts/phase_report.json`` with:
         - ``val``: best ``k`` and RMSE before/after for {all, dry, wet}
         - ``test_applied``: fixed ``k`` from val and RMSE before/after (plus gain)
           for {dry, wet, all(merged)}

    Returns:
      None

    Raises:
      FileNotFoundError: If a required checkpoint under ``weights/`` cannot be found.
    """
    runner = TenYearUnifiedRunner(CSV_DIR, seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH)
    data = runner.load_range_data(2015, 2025, allow_missing_u=True)
    X, Y, _, tgt_dates = runner.prepare_sequences_no_season(data)

    clim = np.load(CLIM_PATH)
    st = json.load(open(NORM_PATH, "r", encoding="utf-8"))
    model = SeasonalFNO1D(modes=64, width=96, num_layers=4, input_features=X.shape[2], dropout_rate=0.1, l2=1e-5)
    _ = model(np.zeros((1, SEQ_LENGTH, X.shape[2]), dtype=np.float32), training=False)
    model.load_weights(_find_ckpt())

    ids = _split_indices(runner, tgt_dates)
    def series(idx):
        d = pd.to_datetime(np.asarray(tgt_dates)[idx]).normalize()
        yt = Y[idx][:, -1, 0]
        yp = _fno_predict7(model, _norm_inputs_like_train(X[idx], st), d, clim, st)
        # Sort by time
        order = np.argsort(d.values)
        return yt[order], yp[order]

    report = {"val":{}, "test_applied":{}}
    # 1) On 2023 windows (merged/dry/wet), scan k in [-10, 10]
    for key_src, key_dst in [("val_all","all"),("val_dry","dry"),("val_wet","wet")]:
        yt, yp = series(ids[key_src])
        base = _rmse_with_shift(yt, yp, 0)
        best_k, best_rmse = 0, base
        for k in range(-10, 11):
            r = _rmse_with_shift(yt, yp, k)
            if np.isnan(r): continue
            if r < best_rmse: best_rmse, best_k = r, k
        report["val"][key_dst] = {"best_k":int(best_k), "base_rmse":float(base), "aligned_rmse":float(best_rmse), "gain":float(base-best_rmse)}

    # 2) Apply the selected k* to the corresponding 2024 windows
    for key_src, key_dst in [("tst_dry","dry"),("tst_wet","wet")]:
        yt, yp = series(ids[key_src])
        k = report["val"][key_dst]["best_k"]
        base = _rmse_with_shift(yt, yp, 0)
        aligned = _rmse_with_shift(yt, yp, k)
        report["test_applied"][key_dst] = {"k":int(k), "rmse_before":float(base), "rmse_after":float(aligned), "gain":float(base-aligned)}
    # merge（dry+wet）
    yt_d, yp_d = series(ids["tst_dry"])
    yt_w, yp_w = series(ids["tst_wet"])
    yt_all = np.concatenate([yt_d, yt_w]); yp_all = np.concatenate([yp_d, yp_w])
    k_all = report["val"]["all"]["best_k"]
    base = _rmse_with_shift(yt_all, yp_all, 0)
    aligned = _rmse_with_shift(yt_all, yp_all, k_all)
    report["test_applied"]["all"] = {"k":int(k_all), "rmse_before":float(base), "rmse_after":float(aligned), "gain":float(base-aligned)}

    os.makedirs(ART_DIR, exist_ok=True)
    out = os.path.join(ART_DIR, "phase_report.json")
    json.dump(report, open(out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Saved:", out)

if __name__ == "__main__":
    main()
