# scripts/make_residual_band.py
import os, json, glob
import numpy as np
import pandas as pd
import tensorflow as tf

from src.runner import TenYearUnifiedRunner
from src.model_fno import SeasonalFNO1D

ART_DIR = "artifacts"
SEQ_LENGTH, PRED_LENGTH = 120, 7
CSV_DIR = os.environ.get("CSV_DIR", "data")
CLIM_PATH = os.path.join(ART_DIR, "clim_vec.npy")
NORM_PATH = os.path.join(ART_DIR, "norm_stats.json")
WEIGHTS_DIR = "weights"
OUT_PATH = os.path.join(ART_DIR, "residual_sigma.json")

def _find_ckpt():
    """
    Resolve a TensorFlow checkpoint under ``WEIGHTS_DIR``.

    Returns:
      str: Checkpoint prefix path (without the ``.index`` suffix).

    Raises:
      FileNotFoundError: If no checkpoint exists under ``WEIGHTS_DIR``.
    """
    ckpt = tf.train.latest_checkpoint(WEIGHTS_DIR)
    if ckpt: return ckpt
    idx = glob.glob(os.path.join(WEIGHTS_DIR, "*.ckpt.index"))
    if idx:  return idx[0].replace(".index","")
    raise FileNotFoundError("No TF checkpoint in 'weights/'")

def _norm_inputs_like_train(X, st):
    """Apply training-time normalization to inputs.

    The following channels are z-scored with stored training statistics:
      - channel 0 (time_idx): ``(x - t_mean) / (t_std + 1e-8)``
      - channel 2 (h): ``(x - h_in_mean) / (h_in_std + 1e-8)``
      - channel 3 (dh1): ``(x - dh_in_mean) / (dh_in_std + 1e-8)``

    Args:
      X (np.ndarray): Input tensor of shape (N, L, C).
      st (dict): Normalization stats containing keys
        ``{'t_mean','t_std','h_in_mean','h_in_std','dh_in_mean','dh_in_std'}``.

    Returns:
      np.ndarray: Normalized copy of ``X`` with the same shape.

    Notes:
      Channels other than {0, 2, 3} are passed through unchanged.
    """
    Xn = X.copy()
    Xn[:, :, 0] = (Xn[:, :, 0] - st['t_mean']) / (st['t_std'] + 1e-8)
    Xn[:, :, 2] = (Xn[:, :, 2] - st['h_in_mean']) / (st['h_in_std'] + 1e-8)
    Xn[:, :, 3] = (Xn[:, :, 3] - st['dh_in_mean']) / (st['dh_in_std'] + 1e-8)
    return Xn

def _split_indices(runner, tgt_dates):
    """Build evaluation index windows for residual-band fitting.

    Creates three index sets:
      - ``val_all``: 2023 validation, dry ∪ wet (merged), 60 days each, concatenated and sorted.
      - ``tst_dry``: 2024 test dry window, 60 days.
      - ``tst_wet``: 2024 test wet window, 60 days.

    Args:
      runner (TenYearUnifiedRunner): Provides date utilities and season anchors.
      tgt_dates (Sequence): Target dates for each sample (array-like, pandas-compatible).

    Returns:
      dict[str, np.ndarray]: Dictionary with keys ``'val_all'``, ``'tst_dry'``, ``'tst_wet'``,
      whose values are integer index arrays referring into the sample dimension.

    Notes:
      Windows are constructed via ``runner._year_window_indices`` on normalized datetimes.
    """
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
    """Predict absolute WL for horizons 1..7 using **target-7 climatology mode**.

    The model outputs standardized anomalies for 7 horizons. This function:
      1) Runs a forward pass to obtain standardized anomalies ``(N, 7, 1)``.
      2) De-standardizes anomalies using ``st['h_mean']`` and ``st['h_std']``.
      3) Computes DOY from the **7th-day target** and adds that same
         climatology value to **all 7 horizons** (consistent with the app).

    Args:
      model (tf.keras.Model): FNO model whose output is (N, 7, 1) standardized anomaly.
      Xn (np.ndarray): Normalized inputs of shape (N, L, C).
      dates (Sequence): Target dates per sample (length N).
      clim_vec (np.ndarray): Climatology vector indexed by no-leap DOY.
      st (dict): Dict with keys ``{'h_mean','h_std'}`` for de-standardization.

    Returns:
      np.ndarray: Shape (N, 7), absolute water-level predictions for horizons 1..7.

    Notes:
      Uses ``doy_no_leap_vec`` (no-leap DOY) to match training behavior (skips Feb 29).
      This “target7 climatology for all steps” matches the app’s visualization logic.
    """
    from src.time_features import doy_no_leap_vec
    y_pred_n = model.predict(Xn, verbose=0)          # (N,7,1) standardized anomaly
    y_anom   = (y_pred_n * st['h_std'] + st['h_mean'])[:, :, 0]  # (N,7)
    tgt_doy  = doy_no_leap_vec(pd.to_datetime(dates))            # (N,)
    clim_add = clim_vec[tgt_doy][:, None]                        # (N,1)
    y_abs    = y_anom + clim_add                                 # (N,7)
    return y_abs

def main():
    """Compute residual-band sigmas (per-horizon and overall) and persist to JSON.

    Workflow:
      1) Load sequences (2015–2025) and build samples with the runner.
      2) Load climatology vector, normalization stats, and FNO weights.
      3) Construct evaluation indices: 2023 merged (val_all) and 2024 dry/wet (tst_dry/tst_wet).
      4) Predict absolute WL for horizons 1..7 using **target-7 climatology mode**.
      5) Compute residuals ``(y_true_abs - y_pred_abs)`` and estimate:
         - ``by_horizon``: std across samples for each horizon (length 7).
         - ``overall``: std across all horizons and samples.
      6) Save to ``artifacts/residual_sigma.json`` with sample count and a short note.

    Returns:
      None

    Raises:
      FileNotFoundError: If no TensorFlow checkpoint can be found under ``WEIGHTS_DIR``.
    """
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
