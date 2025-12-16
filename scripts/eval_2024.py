# scripts/eval_2024.py
import os, json, glob, argparse, time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from src.runner import TenYearUnifiedRunner
from src.model_fno import SeasonalFNO1D
from src.baselines import predict_persistence_7th, predict_climatology_7th, predict_linear_7th
from src.metrics import rmse, mae, nse, pearson_r

SEQ_LENGTH = 120
PRED_LENGTH = 7
CSV_DIR = os.environ.get("CSV_DIR", ".")
ART_DIR = "artifacts"
WEIGHTS_DIR = "weights"
CLIM_PATH = os.path.join(ART_DIR, "clim_vec.npy")
NORM_PATH = os.path.join(ART_DIR, "norm_stats.json")

def _find_ckpt(weights_dir=WEIGHTS_DIR):
    """
    Find the latest TensorFlow checkpoint path under weights directory.

    Args:
      weights_dir (str): directory that contains TF checkpoint files.

    Returns:
      str: Absolute or relative path to the checkpoint prefix (without ``.index``).

    Raises:
      FileNotFoundError: If no checkpoint can be found in ``weights_dir``.
    """
    ckpt = tf.train.latest_checkpoint(weights_dir)
    if ckpt: return ckpt
    idx = glob.glob(os.path.join(weights_dir, "*.ckpt.index"))
    if idx:  return idx[0].replace(".index","")
    raise FileNotFoundError("No TF checkpoint found in 'weights/'")

def _norm_inputs_like_train(X, st):
    """
    Normalize input features using training-time statistics.

    Args:
      X (np.ndarray): shape (N, L, C). Raw input features.
      st (dict): training stats.

    Returns:
      np.ndarray: Same shape as ``X``, normalized copy.
    """
    Xn = X.copy()
    Xn[:, :, 0] = (Xn[:, :, 0] - st['t_mean']) / (st['t_std'] + 1e-8)
    Xn[:, :, 2] = (Xn[:, :, 2] - st['h_in_mean']) / (st['h_in_std'] + 1e-8)
    Xn[:, :, 3] = (Xn[:, :, 3] - st['dh_in_mean']) / (st['dh_in_std'] + 1e-8)
    return Xn

def _fno_predict7(model, Xn, dates, clim_vec, st):
    """
    Predict day-7 absolute water level from normalized inputs.

    Args:
      model (tf.keras.Model): the trained FNO model.
      Xn (np.ndarray): shape (N, L, C). Normalized inputs.
      dates (datetime.date): array-like of pandas-compatible datetimes, target dates (one per sample).
      clim_vec (np.ndarray): shape (366,) or (365,), climatology indexed by DOY (no leap).
      st (dict): de-standardization stats with keys {'h_mean','h_std'}.

    Returns:
      np.ndarray: shape (N,), absolute WL prediction for horizon=7 (day-7).
    """
    from src.time_features import doy_no_leap_vec
    y_pred_n = model.predict(Xn, verbose=0)                # (N,7,1) standardized anomaly
    y_anom   = y_pred_n * st['h_std'] + st['h_mean']       # (N,7,1)
    tgt_doy  = doy_no_leap_vec(pd.to_datetime(dates))
    clim_add = clim_vec[tgt_doy][:, None, None]            # (N,1,1)
    y_abs    = y_anom + clim_add                           # (N,7,1)
    return y_abs[:, -1, 0]                                 # day-7

def _split_indices(runner, tgt_dates):
    """
    Build validation/test 60-day window indices for dry/wet seasons.

    Args:
      runner (TenYearUnifiedRunner): provides year anchors and index utility.
      tgt_dates: array-like of datetimes, target dates for which to build windows.

    Returns:
      dict[str, np.ndarray]: Mapping with keys
        {'val_dry','val_wet','tst_dry','tst_wet'} to integer index arrays.
    """
    td = pd.to_datetime(tgt_dates).normalize()
    yd = lambda y, m, d: runner._year_window_indices(td, y, m, d, length_days=60)
    return dict(
        val_dry = yd(runner.val_year, *runner.dry_start_month_day),
        val_wet = yd(runner.val_year, *runner.wet_start_month_day),
        tst_dry = yd(runner.test_year, *runner.dry_start_month_day),
        tst_wet = yd(runner.test_year, *runner.wet_start_month_day),
    )

# ====== full-year plot ======
def _plot_year_all_models(X, Y, tgt_dates, model, st, clim_vec, year: int, lookback: int = 14, models_to_plot=None):
    """
    Plot a full-year day-7 curve for selected models and save the figure.

    Args:
      X (np.ndarray): shape (N, L, C). Raw inputs (not normalized).
      Y (np.ndarray): shape (N, 7, 1). Ground-truth absolute WL for 7 horizons.
      tgt_dates: array-like of datetimes, one per sample.
      model (tf.keras.Model): trained FNO model.
      st (dict): normalization/de-standardization stats (see ``_norm_inputs_like_train``).
      clim_vec (np.ndarray): climatology vector indexed by DOY (no leap).
      year (int): target calendar year to plot.
      lookback (int): lookback length used by the linear baseline.
      models_to_plot (Optional[Iterable[str]]): subset of lines to show, e.g.
        {'truth','FNO','Pers','Clim','Linear'}. If None, plots all.

    Returns:
      None. Saves the PNG to the artifacts directory as a side-effect.

    Raises:
      ValueError: If input shapes are inconsistent (rare; not explicitly checked).
    """
    td = pd.to_datetime(tgt_dates).normalize()
    idx = np.where(td.year == int(year))[0]
    if idx.size == 0:
        print(f"No samples in year {year}. Skip full-year plot.")
        return

    dates = td[idx]
    yt7   = Y[idx][:, -1, 0]
    Xn    = _norm_inputs_like_train(X[idx], st)

    # predictions
    yp_fno    = _fno_predict7(model, Xn, dates, clim_vec, st)
    yp_pers   = predict_persistence_7th(X[idx])
    yp_clim   = predict_climatology_7th(dates, clim_vec)
    yp_linear = predict_linear_7th(X[idx], lookback=lookback)

    # sort by time for clean lines
    order = np.argsort(dates.values)
    dates = dates.values[order]
    yt7   = yt7[order]
    yhat  = {
        "FNO": yp_fno[order],
        "Pers": yp_pers[order],
        "Clim": yp_clim[order],
        "Linear": yp_linear[order],
    }

    # Assemble plottable series (including truth)
    series = {
        "truth": yt7,
        "FNO": yhat["FNO"],
        "Pers": yhat["Pers"],
        "Clim": yhat["Clim"],
        "Linear": yhat["Linear"],
    }
    # Keep only the user-specified ones (default: truth/FNO/Pers)
    if models_to_plot is not None:
        series = {k: v for k, v in series.items() if k in models_to_plot}

    plt.figure(figsize=(14, 5))
    # Plot truth first (thicker solid line)
    if "truth" in series:
        plt.plot(dates, series.pop("truth"), label="Truth", linewidth=2.2)


    for name, y in series.items():
        ls = "--" if name == "FNO" else ":"
        lw = 1.8 if name == "FNO" else 1.6
        plt.plot(dates, y, ls, label=name, linewidth=lw)

    subset_txt = ", ".join(sorted(models_to_plot)) if models_to_plot else "all"
    plt.title(f"Year={year} — Day-7 absolute WL (subset: {subset_txt})")
    plt.xlabel("Date");
    plt.ylabel("Water Level (m)")
    plt.grid(True, alpha=0.3);
    plt.legend();
    plt.tight_layout()

    os.makedirs(ART_DIR, exist_ok=True)
    out_png = os.path.join(ART_DIR, f"fig_year_{year}_subset.png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved full-year figure to: {out_png}")

def main():
    ap = argparse.ArgumentParser("Offline evaluation: metrics table + optional figure")
    ap.add_argument("--plot_window", type=str, default="tst_wet",
                    choices=["val_dry","val_wet","tst_dry","tst_wet","none"],
                    help="Which 60-day window to plot (or 'none')")
    # full-year plot
    ap.add_argument("--plot_year", type=int, default=None,
                    help="Plot a full-year Day-7 curve (e.g., 2024) and save to artifacts/fig_year_<year>.png")

    ap.add_argument(
        "--plot_models",
        type=str,
        default="truth,FNO,Pers",
        help="Comma-separated subset to plot, e.g. 'truth,FNO,Pers' or 'truth,FNO,Pers,Clim,Linear'"
    )

    args = ap.parse_args()

    models_to_plot = {s.strip() for s in args.plot_models.split(",")}

    # 1) Load data and assemble samples
    runner = TenYearUnifiedRunner(CSV_DIR, seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH)
    data = runner.load_range_data(2015, 2025, allow_missing_u=True)
    X, Y, tgt_season, tgt_dates = runner.prepare_sequences_no_season(data)

    # 2) Load climatology/norm_stats and model weights
    clim = np.load(CLIM_PATH)
    with open(NORM_PATH, "r", encoding="utf-8") as f:
        st = json.load(f)
    model = SeasonalFNO1D(modes=64, width=96, num_layers=4, input_features=X.shape[2], dropout_rate=0.1, l2=1e-5)
    _ = model(np.zeros((1, SEQ_LENGTH, X.shape[2]), dtype=np.float32), training=False)  # build graph
    model.load_weights(_find_ckpt())

    # 3) Split into windows
    idxs = _split_indices(runner, tgt_dates)

    # 4) Compute metrics (4 windows × 4 models × 4 metrics)
    rows = []
    for key in ["val_dry", "val_wet", "tst_dry", "tst_wet"]:
        idx = idxs[key]
        if len(idx) == 0: continue
        dates = pd.to_datetime(np.asarray(tgt_dates)[idx]).normalize()
        yt7   = Y[idx][:, -1, 0]                # Ground truth (day-7, absolute)
        Xn    = _norm_inputs_like_train(X[idx], st)

        yp_fno    = _fno_predict7(model, Xn, dates, clim, st)
        yp_pers   = predict_persistence_7th(X[idx])
        yp_clim   = predict_climatology_7th(dates, clim)
        yp_linear = predict_linear_7th(X[idx], lookback=14)

        for name, yp in [("FNO", yp_fno), ("Pers", yp_pers), ("Clim", yp_clim), ("Linear", yp_linear)]:
            rows.append({
                "window": key, "model": name, "n": int(len(yt7)),
                "rmse": rmse(yt7, yp), "mae": mae(yt7, yp),
                "nse": nse(yt7, yp), "r": pearson_r(yt7, yp)
            })

    df = pd.DataFrame(rows).sort_values(["window","model"]).reset_index(drop=True)

    # 5) Save the table
    os.makedirs(ART_DIR, exist_ok=True)
    out_csv = os.path.join(ART_DIR, "metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved metrics to: {out_csv}")

    # 6) Optional plotting:
    # for the chosen window, plot only selected models
    # (default: truth/FNO/Pers)
    if args.plot_window != "none":
        key = args.plot_window
        idx = idxs[key]
        if len(idx) > 0:
            dates = pd.to_datetime(np.asarray(tgt_dates)[idx]).normalize()
            yt7 = Y[idx][:, -1, 0]
            Xn = _norm_inputs_like_train(X[idx], st)

            # Prepare each series
            series = {
                "truth": yt7,
                "FNO": _fno_predict7(model, Xn, dates, clim, st),
                "Pers": predict_persistence_7th(X[idx]),
                "Clim": predict_climatology_7th(dates, clim),
                "Linear": predict_linear_7th(X[idx], lookback=14),
            }
            # Keep only those selected via the command line
            series = {k: v for k, v in series.items() if k in models_to_plot}

            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4.8))

            if "truth" in series:
                plt.plot(dates, series.pop("truth"), label="Truth", linewidth=2.2)

            for name, yhat in series.items():
                ls = "--" if name == "FNO" else ":"
                lw = 1.8 if name == "FNO" else 1.6
                plt.plot(dates, yhat, ls, label=name, linewidth=lw)

            plt.title(f"Window={key} — Day-7 absolute WL (subset: {', '.join(sorted(models_to_plot))})")
            plt.xlabel("Date");
            plt.ylabel("Water Level (m)")
            plt.grid(True, alpha=0.3);
            plt.legend();
            plt.tight_layout()
            fig_path = os.path.join(ART_DIR, f"fig_{key}_subset.png")
            plt.savefig(fig_path, dpi=150)
            print(f"Saved figure to: {fig_path}")

    # 7) Optional plotting: full-year single figure
    if args.plot_year is not None:
        _plot_year_all_models(
            X, Y, tgt_dates, model, st, clim,
            year=int(args.plot_year),
            models_to_plot=models_to_plot
        )

if __name__ == "__main__":
    main()
