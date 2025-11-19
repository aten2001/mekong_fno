# tests/test_infer.py
import os, glob, json
import numpy as np
import pandas as pd
import tensorflow as tf
import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from src.runner import TenYearUnifiedRunner
from src.model_fno import SeasonalFNO1D

def _find_ckpt(path="weights"):
    """
    Resolve a model-weights location into a loadable path for `model.load_weights()`.

    Args:
      path: Directory containing checkpoints, a checkpoint prefix (without extension),
        or a single `.weights.h5` file. Defaults to `"weights"`.

    Returns:
      str: A path acceptable to `tf.keras.Model.load_weights(...)`.
    """
    if os.path.isdir(path):
        ckpt = tf.train.latest_checkpoint(path)
        if ckpt:
            return ckpt
        idx = glob.glob(os.path.join(path, "*.ckpt.index"))
        if idx:
            return idx[0][:-len(".index")]
        h5 = glob.glob(os.path.join(path, "*.weights.h5"))
        if h5:
            return h5[0]
        import pytest
        pytest.skip(f"No checkpoint found in {path}. Run scripts/train_export.py first.")
    else:
        if os.path.exists(path + ".index") or path.endswith(".weights.h5"):
            return path
        import pytest
        pytest.skip(f"Checkpoint {path} not found")

def test_import_and_shape():
    """
    Sanity-check model import and output tensor shape.
    Builds a `SeasonalFNO1D` with expected hyperparameters, does a single
    forward pass on a dummy batch, and asserts the output shape is `(1, 7, 1)`.

    Raises:
      AssertionError: If the returned tensor has an unexpected shape.
    """
    model = SeasonalFNO1D(modes=64, width=96, num_layers=4, input_features=6)
    y = model(np.zeros((1, 120, 6), dtype=np.float32), training=False)
    assert y.shape == (1, 7, 1)

def test_metrics_json_exist():
    """
    Verify that phase/alignment report exists and has the expected top-level keys.
    If `artifacts/phase_report.json` is missing, the test is skipped (helpful for
    fresh environments). Otherwise, it asserts the presence of `"val"` and
    `"test_applied"` sections.

    Raises:
      AssertionError: If the JSON structure is present but missing required keys.
    """
    p = "artifacts/phase_report.json"
    if not os.path.exists(p):
        pytest.skip("Missing artifacts/phase_report.json (run training export first).")
    rep = json.load(open(p, "r", encoding="utf-8"))
    assert "val" in rep and "test_applied" in rep

def test_predict_with_mini_data():
    """
    End-to-end smoke test: mini dataset → runner → load weights → 7-day forecast.

    Workflow:
      1) Check the presence of required artifacts: `clim_vec.npy`, `norm_stats.json`,
         and a small CSV (`data-mini/water_level_sample.csv`). Skip if missing.
      2) Build `water_daily` (pd.Series indexed by python `date` → float `h`).
      3) Initialize `TenYearUnifiedRunner`, set climatology and normalization stats.
      4) Build the model graph, resolve weights (ckpt prefix or `.weights.h5`), and load.
      5) Use the last day in the mini dataset as the window end; predict next 7 days.
      6) Assert that predictions are length 7, finite, and the returned dates are sorted.

    Raises:
      AssertionError: If predictions have wrong length, contain non-finite values,
        or dates are not in ascending order.
    """
    # File existence check
    need = ["artifacts/clim_vec.npy", "artifacts/norm_stats.json", "data-mini/water_level_sample.csv"]
    for p in need:
        if not os.path.exists(p):
            pytest.skip(f"Missing {p}. Run scripts/make_data_mini.py and scripts/train_export.py.")

    # Convert the mini dataset to a pandas.Series (date → h)
    df = pd.read_csv("data-mini/water_level_sample.csv")
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    water_daily = pd.Series(df["h"].values, index=df["Date"])

    # Runner + (climatology / norm_stats)
    runner = TenYearUnifiedRunner(csv_files_path=".", seq_length=120, pred_length=7)
    clim = np.load("artifacts/clim_vec.npy")
    runner.set_climatology(clim)
    runner.norm_stats = json.load(open("artifacts/norm_stats.json", "r", encoding="utf-8"))

    # Build the graph & load weights
    model = SeasonalFNO1D(modes=64, width=96, num_layers=4, input_features=6)
    _ = model(np.zeros((1, 120, 6), dtype=np.float32), training=False)
    ckpt = _find_ckpt("weights")
    if ckpt.endswith(".h5"):
        model.load_weights(ckpt)
    else:
        status = model.load_weights(ckpt)
        try:
            status.expect_partial()  # For inference, restore only use model parameters
        except Exception:
            pass
    runner.model = model

    # Use the last day of the mini dataset as the window end and predict the next 7 days
    date_anchor = pd.Timestamp(sorted(water_daily.index)[-1])
    dates, h7 = runner.predict_h_range(water_daily, date_anchor=date_anchor, return_dates=True)

    # Assertions: length = 7; values are finite; dates are in ascending order
    assert len(h7) == 7
    assert all(np.isfinite(h7))
    assert list(pd.to_datetime(dates)) == sorted(pd.to_datetime(dates))
