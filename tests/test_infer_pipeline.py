# tests/test_infer_pipeline.py
import os, json, numpy as np, pandas as pd
from src.runner import TenYearUnifiedRunner
from src.model_fno import SeasonalFNO1D
from src.dataio import today_in_station_tz, doy_no_leap, doy_no_leap_vec

def test_artifacts_exist_and_keys():
    """
    Verify presence of required artifacts and expected JSON structure.

    Raises:
      AssertionError: If any artifact is missing or required keys are absent.
    """
    assert os.path.exists("artifacts/clim_vec.npy")
    assert os.path.exists("artifacts/norm_stats.json")
    assert os.path.exists("artifacts/phase_report.json")
    assert os.path.exists("weights")
    rep = json.load(open("artifacts/phase_report.json","r",encoding="utf-8"))
    assert "val" in rep and "test_applied" in rep

def test_time_utils():
    """
    Validate time and DOY utilities (leap-day removed convention).

    Raises:
      AssertionError: If any invariant is violated.
    """
    t = today_in_station_tz()
    assert isinstance(t, pd.Timestamp)
    assert t.hour == 0 and t.minute == 0 and t.tz is None
    # DOY (remove leap day)
    d = pd.Timestamp("2024-02-28"); d2 = pd.Timestamp("2023-02-28")
    assert doy_no_leap(d) == doy_no_leap(d2)
    arr = doy_no_leap_vec([d, d2])
    assert arr.shape == (2,)
    assert arr[0] == arr[1]

def test_model_forward_shape():
    """
    Sanity check: SeasonalFNO1D forward pass returns (1, 7, 1).

    Raises:
      AssertionError: If the output shape is not `(1, 7, 1)`.
    """
    model = SeasonalFNO1D(modes=64, width=96, num_layers=4, input_features=6)
    y = model(np.zeros((1,120,6), dtype=np.float32), training=False)
    assert y.shape == (1,7,1)

def test_predict_api_with_mini_data():
    """
    Integration test: mini dataset → runner → weights → 7-day forecast.

    Workflow:
      1) Load `data-mini/water_level_sample.csv` and build `water_daily` (date→h).
      2) Initialize `TenYearUnifiedRunner`, set climatology and `norm_stats`.
      3) Build `SeasonalFNO1D`, resolve latest checkpoint (or fallback prefix), and load weights.
      4) Use the last available day as `date_anchor`; call `predict_h_range(..., return_dates=True)`.
      5) Assert 7 outputs and strictly ascending target dates.

    Raises:
      AssertionError: If forecast length != 7 or dates are not sorted.
    """
    df = pd.read_csv("data-mini/water_level_sample.csv")
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    water_daily = pd.Series(df['h'].values, index=df['Date'])

    runner = TenYearUnifiedRunner(csv_files_path=".", seq_length=120, pred_length=7)
    clim = np.load("artifacts/clim_vec.npy"); runner.set_climatology(clim)
    runner.norm_stats = json.load(open("artifacts/norm_stats.json","r",encoding="utf-8"))
    model = SeasonalFNO1D(modes=64, width=96, num_layers=4, input_features=6)
    _ = model(np.zeros((1,120,6), dtype=np.float32), training=False)
    # allow using the latest checkpoint
    # auto-resolved from the directory or a fixed prefix
    import tensorflow as tf, glob
    ckpt = tf.train.latest_checkpoint("weights")
    if ckpt is None:
        # Fallback to using a prefix
        idx_files = glob.glob(os.path.join("weights","*.ckpt.index"))
        assert idx_files, "No TF checkpoint found."
        ckpt = idx_files[0].replace(".index","")
    model.load_weights(ckpt)
    runner.model = model

    date_anchor = pd.Timestamp(sorted(water_daily.index)[-1])
    dates, h7 = runner.predict_h_range(water_daily, date_anchor=date_anchor, return_dates=True)
    assert len(h7) == 7
    assert list(pd.to_datetime(dates)) == sorted(pd.to_datetime(dates))
