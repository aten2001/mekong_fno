# scripts/quick_infer.py
import os, glob, json
import numpy as np
import pandas as pd
import tensorflow as tf

from src.runner import TenYearUnifiedRunner
from src.model_fno import SeasonalFNO1D

SEQ_LENGTH   = 120
PRED_LENGTH  = 7

WEIGHTS_LOC  = "weights"
CLIM_PATH    = "artifacts/clim_vec.npy"
NORM_PATH    = "artifacts/norm_stats.json"
MINI_CSV     = "data-mini/water_level_sample.csv"

def find_weights(path_or_dir: str) -> str:
    """
    Return a path that can be passed directly to model.load_weights(...):
    - If it’s a directory: prefer latest_checkpoint;
    otherwise, find a *.ckpt.index file and strip .index to obtain the prefix
    - If it’s a single H5 file: return it as is.
    - If it’s a prefix string: check whether <prefix>.index exists.
    """
    # Single H5 file
    if path_or_dir.lower().endswith(".h5"):
        if os.path.exists(path_or_dir):
            return path_or_dir
        raise FileNotFoundError(f"H5 weights not found: {path_or_dir}")

    # Directory: use latest_checkpoint or the first *.ckpt.index
    if os.path.isdir(path_or_dir):
        ckpt = tf.train.latest_checkpoint(path_or_dir)
        if ckpt:
            return ckpt
        idx_files = glob.glob(os.path.join(path_or_dir, "*.ckpt.index"))
        if idx_files:
            return idx_files[0][:-len(".index")]
        raise FileNotFoundError(f"No TF checkpoint found in directory: {path_or_dir}")

    # Prefix: check for .index
    if os.path.exists(path_or_dir + ".index"):
        return path_or_dir

    raise FileNotFoundError(
        f"Cannot resolve weights from '{path_or_dir}'. "
        f"Pass a directory with checkpoints, a checkpoint prefix, or an .h5 file."
    )

def main():
    # 0) Existence checks
    for p in [CLIM_PATH, NORM_PATH, MINI_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    # 1) Read mini dataset → pandas.Series (date → h)
    df = pd.read_csv(MINI_CSV)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    water_daily = pd.Series(df["h"].values, index=df["Date"])
    if len(water_daily) < SEQ_LENGTH:
        raise ValueError(f"Mini data too short: need >= {SEQ_LENGTH}, got {len(water_daily)}")

    # 2) Construct the runner (only for inference pipeline)
    runner = TenYearUnifiedRunner(csv_files_path=".", seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH)

    # 3) Load climatology / norm_stats / weights
    clim = np.load(CLIM_PATH)
    runner.set_climatology(clim)

    with open(NORM_PATH, "r", encoding="utf-8") as f:
        runner.norm_stats = json.load(f)  # dict: t_mean/t_std/h_in_mean/...

    # 4) Build the graph & load weights (run one forward pass before load_weights)
    model = SeasonalFNO1D(modes=64, width=96, num_layers=4, input_features=6, dropout_rate=0.1, l2=1e-5)
    _ = model(np.zeros((1, SEQ_LENGTH, 6), dtype=np.float32), training=False)

    weights_path = find_weights(WEIGHTS_LOC)
    model.load_weights(weights_path)
    runner.model = model

    # 5) Use the last day in the mini data as the window end date
    # and predict the next 7 days of absolute water level
    date_anchor = pd.Timestamp(sorted(water_daily.index)[-1])
    dates, h7 = runner.predict_h_range(water_daily, date_anchor=date_anchor, return_dates=True)

    # 6) Print results
    print("dates:", [str(pd.to_datetime(d).date()) for d in dates])
    print("h7:", [round(float(x), 3) for x in h7])

if __name__ == "__main__":
    main()
