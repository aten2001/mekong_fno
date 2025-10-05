# selfcheck_predict_range.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.runner import TenYearUnifiedRunner

# ---- 1) Prepare water_daily (at least the most recent L days; the index must be Python date objects!) ----
L = 120
date_anchor = "2024-12-31"
end = pd.to_datetime(date_anchor)
days = pd.date_range(end=end, periods=L, freq="D")
# Convert the index to Python date to match the key type used inside the runner
idx = [d.date() for d in days]
# Make up some smooth water level data (for wiring up the pipeline only).
vals = 5.0 + 0.01 * np.arange(len(idx))
water_daily = pd.Series(vals.astype(float), index=idx)

# ---- 2) Construct the Runner and inject the required clim and norm_stats ----
runner = TenYearUnifiedRunner(csv_files_path="src", seq_length=L, pred_length=7)

# Climatology: length 366 with indices 1..365 valid;
# here provide a simple sinusoidal annual cycle
clim = np.zeros(366, dtype=np.float32)
doy = np.arange(1, 366)
clim[1:] = (5.0 + 0.5 * np.sin(2 * np.pi * (doy - 1) / 365.0)).astype(np.float32)
runner.clim = clim

# Normalization statistics obtained from training
# (use safe placeholders here to avoid division by zero).
runner.norm_stats = dict(
    t_mean=0.0, t_std=1.0,
    h_in_mean=float(water_daily.mean()),
    h_in_std=float(max(1e-3, water_daily.std())),
    dh_in_mean=0.0,
    dh_in_std=1.0,
    # Standardization parameters for the target (anomaly);
    # set to 0/1 so that “de-standardization” becomes the identity mapping
    h_mean=0.0,
    h_std=1.0
)

# ---- 3) Use a DummyModel as a stand-in: outputs all zeros
# (i.e., standardized anomalies are all zeros) ----
# Thus, the absolute WL from predict_h_range equals the climatology
# (under the target7 rule).
class DummyModel:
    def predict(self, X, verbose=0):
        # X.shape: [B, L, 6] → return [B, 7, 1]
        B = X.shape[0]
        return np.zeros((B, 7, 1), dtype=np.float32)

runner.model = DummyModel()

# ---- 4) Call the unified inference API and add assertions
# (length 7, dates ascending, values are floats) ----
dates, h7 = runner.predict_h_range(
    water_daily=water_daily,
    date_anchor=date_anchor,
    return_dates=True  # Request to return dates
)

# Assert: length should be 7
assert len(h7) == 7, f"expected 7 values, got {len(h7)}"
# Assert: dates are in ascending order
assert list(dates) == sorted(list(dates)), "dates are not in ascending order"
# Assert: all values are floats
assert all(isinstance(float(v), float) for v in h7), "values are not float-like"

print("✅ self-check passed:")
print("dates:", [str(pd.to_datetime(d).date()) for d in dates])
print("h_abs:", [round(float(v), 3) for v in h7])

# Since DummyModel outputs 0, with the default clim_mode='target7'
# the “7th-day DOY climatology” is added to all 7 steps
# Therefore, the 7 values in h7 should be identical
# (all equal to the target7 climatology)
if len(h7) == 7:
    all_equal = np.allclose(h7, h7[0])
    print("all steps equal to target7 climatology?", all_equal)