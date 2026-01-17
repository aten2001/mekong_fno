# scripts/export_demo_data.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# ---- Make imports work no matter where you run the script from ----
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runner import TenYearUnifiedRunner


def export_stungtreng_2022_2024(
    csv_dir: str | os.PathLike | None = None,
    out_path: str | os.PathLike | None = None,
):
    """
    Export a 3-year daily water level demo CSV for the Stung Treng station.

    Args:
        csv_dir:
            Directory containing the original MRC CSVs.
            If None, use env CSV_DIR, else <repo>/data.
        out_path:
            Output path for the demo CSV.
            If None, use <repo>/data-mini/stungtreng_wl_2022_2024_demo.csv.
            The file will contain only two columns: "Date" and "h".
    """
    # ---- Resolve stable paths (no CWD dependency) ----
    if csv_dir is None:
        csv_dir = os.environ.get("CSV_DIR") or str(REPO_ROOT / "data")
    if out_path is None:
        out_path = str(REPO_ROOT / "data-mini" / "stungtreng_wl_2022_2024_demo.csv")

    csv_dir_p = Path(csv_dir).expanduser().resolve()
    out_path_p = Path(out_path).expanduser()
    # If user gave a relative out_path, make it relative to repo root (stable)
    if not out_path_p.is_absolute():
        out_path_p = (REPO_ROOT / out_path_p).resolve()

    # Instantiate the runner
    runner = TenYearUnifiedRunner(
        csv_files_path=str(csv_dir_p),
        seq_length=120,
        pred_length=7,
    )

    # Load data from 2022–2024 period
    data_3yr = runner.load_range_data(
        start_year=2022,
        end_year=2024,
        allow_missing_u=True,
    )

    if len(data_3yr) == 0:
        raise RuntimeError(f"No usable days found for 2022–2024 – check csv_dir: {csv_dir_p}")

    # Convert to DataFrame and build the Date column
    df = pd.DataFrame(
        data_3yr,
        columns=["time_idx", "x_pos", "u", "h", "ts"],
    )
    df["Date"] = pd.to_datetime(df["ts"]).dt.date

    # Aggregate to daily mean water level and keep only Date / h
    daily = (
        df.groupby("Date")["h"]
        .mean()
        .reset_index()
    )

    # Export the mini-dataset
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_path_p, index=False, encoding="utf-8")

    print("Saved 2022–2024 daily demo CSV to:")
    print(f"  {out_path_p}")
    print(f"Rows: {len(daily)}, range: {daily['Date'].min()} → {daily['Date'].max()}")


if __name__ == "__main__":
    export_stungtreng_2022_2024()
