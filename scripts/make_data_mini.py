# scripts/make_data_mini.py
import os
import sys
import argparse
from pathlib import Path

import pandas as pd

# --- make paths stable regardless of current working directory ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runner import TenYearUnifiedRunner

def _resolve_under_repo(p: str) -> Path:
    """
    If p is absolute -> return as-is.
    If p is relative -> resolve as REPO_ROOT / p.
    """
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)


def parse_args():
    p = argparse.ArgumentParser("Make a small daily water-level sample (last N days)")

    p.add_argument(
        "--csv_dir",
        type=str,
        default=os.environ.get("CSV_DIR", "data"),
        help="Folder that contains the raw CSVs (Water Level..., Discharge...)",
    )
    p.add_argument("--seq_length", type=int, default=120, help="Model input L (kept for consistency)")
    p.add_argument("--pred_length", type=int, default=7, help="Model H (kept for consistency)")
    p.add_argument("--start_year", type=int, default=2015, help="Inclusive start year")
    p.add_argument("--end_year", type=int, default=2025, help="Inclusive end year")
    p.add_argument("--days", type=int, default=150, help="How many recent days to export (120~150 OK)")

    p.add_argument(
        "--out",
        type=str,
        default="data-mini/water_level_sample.csv",
        help="Output CSV path (relative paths are resolved under repo root)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    csv_dir = _resolve_under_repo(args.csv_dir)
    out_path = _resolve_under_repo(args.out)

    # 1) Validate output path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 2) Load the full dataset
    runner = TenYearUnifiedRunner(str(csv_dir), seq_length=args.seq_length, pred_length=args.pred_length)
    data = runner.load_range_data(args.start_year, args.end_year, allow_missing_u=True)
    if len(data) == 0:
        raise RuntimeError(
            f"No usable days found from {args.start_year}..{args.end_year}. "
            f"Check --csv_dir={csv_dir}"
        )

    # 3) Extract daily-mean water level for last N days (Date, h)
    df = pd.DataFrame(data, columns=["time_idx", "x_pos", "u", "h", "ts"])
    df["Date"] = pd.to_datetime(df["ts"]).dt.date
    s = df.groupby("Date")["h"].mean().tail(int(args.days))
    out = s.rename_axis("Date").reset_index().rename(columns={"h": "h"})

    # 4) Save
    out.to_csv(str(out_path), index=False, encoding="utf-8")
    print(f"Saved: {out_path}  (rows={len(out)})")
    print(f"CSV_DIR: {csv_dir}")


if __name__ == "__main__":
    main()
