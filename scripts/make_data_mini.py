# scripts/make_data_mini.py
import os
import argparse
import pandas as pd
from src.runner import TenYearUnifiedRunner

def parse_args():
    p = argparse.ArgumentParser("Make a small daily water-level sample (last N days)")
    p.add_argument("--csv_dir", type=str, default=os.environ.get("CSV_DIR", "data"),
                   help="Folder that contains the raw CSVs (Water Level..., Discharge...)")
    p.add_argument("--seq_length", type=int, default=120, help="Model input L (kept for consistency)")
    p.add_argument("--pred_length", type=int, default=7, help="Model H (kept for consistency)")
    p.add_argument("--start_year", type=int, default=2015, help="Inclusive start year")
    p.add_argument("--end_year", type=int, default=2025, help="Inclusive end year")
    p.add_argument("--days", type=int, default=150, help="How many recent days to export (120~150 OK)")
    p.add_argument("--out", type=str, default="data-mini/water_level_sample.csv",
                   help="Output CSV path")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Validate output paths first
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 2) Load the full dataset (leap days already removed; and allow missing discharge values)
    runner = TenYearUnifiedRunner(args.csv_dir, seq_length=args.seq_length, pred_length=args.pred_length)
    data = runner.load_range_data(args.start_year, args.end_year, allow_missing_u=True)
    if len(data) == 0:
        raise RuntimeError(f"No usable days found from {args.start_year}..{args.end_year}. "
                           f"Check --csv_dir={args.csv_dir}")

    # 3) Extract the daily-mean water level for the last N days (Date, h)
    df = pd.DataFrame(data, columns=['time_idx', 'x_pos', 'u', 'h', 'ts'])
    df['Date'] = pd.to_datetime(df['ts']).dt.date
    # Daily mean
    s = df.groupby('Date')['h'].mean().tail(int(args.days))
    out = s.rename_axis("Date").reset_index().rename(columns={'h': 'h'})

    # 4) Save mini-dataset
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Saved {args.out}  (rows={len(out)})")

if __name__ == "__main__":
    main()
