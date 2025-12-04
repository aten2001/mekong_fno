import os
import pandas as pd
from src.runner import TenYearUnifiedRunner

def export_stungtreng_2024(csv_dir="data",
                                      out_path="data-mini/stungtreng_wl_2024_demo.csv"):
    # instantiate the runner
    runner = TenYearUnifiedRunner(csv_files_path=csv_dir,
                                  seq_length=120,
                                  pred_length=7)

    # load data from year 2024
    data_2024 = runner.load_range_data(start_year=2024,
                                       end_year=2024,
                                       allow_missing_u=True)

    if len(data_2024) == 0:
        raise RuntimeError("No usable days found for 2024 – check your csv_dir.")

    df = pd.DataFrame(data_2024,
                      columns=["time_idx", "x_pos", "u", "h", "ts"])
    df["Date"] = pd.to_datetime(df["ts"]).dt.date

    daily = (
        df.groupby("Date")["h"]
        .mean()
        .reset_index()
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    daily.to_csv(out_path, index=False)

    print(f"Saved 2024 daily demo CSV to:\n  {out_path}")
    print(f"Rows: {len(daily)}, range: {daily['Date'].min()} → {daily['Date'].max()}")

if __name__ == "__main__":
    export_stungtreng_2024()
