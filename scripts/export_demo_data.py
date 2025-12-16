import os
import pandas as pd
from src.runner import TenYearUnifiedRunner

def export_stungtreng_2022_2024(csv_dir="data",
                                out_path="data-mini/stungtreng_wl_2022_2024_demo.csv"):
    """
    Export a 3-year daily water level demo CSV for the Stung Treng station.

    Args:
        csv_dir (str): directory containing the original MRC CSVs.
        out_path (str): Output path for the demo CSV. The file will contain
            only two columns: "Date" and "h".
    """

    # Instantiate the runner
    runner = TenYearUnifiedRunner(
        csv_files_path=csv_dir,
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
        raise RuntimeError("No usable days found for 2022–2024 – check your csv_dir.")

    # Convert to DataFrame and build the Date column
    df = pd.DataFrame(
        data_3yr,
        columns=["time_idx", "x_pos", "u", "h", "ts"]
    )
    df["Date"] = pd.to_datetime(df["ts"]).dt.date

    # Aggregate to daily mean water level and keep only Date / h
    daily = (
        df.groupby("Date")["h"]
        .mean()
        .reset_index()
    )

    # export the mini-dataset
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    daily.to_csv(out_path, index=False)

    print(f"Saved 2022–2024 daily demo CSV to:\n  {out_path}")
    print(f"Rows: {len(daily)}, range: {daily['Date'].min()} → {daily['Date'].max()}")

if __name__ == "__main__":
    export_stungtreng_2022_2024()
