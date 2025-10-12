# scripts/fetch_live_daily.py
import os
import pandas as pd
from src.live_mrc import fetch_recent_measurements, recent_to_daily_mean

if __name__ == "__main__":
    code = os.environ.get("STUNG_TRENG_CODE", "014501")
    df = fetch_recent_measurements(code)
    print("recent rows:", len(df))
    ser = recent_to_daily_mean(df)
    print("daily mean (UTC+07):")
    print(pd.DataFrame({"date": ser.index, "mean_w": ser.values}))
