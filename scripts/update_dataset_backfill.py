import os, json, time, sys
from pathlib import Path
import pandas as pd

# --- ensure repo root is importable in GitHub Actions ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download
from src.live_mrc import get_recent_daily_cached
from src.backfill import series_from_any

def today_utc7_date():
    return pd.Timestamp.now(tz="UTC").tz_convert("Asia/Bangkok").date()

def load_backfill_parquet(p: Path) -> pd.DataFrame:
    if (p is None) or (not p.exists()):
        return pd.DataFrame(columns=["date", "h"])
    df = pd.read_parquet(p)
    if "date" not in df.columns or "h" not in df.columns:
        raise ValueError(f"unexpected schema: {list(df.columns)} (need ['date','h'])")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["h"] = pd.to_numeric(df["h"], errors="coerce")
    df = df.dropna(subset=["date", "h"]).sort_values("date").reset_index(drop=True)
    return df

def main():
    out = Path(os.environ.get("OUT_DIR", "out"))
    out.mkdir(parents=True, exist_ok=True)

    repo_id = os.environ.get("HF_DATASET_REPO", "").strip()
    token   = os.environ.get("HF_TOKEN", "").strip()

    if not repo_id:
        raise RuntimeError("HF_DATASET_REPO is missing")
    if not token:
        raise RuntimeError("HF_TOKEN is missing (dataset is private; token required)")

    station_code = os.environ.get("STATION_CODE", "014501").strip() or "014501"

    print("[env] HF_DATASET_REPO =", repo_id)
    print("[env] STATION_CODE   =", station_code)

    tmp_cache = out / "tmp_live_recent_daily.json"

    try:
        bf_local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="live_backfill.parquet",
            token=token,
            local_dir=str(out),
            local_dir_use_symlinks=False,
        )
        bf_path = Path(bf_local)
        print("[download] backfill ->", bf_path)
    except Exception as e:
        bf_path = None
        print("[download][warn] no existing backfill:", repr(e))

    df_bf = load_backfill_parquet(bf_path)

    live = get_recent_daily_cached(
        station_code=station_code,
        cache_path=str(tmp_cache),
        ttl_seconds=0,
    )
    live = series_from_any(live)

    cutoff = today_utc7_date() - pd.Timedelta(days=1)
    stable_rows = []
    if live is not None and len(live) > 0:
        for d, v in live.items():
            dd = pd.to_datetime(d).date()
            if pd.notna(v) and dd <= cutoff:
                stable_rows.append((pd.to_datetime(d).normalize(), float(v)))

    df_new = pd.DataFrame(stable_rows, columns=["date", "h"])
    if len(df_new) > 0:
        df_new = df_new.dropna().sort_values("date").reset_index(drop=True)

    df_all = pd.concat([df_bf, df_new], ignore_index=True) if len(df_new) else df_bf
    if len(df_all) > 0:
        df_all["date"] = pd.to_datetime(df_all["date"]).dt.normalize()
        df_all = df_all.sort_values("date").groupby("date", as_index=False).last()
        df_all = df_all.sort_values("date").reset_index(drop=True)

    out_bf = out / "live_backfill.parquet"
    df_all.to_parquet(out_bf, index=False)

    dmin = df_all["date"].dt.date.min() if len(df_all) else None
    dmax = df_all["date"].dt.date.max() if len(df_all) else None

    status = {
        "updated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "station_code": station_code,
        "rows": int(len(df_all)),
        "range": [str(dmin) if dmin else None, str(dmax) if dmax else None],
        "cutoff_utc7_yesterday": str(cutoff),
        "files": ["live_backfill.parquet", "status.json"],
    }
    (out / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[backfill] rows=", len(df_all), "range=", dmin, "->", dmax, "out=", out_bf)

if __name__ == "__main__":
    main()