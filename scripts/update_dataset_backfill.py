import sys
from pathlib import Path

# --- ensure repo root is importable in GitHub Actions ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jobs.refresh_live import (
    load_backfill_parquet,
    main,
    merge_backfill_frames,
    refresh_live_job,
    stable_live_rows,
    today_utc7_date,
)


if __name__ == "__main__":
    main()
