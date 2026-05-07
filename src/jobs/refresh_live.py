"""Local live/backfill refresh job boundary."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd


BACKFILL_FILENAME = "live_backfill.parquet"
STATUS_FILENAME = "status.json"
LATEST_INPUTS_FILENAME = "latest_inputs.json"
LIVE_CACHE_FILENAME = "live_cache.json"


def _artifact_label(ref) -> str:
    return getattr(ref, "uri", str(ref))


def _runtime_ref(storage, station_code: str, filename: str, *, area: str):
    try:
        return storage.runtime_path(filename, station=station_code, area=area)
    except TypeError:
        return storage.runtime_path(station_code, filename, area=area)


def today_utc7_date():
    return pd.Timestamp.now(tz="UTC").tz_convert("Asia/Bangkok").date()


def load_backfill_parquet(path: Path | None) -> pd.DataFrame:
    """Load the current backfill parquet using the existing date/h schema."""
    if path is None or not path.exists():
        return pd.DataFrame(columns=["date", "h"])

    df = pd.read_parquet(path)
    if "date" not in df.columns or "h" not in df.columns:
        raise ValueError(f"unexpected schema: {list(df.columns)} (need ['date','h'])")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["h"] = pd.to_numeric(df["h"], errors="coerce")
    df = df.dropna(subset=["date", "h"]).sort_values("date").reset_index(drop=True)
    return df


def stable_live_rows(live_daily, cutoff) -> pd.DataFrame:
    """Return stable live rows through the cutoff date in backfill parquet schema."""
    stable_rows = []
    if live_daily is not None and len(live_daily) > 0:
        for d, value in live_daily.items():
            day = pd.to_datetime(d).date()
            if pd.notna(value) and day <= cutoff:
                stable_rows.append((pd.to_datetime(d).normalize(), float(value)))

    df = pd.DataFrame(stable_rows, columns=["date", "h"])
    if len(df) > 0:
        df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


def merge_backfill_frames(df_backfill: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Merge existing and newly stable backfill rows, preferring newer values."""
    df_all = pd.concat([df_backfill, df_new], ignore_index=True) if len(df_new) else df_backfill
    if len(df_all) > 0:
        df_all["date"] = pd.to_datetime(df_all["date"]).dt.normalize()
        df_all = df_all.sort_values("date").groupby("date", as_index=False).last()
        df_all = df_all.sort_values("date").reset_index(drop=True)
    return df_all


def _download_existing_backfill(
    *,
    repo_id: str,
    token: str,
    out_dir: Path,
) -> Path | None:
    from huggingface_hub import hf_hub_download

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=BACKFILL_FILENAME,
            token=token,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )
        path = Path(downloaded)
        print("[download] backfill ->", path)
        return path
    except Exception as exc:
        print("[download][warn] no existing backfill:", repr(exc))
        return None


def _fetch_live_daily(
    *,
    station_code: str,
    cache_path: Path,
    ttl_seconds: int,
):
    from src.backfill import series_from_any
    from src.live_mrc import get_recent_daily_cached

    live = get_recent_daily_cached(
        station_code=station_code,
        cache_path=str(cache_path),
        ttl_seconds=ttl_seconds,
    )
    return series_from_any(live)


def _build_status(*, station_code: str, rows: int, dmin, dmax, cutoff) -> dict[str, Any]:
    return {
        "updated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "station_code": station_code,
        "rows": int(rows),
        "range": [str(dmin) if dmin else None, str(dmax) if dmax else None],
        "cutoff_utc7_yesterday": str(cutoff),
        "files": [BACKFILL_FILENAME, STATUS_FILENAME],
    }


def _backfill_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in df.itertuples(index=False):
        records.append(
            {
                "date": str(pd.to_datetime(row.date).date()),
                "h": float(row.h),
            }
        )
    return records


def _latest_inputs_payload(*, station_code: str, df: pd.DataFrame, latest_data_date: str | None) -> dict[str, Any]:
    latest_value = None
    if len(df):
        latest_value = float(df.iloc[-1]["h"])
    return {
        "station_code": station_code,
        "latest_data_date": latest_data_date,
        "latest_value": latest_value,
        "rows": int(len(df)),
        "writer": "refresh_live_job",
    }


def _write_storage_runtime_artifacts(
    *,
    storage,
    station_code: str,
    df: pd.DataFrame,
    status: dict[str, Any],
) -> list[str]:
    latest_data_date = status["range"][1]
    latest_inputs = _latest_inputs_payload(
        station_code=station_code,
        df=df,
        latest_data_date=latest_data_date,
    )
    live_cache = {
        "station_code": station_code,
        "latest_data_date": latest_data_date,
        "rows": int(len(df)),
        "records": _backfill_records(df),
        "writer": "refresh_live_job",
    }

    refs = [
        (_runtime_ref(storage, station_code, STATUS_FILENAME, area="artifacts"), status),
        (_runtime_ref(storage, station_code, LATEST_INPUTS_FILENAME, area="artifacts"), latest_inputs),
        (_runtime_ref(storage, station_code, LIVE_CACHE_FILENAME, area="cache"), live_cache),
    ]
    written: list[str] = []
    for ref, payload in refs:
        storage.write_json(ref, payload)
        written.append(_artifact_label(ref))
    return written


def refresh_live_job(
    *,
    out_dir: str | Path = "out",
    repo_id: str | None = None,
    token: str | None = None,
    station_code: str = "014501",
    cache_path: str | Path | None = None,
    ttl_seconds: int = 0,
    download_existing: bool = True,
    live_daily=None,
    storage=None,
    active_model_id: str | None = None,
    backend_mode: str = "local",
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Refresh local live backfill outputs using the existing HF/local artifact names.

    This function is intentionally local-file oriented. It does not own app runtime
    root selection, locking, or artifact sync; scripts decide where ``out_dir`` points.
    """
    out = Path(out_dir)
    station_code = str(station_code).strip() or "014501"
    cache = Path(cache_path) if cache_path is not None else out / "tmp_live_recent_daily.json"

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "station_code": station_code,
            "out_dir": str(out),
            "cache_path": str(cache),
            "files": [BACKFILL_FILENAME, STATUS_FILENAME],
            "written": [],
            "warnings": [],
            "latest_data_date": None,
            "active_model_id": active_model_id,
            "backend_mode": backend_mode,
        }

    if storage is None or download_existing:
        out.mkdir(parents=True, exist_ok=True)

    if download_existing and not repo_id:
        raise RuntimeError("HF_DATASET_REPO is missing")
    if download_existing and not token:
        raise RuntimeError("HF_TOKEN is missing (dataset is private; token required)")

    existing_path = None
    if download_existing:
        existing_path = _download_existing_backfill(repo_id=repo_id or "", token=token or "", out_dir=out)

    df_backfill = load_backfill_parquet(existing_path)

    if live_daily is None:
        live_daily = _fetch_live_daily(
            station_code=station_code,
            cache_path=cache,
            ttl_seconds=ttl_seconds,
        )

    cutoff = today_utc7_date() - pd.Timedelta(days=1)
    df_new = stable_live_rows(live_daily, cutoff)
    df_all = merge_backfill_frames(df_backfill, df_new)

    dmin = df_all["date"].dt.date.min() if len(df_all) else None
    dmax = df_all["date"].dt.date.max() if len(df_all) else None
    status = _build_status(
        station_code=station_code,
        rows=len(df_all),
        dmin=dmin,
        dmax=dmax,
        cutoff=cutoff,
    )
    status["latest_data_date"] = status["range"][1]
    status["active_model_id"] = active_model_id
    status["backend_mode"] = backend_mode
    status["writer"] = "refresh_live_job"

    if storage is not None:
        written = _write_storage_runtime_artifacts(
            storage=storage,
            station_code=station_code,
            df=df_all,
            status=status,
        )
        return {
            "ok": True,
            "dry_run": False,
            "station_code": station_code,
            "rows": int(len(df_all)),
            "range": status["range"],
            "latest_data_date": status["latest_data_date"],
            "active_model_id": active_model_id,
            "backend_mode": backend_mode,
            "written": written,
            "warnings": [],
            "status": status,
        }

    out_backfill = out / BACKFILL_FILENAME
    df_all.to_parquet(out_backfill, index=False)

    status_path = out / STATUS_FILENAME
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[backfill] rows=", len(df_all), "range=", dmin, "->", dmax, "out=", out_backfill)
    return {
        "ok": True,
        "dry_run": False,
        "station_code": station_code,
        "rows": int(len(df_all)),
        "range": status["range"],
        "out_backfill": str(out_backfill),
        "status_path": str(status_path),
        "latest_data_date": status["latest_data_date"],
        "active_model_id": active_model_id,
        "backend_mode": backend_mode,
        "written": [str(out_backfill), str(status_path)],
        "warnings": [],
        "status": status,
    }


def main() -> None:
    out_dir = Path(os.environ.get("OUT_DIR", "out"))
    repo_id = os.environ.get("HF_DATASET_REPO", "").strip()
    token = os.environ.get("HF_TOKEN", "").strip()
    station_code = os.environ.get("STATION_CODE", "014501").strip() or "014501"

    print("[env] HF_DATASET_REPO =", repo_id)
    print("[env] STATION_CODE   =", station_code)

    refresh_live_job(
        out_dir=out_dir,
        repo_id=repo_id,
        token=token,
        station_code=station_code,
        ttl_seconds=0,
        download_existing=True,
    )


if __name__ == "__main__":
    main()
