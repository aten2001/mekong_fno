from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from app.runtime_lock import lock_for_path, _tmp_path


def _atomic_copy_file(src: Path, dst: Path) -> None:
    """
    Atomically copy a file to dst by:
    1) copying src to a unique temp file in dst's directory
    2) replacing dst with os.replace()

    This assumes src already exists and dst.parent is writable.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    tmp = _tmp_path(dst, suffix=".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def sync_backfill_from_dataset(
    repo_id: str,
    dst_path: Path,
    token: str | None = None,
) -> None:
    """
    Download live_backfill.parquet from HF Dataset cache, then atomically
    replace dst_path under a file lock.
    """
    src = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="live_backfill.parquet",
            token=token,
        )
    )

    with lock_for_path(dst_path):
        _atomic_copy_file(src, dst_path)
        print(f"[sync] backfill updated: {dst_path} (mtime={dst_path.stat().st_mtime})")


def sync_status_from_dataset(
    repo_id: str,
    dst_dir: Path,
    token: str | None = None,
) -> None:
    """
    Download status.json from HF Dataset cache, then atomically replace
    dst_dir/status.json under a file lock.
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / "status.json"

    try:
        src = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename="status.json",
                token=token,
            )
        )

        with lock_for_path(dst_path):
            _atomic_copy_file(src, dst_path)

        print(f"[sync] status updated: {dst_path}")
    except Exception as e:
        print(f"[sync][warn] status sync failed: {repr(e)}")