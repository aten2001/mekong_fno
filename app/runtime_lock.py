# app/runtime_lock.py
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from filelock import FileLock


def lock_for_path(p: Path) -> FileLock:
    """
    Create a file lock for a target path.
    The lock file lives next to the target file: <target>.lock
    """
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(p) + ".lock")


def _tmp_path(p: Path, suffix: str = ".tmp") -> Path:
    """
    Build a unique temp path in the same directory as p.
    """
    p = Path(p)
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    return p.with_name(p.name + f"{suffix}.{token}")


def atomic_write_text(p: Path, text: str, *, encoding: str = "utf-8") -> None:
    """
    Atomically write text to p by writing to a temp file then replace().
    """
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(p, suffix=".tmp")

    tmp.write_text(text, encoding=encoding)
    tmp.replace(p)


def atomic_write_json(p: Path, obj: Any, *, encoding: str = "utf-8", indent: int = 2) -> None:
    """
    Atomically dump JSON to p.
    """
    s = json.dumps(obj, ensure_ascii=False, indent=indent)
    atomic_write_text(p, s, encoding=encoding)


def atomic_write_parquet(p: Path, df, *, index: bool = False) -> None:
    """
    Atomically write parquet to p by writing temp then replace().
    Requires a parquet engine (pyarrow/fastparquet) installed.
    """
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(p, suffix=".tmp")

    df.to_parquet(tmp, index=index)
    tmp.replace(p)
