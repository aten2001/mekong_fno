# app/runtime_paths.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeLayout:
    root: Path
    cache: Path
    artifacts: Path


def _is_writable_dir(p: Path) -> bool:
    """
    True if p can be created (if missing) and is writable.
    We verify by writing & deleting a small test file.
    """
    try:
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def get_runtime_root() -> Path:
    """
    Decide a writable runtime root directory.

    Priority:
      1) RUNTIME_ROOT env override (no validation here; user assumes responsibility)
      2) HF persistent storage: /data/runtime  (POSIX only, and /data exists)
      3) local repo runtime: .runtime
      4) fallback: /tmp/.runtime
    """
    # 1) env override
    env = os.environ.get("RUNTIME_ROOT", "").strip()
    if env:
        return Path(env)

    # 2) HF persistent storage mount (avoid Windows mis-detection)
    if os.name == "posix" and Path("/data").exists():
        p = Path("/data/runtime")
        if _is_writable_dir(p):
            return p

    # 3) local repo runtime
    p = Path(".runtime")
    if _is_writable_dir(p):
        return p

    # 4) fallback
    return Path("/tmp/.runtime")


def get_layout() -> RuntimeLayout:
    """
    Create and return a stable runtime layout:
      root/
        cache/
        artifacts/
    """
    root = get_runtime_root()
    cache = root / "cache"
    artifacts = root / "artifacts"
    cache.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    return RuntimeLayout(root=root, cache=cache, artifacts=artifacts)
