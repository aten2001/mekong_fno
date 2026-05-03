"""Local filesystem implementation of the storage boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.storage.base import PathPart, RuntimeArea


def _join(base: Path, parts: tuple[PathPart, ...]) -> Path:
    path = base
    for part in parts:
        path = path / Path(part)
    return path


def _safe_name(value: str) -> str:
    """Mirror the current runtime filename token convention without importing app code."""
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(value))


class LocalStorageBackend:
    """A thin adapter around the existing local project/runtime directory layout."""

    def __init__(
        self,
        *,
        project_root: PathPart,
        runtime_root: PathPart,
        assets_dir: PathPart | None = None,
        weights_dir: PathPart | None = None,
        data_dir: PathPart | None = None,
        runtime_cache_dir: PathPart | None = None,
        runtime_artifacts_dir: PathPart | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.assets_dir = Path(assets_dir) if assets_dir is not None else self.project_root / "assets"
        self.weights_dir = Path(weights_dir) if weights_dir is not None else self.project_root / "weights"
        self.data_dir = Path(data_dir) if data_dir is not None else self.project_root / "data"
        self.runtime_root = Path(runtime_root)
        self.runtime_cache_dir = (
            Path(runtime_cache_dir) if runtime_cache_dir is not None else self.runtime_root / "cache"
        )
        self.runtime_artifacts_dir = (
            Path(runtime_artifacts_dir)
            if runtime_artifacts_dir is not None
            else self.runtime_root / "artifacts"
        )

    @classmethod
    def from_runtime_layout(
        cls,
        *,
        project_root: PathPart,
        runtime_layout: Any,
        assets_dir: PathPart | None = None,
        weights_dir: PathPart | None = None,
        data_dir: PathPart | None = None,
    ) -> "LocalStorageBackend":
        """Build from app.runtime_paths.RuntimeLayout without owning root resolution."""
        return cls(
            project_root=project_root,
            runtime_root=runtime_layout.root,
            runtime_cache_dir=runtime_layout.cache,
            runtime_artifacts_dir=runtime_layout.artifacts,
            assets_dir=assets_dir,
            weights_dir=weights_dir,
            data_dir=data_dir,
        )

    def resolve_model_path(self, *parts: PathPart) -> Path:
        return _join(self.weights_dir, parts)

    def resolve_asset_path(self, *parts: PathPart) -> Path:
        return _join(self.assets_dir, parts)

    def resolve_data_path(self, *parts: PathPart) -> Path:
        return _join(self.data_dir, parts)

    def runtime_path(self, *parts: PathPart, area: RuntimeArea = "root") -> Path:
        if area == "root":
            base = self.runtime_root
        elif area == "cache":
            base = self.runtime_cache_dir
        elif area == "artifacts":
            base = self.runtime_artifacts_dir
        else:
            raise ValueError(f"Unknown runtime area: {area!r}")
        return _join(base, parts)

    def backtest_path(
        self,
        *parts: PathPart,
        station: str | None = None,
        model_id: str | None = None,
    ) -> Path:
        base = self.runtime_cache_dir / "backtests"
        if station is not None:
            base = base / _safe_name(station)
        if model_id is not None:
            base = base / _safe_name(model_id)
        return _join(base, parts)

    def snapshot_path(self, *parts: PathPart) -> Path:
        return _join(self.runtime_artifacts_dir / "snapshots", parts)

    def read_text(self, path: PathPart, *, encoding: str = "utf-8") -> str:
        return Path(path).read_text(encoding=encoding)

    def write_text(self, path: PathPart, text: str, *, encoding: str = "utf-8") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=encoding)

    def read_json(self, path: PathPart, *, encoding: str = "utf-8") -> Any:
        return json.loads(self.read_text(path, encoding=encoding))

    def write_json(
        self,
        path: PathPart,
        obj: Any,
        *,
        encoding: str = "utf-8",
        indent: int = 2,
    ) -> None:
        text = json.dumps(obj, ensure_ascii=False, indent=indent)
        self.write_text(path, text, encoding=encoding)
