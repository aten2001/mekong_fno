"""Small file-oriented storage boundary for forecasting artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol


PathPart = str | Path
RuntimeArea = Literal["root", "cache", "artifacts"]


class StorageBackend(Protocol):
    """Minimal storage operations used by the app layer."""

    project_root: Path
    assets_dir: Path
    weights_dir: Path
    data_dir: Path
    runtime_root: Path
    runtime_cache_dir: Path
    runtime_artifacts_dir: Path

    def resolve_model_path(self, *parts: PathPart) -> Path:
        """Resolve a model/checkpoint path."""
        ...

    def resolve_asset_path(self, *parts: PathPart) -> Path:
        """Resolve a static asset path."""
        ...

    def resolve_data_path(self, *parts: PathPart) -> Path:
        """Resolve a local data path."""
        ...

    def runtime_path(self, *parts: PathPart, area: RuntimeArea = "root") -> Path:
        """Resolve a runtime path under root/cache/artifacts."""
        ...

    def backtest_path(
        self,
        *parts: PathPart,
        station: str | None = None,
        model_id: str | None = None,
    ) -> Path:
        """Resolve a backtest artifact/cache path."""
        ...

    def snapshot_path(self, *parts: PathPart) -> Path:
        """Resolve a snapshot artifact path."""
        ...

    def read_text(self, path: PathPart, *, encoding: str = "utf-8") -> str:
        """Read UTF-8 text from a local path."""
        ...

    def write_text(self, path: PathPart, text: str, *, encoding: str = "utf-8") -> None:
        """Write UTF-8 text to a local path."""
        ...

    def read_json(self, path: PathPart, *, encoding: str = "utf-8") -> Any:
        """Read JSON from a local path."""
        ...

    def write_json(
        self,
        path: PathPart,
        obj: Any,
        *,
        encoding: str = "utf-8",
        indent: int = 2,
    ) -> None:
        """Write JSON to a local path."""
        ...
