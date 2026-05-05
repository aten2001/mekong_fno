"""Centralized S3 key design for Mekong v2 artifacts."""

from __future__ import annotations

from pathlib import Path

from src.storage.base import PathPart


def safe_token(value: PathPart) -> str:
    """Return a single S3-safe identity token.

    Intended for station/model/version/snapshot identifiers, not arbitrary
    filenames. Unsafe characters become underscores.
    """
    raw = value.as_posix() if isinstance(value, Path) else str(value)
    raw = raw.replace("\\", "/").strip().strip("/")
    if len(raw) >= 2 and raw[1] == ":":
        raw = raw[2:].strip("/")
    token = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in raw)
    token = token.strip("_")
    if not token:
        raise ValueError("S3 key token must not be empty")
    return token


def _path_tokens(part: PathPart) -> list[str]:
    raw = part.as_posix() if isinstance(part, Path) else str(part).replace("\\", "/")
    raw = raw.strip()
    if len(raw) >= 2 and raw[1] == ":":
        raw = raw[2:]
    raw = raw.strip("/")

    tokens: list[str] = []
    for token in raw.split("/"):
        token = token.strip()
        if not token or token == ".":
            continue
        if token == "..":
            raise ValueError("S3 key parts must not contain '..'")
        tokens.append(token)
    return tokens


def join_key(*parts: PathPart) -> str:
    """Join S3 key parts deterministically without leading or duplicate slashes."""
    tokens: list[str] = []
    for part in parts:
        tokens.extend(_path_tokens(part))
    return "/".join(tokens)


class S3KeyBuilder:
    """Build deterministic Mekong v2 S3 keys under an optional prefix."""

    def __init__(self, prefix: str = "") -> None:
        self.prefix = join_key(prefix)

    def join_key(self, *parts: PathPart) -> str:
        return join_key(self.prefix, *parts)

    def safe_token(self, value: PathPart) -> str:
        return safe_token(value)

    def model_manifest_key(self) -> str:
        return self.join_key("manifests", "model_manifest.json")

    def model_root_key(self, model_id: str) -> str:
        return self.join_key("models", safe_token(model_id))

    def model_key(self, model_id: str, *parts: PathPart) -> str:
        return self.join_key("models", safe_token(model_id), *parts)

    def model_manifest_key_for_model(self, model_id: str) -> str:
        return self.model_key(model_id, "manifest.json")

    def model_weights_key(self, model_id: str, filename: PathPart) -> str:
        return self.model_key(model_id, "weights", filename)

    def model_asset_key(self, model_id: str, filename: PathPart) -> str:
        return self.model_key(model_id, "assets", filename)

    def asset_key(self, version: str, *parts: PathPart) -> str:
        return self.join_key("assets", safe_token(version), *parts)

    def runtime_status_key(self, station: str) -> str:
        return self.join_key("runtime", safe_token(station), "status.json")

    def runtime_latest_inputs_key(self, station: str) -> str:
        return self.join_key("runtime", safe_token(station), "latest_inputs.json")

    def runtime_cache_key(self, station: str, *parts: PathPart) -> str:
        return self.join_key("runtime", safe_token(station), "cache", *parts)

    def runtime_artifact_key(self, station: str, *parts: PathPart) -> str:
        return self.join_key("runtime", safe_token(station), "artifacts", *parts)

    def backtest_summary_key(self, station: str, model_id: str | None = None) -> str:
        if model_id is None:
            return self.join_key("backtests", safe_token(station), "summary.json")
        return self.join_key("backtests", safe_token(station), safe_token(model_id), "summary.json")

    def backtest_artifact_key(
        self,
        station: str,
        *parts: PathPart,
        model_id: str | None = None,
    ) -> str:
        if model_id is None:
            return self.join_key("backtests", safe_token(station), *parts)
        return self.join_key("backtests", safe_token(station), safe_token(model_id), *parts)

    def snapshot_key(self, snapshot_id: str, *parts: PathPart) -> str:
        return self.join_key("snapshots", safe_token(snapshot_id), *parts)

    def snapshot_status_key(self, snapshot_id: str) -> str:
        return self.snapshot_key(snapshot_id, "status.json")

    def snapshot_runtime_status_key(self, snapshot_id: str, station: str) -> str:
        return self.snapshot_key(snapshot_id, "runtime", safe_token(station), "status.json")

    def snapshot_backtest_summary_key(
        self,
        snapshot_id: str,
        station: str,
        model_id: str | None = None,
    ) -> str:
        if model_id is None:
            return self.snapshot_key(snapshot_id, "backtests", safe_token(station), "summary.json")
        return self.snapshot_key(
            snapshot_id,
            "backtests",
            safe_token(station),
            safe_token(model_id),
            "summary.json",
        )


__all__ = [
    "S3KeyBuilder",
    "join_key",
    "safe_token",
]
