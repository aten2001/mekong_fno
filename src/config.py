"""Lightweight environment configuration for local and AWS runtimes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


class ConfigError(ValueError):
    """Raised when environment configuration is invalid."""


@dataclass(frozen=True)
class Settings:
    app_env: str
    artifact_backend: str
    aws_region: str
    s3_bucket: str
    s3_prefix: str
    target_station: str
    active_model_id: str
    model_manifest_path: str
    model_manifest_key: str
    log_level: str
    port: int


def _read(env: Mapping[str, str], name: str, default: str = "") -> str:
    return str(env.get(name, default)).strip()


def _normalize_backend(value: str) -> str:
    backend = value.strip().lower() or "local"
    if backend not in {"local", "s3", "remote"}:
        raise ConfigError(f"ARTIFACT_BACKEND must be one of local, s3, remote; got {value!r}")
    return backend


def _parse_port(value: str) -> int:
    raw = value.strip() or "8000"
    try:
        port = int(raw)
    except ValueError as exc:
        raise ConfigError(f"PORT must be an integer; got {value!r}") from exc
    if port <= 0:
        raise ConfigError(f"PORT must be positive; got {port}")
    return port


def load_settings(
    *,
    env: Mapping[str, str] | None = None,
    validate: bool = True,
) -> Settings:
    """Load process settings without network calls or AWS side effects."""
    config = os.environ if env is None else env
    settings = Settings(
        app_env=_read(config, "APP_ENV", "local") or "local",
        artifact_backend=_normalize_backend(_read(config, "ARTIFACT_BACKEND", "local")),
        aws_region=_read(config, "AWS_REGION", "ap-southeast-1") or "ap-southeast-1",
        s3_bucket=_read(config, "S3_BUCKET"),
        s3_prefix=_read(config, "S3_PREFIX", "mekong/v2/dev") or "mekong/v2/dev",
        target_station=_read(config, "TARGET_STATION", "014501") or "014501",
        active_model_id=_read(config, "ACTIVE_MODEL_ID", "seasonal_fno_v1") or "seasonal_fno_v1",
        model_manifest_path=(
            _read(config, "MODEL_MANIFEST_PATH", "assets/model_manifest.json") or "assets/model_manifest.json"
        ),
        model_manifest_key=(
            _read(config, "MODEL_MANIFEST_KEY", "manifests/model_manifest.json")
            or "manifests/model_manifest.json"
        ),
        log_level=(_read(config, "LOG_LEVEL", "INFO") or "INFO").upper(),
        port=_parse_port(_read(config, "PORT", "8000")),
    )

    if validate and settings.artifact_backend == "s3" and not settings.s3_bucket:
        raise ConfigError("S3_BUCKET is required when ARTIFACT_BACKEND=s3")

    return settings
