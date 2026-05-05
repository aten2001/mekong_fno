"""Small local model manifest helpers.

Active model resolution priority:
1. non-empty ``ACTIVE_MODEL_ID`` environment override
2. ``active_model_id`` from the selected manifest file
3. ``None`` when neither is configured
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MANIFEST_RELATIVE_PATH = Path("assets") / "model_manifest.json"


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    station: str | None = None
    horizon: int | None = None
    description: str | None = None
    weights_path: str | None = None
    assets_version: str | None = None
    created_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, key: str, data: Mapping[str, Any]) -> "ModelRecord":
        model_id = str(data.get("model_id") or key)
        if model_id != key:
            raise ValueError(f"model record key {key!r} does not match model_id {model_id!r}")

        known = {
            "model_id",
            "station",
            "horizon",
            "description",
            "weights_path",
            "assets_version",
            "created_at",
        }
        return cls(
            model_id=model_id,
            station=data.get("station"),
            horizon=data.get("horizon"),
            description=data.get("description"),
            weights_path=data.get("weights_path"),
            assets_version=data.get("assets_version"),
            created_at=data.get("created_at"),
            extra={k: v for k, v in data.items() if k not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "model_id": self.model_id,
            "station": self.station,
            "horizon": self.horizon,
            "description": self.description,
            "weights_path": self.weights_path,
            "assets_version": self.assets_version,
            "created_at": self.created_at,
        }
        data.update(self.extra)
        return data


@dataclass(frozen=True)
class ModelManifest:
    active_model_id: str | None = None
    models: dict[str, ModelRecord] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "active_model_id": self.active_model_id,
            "models": {model_id: record.to_dict() for model_id, record in self.models.items()},
        }
        data.update(self.extra)
        return data


def default_model_manifest_path(env: Mapping[str, str] | None = None) -> Path:
    config = os.environ if env is None else env
    override = str(config.get("MODEL_MANIFEST_PATH", "")).strip()
    if override:
        return Path(override)

    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / DEFAULT_MANIFEST_RELATIVE_PATH


def load_model_manifest(path: Path) -> ModelManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("model manifest root must be a JSON object")

    models_raw = raw.get("models") or {}
    if not isinstance(models_raw, dict):
        raise ValueError("model manifest 'models' must be a JSON object")

    models: dict[str, ModelRecord] = {}
    for model_id, record_raw in models_raw.items():
        if not isinstance(record_raw, dict):
            raise ValueError(f"model record {model_id!r} must be a JSON object")
        key = str(model_id)
        models[key] = ModelRecord.from_mapping(key, record_raw)

    known = {"active_model_id", "models"}
    active = raw.get("active_model_id")
    return ModelManifest(
        active_model_id=str(active).strip() if active is not None and str(active).strip() else None,
        models=models,
        extra={k: v for k, v in raw.items() if k not in known},
    )


def save_model_manifest(path: Path, manifest: ModelManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve_active_model_id(
    manifest_path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve active model id from env first, then manifest, without hard failures."""
    config = os.environ if env is None else env

    env_model_id = str(config.get("ACTIVE_MODEL_ID", "")).strip()
    if env_model_id:
        return env_model_id

    path = Path(manifest_path) if manifest_path is not None else default_model_manifest_path(config)
    if not path.exists():
        return None

    try:
        return load_model_manifest(path).active_model_id
    except Exception:
        return None


def get_active_model_record(
    manifest: ModelManifest,
    active_model_id: str | None = None,
) -> ModelRecord | None:
    model_id = active_model_id if active_model_id is not None else manifest.active_model_id
    if not model_id:
        return None
    return manifest.models.get(model_id)


def set_active_model_id(manifest: ModelManifest, model_id: str) -> ModelManifest:
    """Return a copied manifest with active_model_id changed to an existing model."""
    if model_id not in manifest.models:
        raise ValueError(f"unknown model_id {model_id!r}; cannot set active model")
    return ModelManifest(
        active_model_id=model_id,
        models=dict(manifest.models),
        extra=dict(manifest.extra),
    )


__all__ = [
    "DEFAULT_MANIFEST_RELATIVE_PATH",
    "ModelManifest",
    "ModelRecord",
    "default_model_manifest_path",
    "get_active_model_record",
    "load_model_manifest",
    "resolve_active_model_id",
    "save_model_manifest",
    "set_active_model_id",
]
