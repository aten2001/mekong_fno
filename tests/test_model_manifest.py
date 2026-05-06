import json
from pathlib import Path

from src.model_manifest import (
    DEFAULT_MANIFEST_RELATIVE_PATH,
    ModelManifest,
    ModelRecord,
    default_model_manifest_path,
    get_active_model_record,
    load_model_manifest,
    resolve_active_model_id,
    save_model_manifest,
    set_active_model_id,
)


def _manifest_dict(active_model_id="seasonal_fno_v1"):
    return {
        "active_model_id": active_model_id,
        "models": {
            "seasonal_fno_v1": {
                "model_id": "seasonal_fno_v1",
                "station": "014501",
                "horizon": 7,
                "description": "Seasonal FNO baseline model",
                "weights_key": "models/seasonal_fno_v1/weights/model.weights.h5",
                "weights_path": "weights/stung_treng_fno.ckpt",
                "assets_version": "default",
                "created_at": None,
                "extra_note": "kept",
            },
            "seasonal_fno_v2": {
                "model_id": "seasonal_fno_v2",
                "station": "014501",
                "horizon": 7,
            },
        },
        "manifest_extra": True,
    }


def _write_manifest(path: Path, active_model_id="seasonal_fno_v1") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_manifest_dict(active_model_id), indent=2), encoding="utf-8")
    return path


def test_load_model_manifest_reads_active_model_id(tmp_path):
    path = _write_manifest(tmp_path / "model_manifest.json")

    manifest = load_model_manifest(path)

    assert manifest.active_model_id == "seasonal_fno_v1"
    assert "seasonal_fno_v1" in manifest.models
    assert manifest.models["seasonal_fno_v1"].weights_key == "models/seasonal_fno_v1/weights/model.weights.h5"
    assert manifest.models["seasonal_fno_v1"].extra["extra_note"] == "kept"
    assert manifest.extra["manifest_extra"] is True


def test_resolve_active_model_id_uses_env_override(tmp_path):
    path = _write_manifest(tmp_path / "model_manifest.json")

    assert resolve_active_model_id(path, env={"ACTIVE_MODEL_ID": "env_model"}) == "env_model"


def test_env_active_model_id_takes_priority_over_manifest(tmp_path):
    path = _write_manifest(tmp_path / "model_manifest.json", active_model_id="seasonal_fno_v1")

    resolved = resolve_active_model_id(
        path,
        env={"ACTIVE_MODEL_ID": "seasonal_fno_v2"},
    )

    assert resolved == "seasonal_fno_v2"


def test_resolve_active_model_id_falls_back_to_manifest(tmp_path):
    path = _write_manifest(tmp_path / "model_manifest.json", active_model_id="seasonal_fno_v1")

    assert resolve_active_model_id(path, env={}) == "seasonal_fno_v1"


def test_empty_active_model_env_is_ignored(tmp_path):
    path = _write_manifest(tmp_path / "model_manifest.json", active_model_id="seasonal_fno_v1")

    assert resolve_active_model_id(path, env={"ACTIVE_MODEL_ID": "   "}) == "seasonal_fno_v1"


def test_resolve_active_model_id_returns_none_for_missing_manifest(tmp_path):
    missing = tmp_path / "missing_model_manifest.json"

    assert resolve_active_model_id(missing, env={}) is None


def test_default_model_manifest_path_respects_env_override():
    path = default_model_manifest_path(env={"MODEL_MANIFEST_PATH": "custom/model_manifest.json"})

    assert path == Path("custom/model_manifest.json")


def test_default_model_manifest_path_falls_back_to_assets_manifest():
    path = default_model_manifest_path(env={})

    assert path == Path("src/model_manifest.py").resolve().parents[1] / DEFAULT_MANIFEST_RELATIVE_PATH


def test_set_active_model_id_switches_existing_model():
    manifest = ModelManifest(
        active_model_id="seasonal_fno_v1",
        models={
            "seasonal_fno_v1": ModelRecord(model_id="seasonal_fno_v1"),
            "seasonal_fno_v2": ModelRecord(model_id="seasonal_fno_v2"),
        },
    )

    switched = set_active_model_id(manifest, "seasonal_fno_v2")

    assert switched.active_model_id == "seasonal_fno_v2"
    assert manifest.active_model_id == "seasonal_fno_v1"


def test_set_active_model_id_rejects_unknown_model():
    manifest = ModelManifest(
        active_model_id=None,
        models={"seasonal_fno_v1": ModelRecord(model_id="seasonal_fno_v1")},
    )

    try:
        set_active_model_id(manifest, "unknown")
    except ValueError as exc:
        assert "unknown" in str(exc)
        return
    raise AssertionError("set_active_model_id should reject unknown model_id")


def test_get_active_model_record_returns_record_when_present():
    manifest = ModelManifest(
        active_model_id="seasonal_fno_v1",
        models={"seasonal_fno_v1": ModelRecord(model_id="seasonal_fno_v1", station="014501")},
    )

    record = get_active_model_record(manifest)

    assert record is not None
    assert record.model_id == "seasonal_fno_v1"
    assert record.station == "014501"


def test_get_active_model_record_returns_none_when_missing_or_unknown():
    manifest = ModelManifest(
        active_model_id=None,
        models={"seasonal_fno_v1": ModelRecord(model_id="seasonal_fno_v1")},
    )

    assert get_active_model_record(manifest) is None
    assert get_active_model_record(manifest, "unknown") is None


def test_save_model_manifest_writes_json_that_loads_again(tmp_path):
    path = tmp_path / "nested" / "model_manifest.json"
    manifest = ModelManifest(
        active_model_id="seasonal_fno_v1",
        models={
            "seasonal_fno_v1": ModelRecord(
                model_id="seasonal_fno_v1",
                station="014501",
                horizon=7,
                weights_key="models/seasonal_fno_v1/weights/model.weights.h5",
                extra={"checksum": "abc"},
            )
        },
    )

    save_model_manifest(path, manifest)
    loaded = load_model_manifest(path)

    assert loaded.active_model_id == "seasonal_fno_v1"
    assert loaded.models["seasonal_fno_v1"].weights_key == "models/seasonal_fno_v1/weights/model.weights.h5"
    assert loaded.models["seasonal_fno_v1"].extra["checksum"] == "abc"


def test_load_model_manifest_reports_invalid_json_as_value_error(tmp_path):
    path = tmp_path / "model_manifest.json"
    path.write_text("{not-json", encoding="utf-8")

    try:
        load_model_manifest(path)
    except ValueError as exc:
        assert "invalid model manifest JSON" in str(exc)
        return
    raise AssertionError("invalid manifest JSON should raise ValueError")


def test_model_manifest_module_has_no_heavy_or_cloud_imports():
    source = Path("src/model_manifest.py").read_text(encoding="utf-8").lower()
    blocked = (
        "tensorflow",
        "gr" + "adio",
        "app.app",
        "s" + "3",
        "bo" + "to3",
        "botocore",
    )
    for token in blocked:
        assert token not in source
