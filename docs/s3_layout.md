# Mekong v2 S3 Layout

This document defines the S3 object key layout used by `S3KeyBuilder`. It is a key-design boundary only; the Gradio app, FastAPI skeleton, runtime files, and jobs still use local storage by default.

## Prefix

All keys may be rooted under an optional prefix such as `mekong/v2`. The prefix is normalized so keys do not start with `/` and do not contain duplicate slashes.

## Namespaces

- Global model manifest: `manifests/model_manifest.json`
- Model artifacts: `models/{model_id}/manifest.json`, `models/{model_id}/weights/{filename}`, `models/{model_id}/assets/{filename}`
- Versioned assets: `assets/{version}/{filename}`
- Runtime state: `runtime/{station}/status.json`, `runtime/{station}/latest_inputs.json`, `runtime/{station}/cache/{filename}`, `runtime/{station}/artifacts/{filename}`
- Backtests: `backtests/{station}/summary.json`, `backtests/{station}/{model_id}/summary.json`, `backtests/{station}/{model_id}/{filename}`
- Snapshots: `snapshots/{snapshot_id}/status.json`, `snapshots/{snapshot_id}/runtime/{station}/status.json`, `snapshots/{snapshot_id}/backtests/{station}/summary.json`, `snapshots/{snapshot_id}/backtests/{station}/{model_id}/summary.json`

## Examples

- `mekong/v2/runtime/014501/status.json`
- `mekong/v2/assets/default/norm_stats.json`
- `mekong/v2/models/seasonal_fno_v1/weights/model.keras`
- `mekong/v2/backtests/Stung_Treng/seasonal_fno_v1/summary.json`

Station, model, version, and snapshot identifiers are normalized as safe tokens. Real migration of app/jobs to these S3 keys is intentionally deferred to a later task.
