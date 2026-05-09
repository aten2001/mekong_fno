# Local FastAPI Docker Runtime

This Docker setup runs the lightweight FastAPI service from `app.fastapi_app:app`.
It is for local verification only and does not deploy to AWS or change the
Gradio/HF Space entrypoint.

## Build

```bash
docker build -t mekong-fno-api:local .
```

## Run With Local Backend

Windows cmd:

```cmd
docker run --rm -p 8000:8000 -e ARTIFACT_BACKEND=local -e ACTIVE_MODEL_ID=seasonal_fno_v1 mekong-fno-api:local
```

PowerShell:

```powershell
docker run --rm -p 8000:8000 `
  -e ARTIFACT_BACKEND=local `
  -e ACTIVE_MODEL_ID=seasonal_fno_v1 `
  mekong-fno-api:local
```

Optional manifest override:

```powershell
docker run --rm -p 8000:8000 `
  -e ARTIFACT_BACKEND=local `
  -e ACTIVE_MODEL_ID=seasonal_fno_v1 `
  -e MODEL_MANIFEST_PATH=/app/assets/model_manifest.json `
  mekong-fno-api:local
```

## Future Manual S3 Mode

S3 mode is documented for future/manual verification only. This task does not
add cloud deployment or require AWS credentials for local API startup.

```powershell
docker run --rm -p 8000:8000 `
  -e ARTIFACT_BACKEND=s3 `
  -e AWS_REGION=your-region `
  -e S3_BUCKET=your-bucket `
  mekong-fno-api:local
```

Do not hardcode real AWS credentials or bucket names in the image.

## Check Endpoints

- `http://127.0.0.1:8000/health/live`
- `http://127.0.0.1:8000/status`
- `http://127.0.0.1:8000/docs`

`/forecast` remains a placeholder/mock response until real inference is
connected.

## Local Validation Checklist

Run these from the project root in VS Code or a local terminal where Docker is
installed.

1. Build the image:

```powershell
docker build -t mekong-fno-api:local .
```

2. Start the container:

```powershell
docker run --rm -p 8000:8000 `
  -e ARTIFACT_BACKEND=local `
  -e ACTIVE_MODEL_ID=seasonal_fno_v1 `
  mekong-fno-api:local
```

Expected startup includes Uvicorn listening on `0.0.0.0:8000` inside the
container.

3. In another terminal, check the API:

```powershell
curl.exe http://127.0.0.1:8000/health/live
curl.exe http://127.0.0.1:8000/status
curl.exe -X POST http://127.0.0.1:8000/forecast `
  -H "Content-Type: application/json" `
  -d "{\"station\":\"014501\",\"horizon\":3,\"mode\":\"live\",\"include_backtest\":false,\"include_uncertainty\":true}"
```

Expected:

- `/health/live` returns `{"status":"ok"}`.
- `/status` includes `active_model_id: "seasonal_fno_v1"` and `backend_mode: "local"`.
- `/forecast` returns `station: "014501"`, `horizon: 3`, `mode: "live"`, three prediction points, `model_id: "seasonal_fno_v1"`, and placeholder/read-only warnings.

4. Optional helper check:

```powershell
python scripts/check_container_api.py --expected-model-id seasonal_fno_v1
```

This helper calls `/health/live`, `/status`, and `/forecast`. It does not start
or stop Docker.

5. Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

You should see `/health/live`, `/health/ready`, `/status`, and `/forecast`.

6. Stop the container:

- If running in the foreground, press `Ctrl+C`.
- If running detached, use `docker ps` and `docker stop <container_id>`.

## Current Limitations

- `/forecast` is still placeholder/mock until real FNO inference is connected.
- The online API remains read-only for shared runtime/backtest/status state.
- Scheduled jobs are responsible for shared-state writes.
- S3 mode is not fully validated unless AWS environment/configuration is supplied.

## Supported Environment Variables

- `ARTIFACT_BACKEND=local|s3`
- `ACTIVE_MODEL_ID=seasonal_fno_v1`
- `MODEL_MANIFEST_PATH=/app/assets/model_manifest.json`
