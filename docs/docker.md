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

## Supported Environment Variables

- `ARTIFACT_BACKEND=local|s3`
- `ACTIVE_MODEL_ID=seasonal_fno_v1`
- `MODEL_MANIFEST_PATH=/app/assets/model_manifest.json`
