# AWS Deployment Notes

These notes document the validated AWS backend shape for the Mekong FNO v2 service. They are operational guidance only: this repository does not create AWS infrastructure in code, and the examples below do not include credentials or secrets.

## Deployment Strategy

- Hugging Face Space remains the long-running public demo.
- AWS is the production-oriented cold-standby backend for validation, interviews, and demos.
- When not demonstrating, keep the ECS service at Desired tasks = 0.
- When demonstrating, set Desired tasks = 1 and wait until the target group reports healthy.
- If the backend is not needed for several days, the ALB can be deleted to reduce cost.
- Keep S3, ECR, IAM roles, the ECS cluster, task definitions, security groups, target group, and the CloudWatch log group with 7-day retention.

## Region

- AWS Region: `ap-southeast-1`

## Configuration

Safe local defaults are in [`../env.example`](../env.example). Production values should be injected through ECS task environment variables or Parameter Store, not hardcoded in application code.

Key runtime variables:

- `APP_ENV=prod`
- `ARTIFACT_BACKEND=s3`
- `AWS_REGION=ap-southeast-1`
- `S3_BUCKET=<artifact-bucket>`
- `S3_PREFIX=mekong/v2/prod`
- `TARGET_STATION=014501`
- `ACTIVE_MODEL_ID=seasonal_fno_v1`
- `MODEL_MANIFEST_KEY=manifests/model_manifest.json`
- `LOG_LEVEL=INFO`
- `PORT=8000`

## S3

Bucket naming pattern:

```text
mekong-fno-v2-artifacts-<account-id>-ap-southeast-1-<suffix>
```

Production bucket names must be configured through environment variables or Parameter Store. Do not hardcode bucket names in code.

Core production prefix layout:

```text
mekong/v2/prod/manifests/model_manifest.json
mekong/v2/prod/runtime/{station}/artifacts/status.json
mekong/v2/prod/runtime/{station}/artifacts/latest_inputs.json
mekong/v2/prod/runtime/{station}/cache/live_cache.json
mekong/v2/prod/backtests/{station}/{model_id}/summary.json
```

The online API reads runtime artifacts from S3. Scheduled jobs are the writers for shared runtime and backtest artifacts.

## ECR

Repository:

```text
mekong-fno-api
```

Image tag examples:

```text
v0.1.0
v0.1.1
prod-YYYYMMDD-HHMM
```

Example build/tag/push commands:

```bash
docker build -t mekong-fno-api:local .
docker tag mekong-fno-api:local <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com/mekong-fno-api:v0.1.1
docker push <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com/mekong-fno-api:v0.1.1
```

## ECS/Fargate API Service

- Cluster: `mekong-fno-api-prod-cluster`
- Service: `mekong-fno-api-prod`
- Task definition family: `mekong-fno-api-prod`
- Container: `main`
- Container port: `8000`
- Task role: `mekong-fno-api-task-role`
- Task execution role: `ecsTaskExecutionRole`
- Desired tasks: `0` for cold-standby, `1` for validation/demo

The current implementation uses standard ECS/Fargate, not App Runner, as the primary AWS backend path.

## ALB

- Name: `mekong-fno-api-alb`
- Listener: `HTTP:80`
- Target group: `mekong-fno-api-tg`
- Target type: `IP`
- Target port: `8000`
- Health check path: `/health/live`

`/health/live` is intentionally lightweight and does not require S3 or model loading.

## IAM Model

Online API task role:

- Read-only S3 access for the configured Mekong v2 prefix.
- No `s3:PutObject`.
- No `s3:DeleteObject`.

Scheduled jobs task role:

- Writer permissions for runtime, backtest, status, and snapshot artifacts.

Task execution role:

- ECR image pull.
- CloudWatch Logs delivery.

Separation rule:

- Online API reads runtime artifacts.
- Scheduled jobs write shared runtime/backtest artifacts.
- The online API must remain read-only.

Required online API S3 read actions:

```text
s3:GetObject
s3:ListBucket
s3:GetBucketLocation
```

Recommended scope:

```text
arn:aws:s3:::<artifact-bucket>
arn:aws:s3:::<artifact-bucket>/mekong/v2/prod/*
```

## CloudWatch

- Log group: `/aws/ecs/mekong-fno-api-prod`
- Retention: 7 days

## Parameter Store

Recommended parameter paths:

```text
/mekong-fno-v2/prod/ACTIVE_MODEL_ID
/mekong-fno-v2/prod/ARTIFACT_BACKEND
/mekong-fno-v2/prod/S3_BUCKET
/mekong-fno-v2/prod/S3_PREFIX
/mekong-fno-v2/prod/TARGET_STATION
/mekong-fno-v2/prod/MODEL_MANIFEST_KEY
/mekong-fno-v2/prod/AWS_REGION
```

The current ECS task may still use direct environment variables. Parameter Store integration is optional/planned and should not be read at import time unless implemented safely.

## App Runner

App Runner was considered earlier as a simpler API hosting option, but it is not the current main deployment path. The current backend uses standard ECS/Fargate, ALB, S3, ECR, IAM task roles, and CloudWatch.
