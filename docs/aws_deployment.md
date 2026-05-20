# AWS Deployment Notes

These notes document the validated AWS backend shape for the Mekong FNO v2 service. They are operational guidance only: this repository does not create AWS infrastructure in code, and the examples below use placeholders for sensitive values.

## Deployment Strategy

- Hugging Face Space remains the long-running public demo.
- AWS is the production-oriented cold-standby backend for validation, interviews, and demos.
- The AWS backend is not operated as a continuously running public service by default; it is kept in cold-standby mode and started on demand for validation or demonstrations.
- When not demonstrating, keep the ECS service at Desired tasks = 0.
- When demonstrating, set Desired tasks = 1 and wait until the target group reports healthy.
- If the backend is not needed for several days, the ALB can be deleted to reduce cost.
- Keep S3, ECR, IAM roles, ECS clusters, task definitions, security groups, target groups, and CloudWatch log groups with 7-day retention.
- See [`cost_control.md`](cost_control.md) for the detailed low-cost operating model.

## Deployment Path Decision

App Runner was an earlier considered option for hosting the FastAPI backend. It has been replaced by a Standard ECS/Fargate API service with an Application Load Balancer as the current AWS deployment path.

ECS/Fargate is used because it provides:

- Clear separation between the API task role and task execution role.
- Read-only S3 access for the online API task role.
- ALB target group health checks for the running API task.
- Direct control over Fargate task definitions, container port mapping, CPU/memory, and environment variables.
- Cold-standby cost control by setting Desired tasks = 0 when the AWS backend is not being demonstrated.
- Existing ECS/Fargate scheduled job entrypoints that write shared runtime and backtest artifacts.

Current request/data flow:

```text
HF Space -> ALB -> ECS/Fargate FastAPI -> S3
ECS scheduled jobs -> S3
ECR -> ECS/Fargate
ECS/Fargate API and jobs -> CloudWatch Logs
```

## Region

- AWS Region: `ap-southeast-1`

## Current AWS Resources

| Resource | Current value |
| --- | --- |
| ECS Cluster | `mekong-fno-api-prod-cluster` |
| ECS Service | `mekong-fno-api-prod` |
| ECS Task Definition | `mekong-fno-api-prod` |
| Container | `main` |
| Container Port | `8000` |
| ECR Repository | `mekong-fno-api` |
| ECR Image Tag | `v0.1.1` |
| ALB | `mekong-fno-api-alb` |
| Target Group | `mekong-fno-api-tg` |
| API Task Role | `mekong-fno-api-task-role` |
| Task Execution Role | `ecsTaskExecutionRole` |
| CloudWatch Log Group | `/aws/ecs/mekong-fno-api-prod` |
| Jobs ECS Cluster | `mekong-fno-jobs-prod` |
| refresh_live Task Definition | `mekong-fno-refresh-live` |
| refresh_backtest Task Definition | `mekong-fno-refresh-backtest` |
| Jobs Task Role | `mekong-fno-jobs-task-role` |
| Jobs Security Group | `mekong-fno-jobs-sg` |
| EventBridge Live Schedule | `mekong-fno-refresh-live-daily` (Disabled by default) |
| EventBridge Backtest Schedule | `mekong-fno-refresh-backtest-weekly` (Disabled by default) |
| refresh_live Log Group | `/aws/ecs/mekong-fno-refresh-live` |
| refresh_backtest Log Group | `/aws/ecs/mekong-fno-refresh-backtest` |

## Cold-Standby Operating Model

The AWS backend is not operated as a continuously running public service by default; it is kept in cold-standby mode and started on demand for validation or demonstrations.

Default low-cost state:

- ECS API service Desired tasks = 0.
- EventBridge Scheduler entries are Disabled.
- CloudWatch log retention = 7 days.
- S3, ECR, IAM roles, ECS task definitions, ECS clusters, security groups, target groups, and CloudWatch log groups are retained as deployment assets.
- The ALB can be deleted when it is not needed for demonstrations.

Demo/validation state:

- ECS API service Desired tasks = 1.
- ALB is created or retained.
- Target group health is checked.
- `/health/live`, `/status`, and `/docs` are verified.

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

The current implementation uses Standard ECS/Fargate, not App Runner, as the primary AWS backend path.

## ECS/Fargate Scheduled Jobs

The refresh jobs are ECS/Fargate one-off jobs, not long-running services.

- Jobs cluster: `mekong-fno-jobs-prod`
- `refresh_live` task definition: `mekong-fno-refresh-live`
- `refresh_backtest` task definition: `mekong-fno-refresh-backtest`
- Jobs task role: `mekong-fno-jobs-task-role`
- Jobs security group: `mekong-fno-jobs-sg`
- EventBridge schedule for live refresh: `mekong-fno-refresh-live-daily` (Disabled by default)
- EventBridge schedule for backtest refresh: `mekong-fno-refresh-backtest-weekly` (Disabled by default)

The scheduled jobs are the single writer for shared runtime/backtest artifacts. Keeping the schedules Disabled by default prevents automatic recurring job runs in the cold-standby operating mode.

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

- API log group: `/aws/ecs/mekong-fno-api-prod`
- refresh_live log group: `/aws/ecs/mekong-fno-refresh-live`
- refresh_backtest log group: `/aws/ecs/mekong-fno-refresh-backtest`
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

App Runner was considered earlier as a simpler API hosting option. It has been replaced by Standard ECS/Fargate API Service with Application Load Balancer.

App Runner is therefore historical context only. It is not the active or main FastAPI deployment path. The current backend uses Standard ECS/Fargate, ALB, S3, ECR, IAM task roles, and CloudWatch.
