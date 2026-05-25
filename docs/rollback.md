# Rollback Notes

## Purpose

These notes document how to return the AWS backend and runtime artifacts to a previous known-good state. They are operational guidance for the cold-standby AWS backend and do not create or modify infrastructure from this repository.

## API Image Rollback

- Identify the previous ECR image tag.
- Register or reuse an ECS task definition revision pointing to the previous image.
- Update the ECS service to use the previous task definition revision.
- Set Desired tasks = 1 only for validation.
- Verify `/health/live`, `/status`, and `/docs`.
- Return Desired tasks = 0 after validation if not demonstrating.

## ECS Service Rollback

- Use the previous ECS task definition revision.
- Confirm the task role and execution role are unchanged.
- Confirm environment variables are unchanged or restored.
- Confirm target group health after rollback.

## Runtime Artifact Rollback

- Restore the previous S3 runtime artifact if S3 versioning is enabled.
- Restore the previous `model_manifest.json` or status artifact.
- Keep the online API read-only.
- Use scheduled jobs or manual artifact restore as the write path.

## Model Rollback

- Change `ACTIVE_MODEL_ID`.
- Restore or update `model_manifest.json`.
- Validate `/status` reflects the intended model/backend state.

## Scheduled Jobs Rollback

- Keep EventBridge Scheduler Disabled by default.
- Revert the `refresh_live` or `refresh_backtest` task definition revision if needed.
- Manually Run Task for validation before enabling any schedule.

## Validation Checklist

- `/health/live` works
- `/status` works
- `/docs` works
- CloudWatch logs show no startup error
- S3 artifact paths are readable
- No unexpected running tasks remain after validation
