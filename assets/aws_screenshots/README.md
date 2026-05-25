# AWS Screenshot Guide

This directory stores redacted AWS screenshots for portfolio and interview explanations.

Screenshots are optional documentation artifacts. The architecture diagram is preferred for public sharing, and raw AWS console screenshots should be redacted before commit.

Do not commit unredacted screenshots.

Redact:

- AWS account ID
- ARNs containing account ID
- personal email
- access keys
- billing details
- public DNS if not intended to be shared
- bucket names if account-specific

Suggested screenshot checklist:

- `01_ecs_api_service_desired_tasks_0.png`
- `02_ecs_api_service_desired_tasks_1_demo.png`
- `03_ecr_image_v0_1_1.png`
- `04_s3_runtime_artifacts.png`
- `05_cloudwatch_log_groups.png`
- `06_refresh_live_task_definition.png`
- `07_refresh_backtest_task_definition.png`
- `08_eventbridge_schedules_disabled.png`
- `09_architecture_diagram.png`
- `10_alb_target_group_healthy.png` (optional if ALB is retained during demo)
