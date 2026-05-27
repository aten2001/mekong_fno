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

01_ecs_api_service_desired_tasks_0.png
02_ecs_api_task_definition_roles.png
03_ecs_api_service_desired_tasks_1_demo.png
04_ecr_image_v0_1_1.png
05_s3_runtime_artifacts.png
06_cloudwatch_log_groups.png
07_refresh_live_task_definition.png
08_refresh_backtest_task_definition.png
09_eventbridge_schedules_disabled.png
10_iam_roles_api_and_jobs.png
11_architecture_diagram.png
12_alb_target_group_healthy.png optional if ALB is retained during demo
