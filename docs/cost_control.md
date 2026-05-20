# AWS Cost Control

## Purpose

This project uses AWS as a production-oriented validation/demo backend, while the Hugging Face Space remains the long-running public demo.

The AWS backend exists to show that the system can run with production-style infrastructure: ECS/Fargate, ALB, ECR, S3 artifact storage, IAM role separation, CloudWatch logs, and ECS/Fargate scheduled jobs. It is not intended to run as a 24/7 public service by default.

## Cold-Standby Model

The AWS backend is not operated as a continuously running public service by default; it is kept in cold-standby mode and started on demand for validation or demonstrations.

Default low-cost state:

- ECS API service Desired tasks = 0
- EventBridge Scheduler = Disabled
- CloudWatch log retention = 7 days
- S3, ECR, IAM roles, ECS task definitions, ECS clusters, and documentation are retained
- ALB can be deleted when not needed

Demo/validation state:

- ECS API service Desired tasks = 1
- ALB exists or is recreated
- Target group becomes healthy
- API endpoints checked: `/health/live`, `/status`, and `/docs`

## Cost Boundaries

- ECS Desired tasks = 0 stops Fargate compute cost for the API service.
- ECS Desired tasks = 0 does not stop ALB cost.
- ALB may still generate cost even when no ECS tasks are running.
- EventBridge Scheduler is Disabled by default, so scheduled jobs do not run automatically.
- CloudWatch log retention is set to 7 days to limit log storage cost.
- ECR stores Docker images and should keep only necessary tags.
- S3 artifact storage is retained and generally low-cost for this project size.
- NAT Gateway is intentionally avoided in this low-cost design.
- Public subnet + Public IP is used for short-lived Fargate tasks instead of maintaining a NAT Gateway.

## Recommended Low-Cost State

- ECS API service Desired tasks = 0
- No running ECS tasks
- EventBridge Scheduler entries Disabled
- No NAT Gateway
- No unattached Elastic IP
- CloudWatch retention = 7 days
- ALB removed if not needed for demonstrations
- S3 / ECR / IAM / ECS task definitions retained

## When Demonstrating

- Set ECS API service Desired tasks = 1
- Recreate or retain ALB
- Wait for target group healthy
- Verify `/health/live`, `/status`, and `/docs`
- Optionally run `refresh_live` or `refresh_backtest` manually
- Return Desired tasks to 0 after demonstration

## What Not To Do Yet

- Do not run ECS API 24/7 by default
- Do not enable EventBridge Scheduler permanently unless automatic daily/weekly refresh is required
- Do not add NAT Gateway for this portfolio/demo mode
- Do not add complex auto-deployment yet
- Do not keep ALB running for long periods if no demo is needed
