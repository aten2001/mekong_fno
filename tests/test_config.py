import sys
from pathlib import Path

from src.config import ConfigError, load_settings


def test_config_defaults_are_local_and_safe():
    settings = load_settings(env={})

    assert settings.app_env == "local"
    assert settings.artifact_backend == "local"
    assert settings.aws_region == "ap-southeast-1"
    assert settings.s3_bucket == ""
    assert settings.s3_prefix == "mekong/v2/dev"
    assert settings.target_station == "014501"
    assert settings.active_model_id == "seasonal_fno_v1"
    assert settings.model_manifest_path == "assets/model_manifest.json"
    assert settings.model_manifest_key == "manifests/model_manifest.json"
    assert settings.log_level == "INFO"
    assert settings.port == 8000


def test_config_local_mode_allows_empty_s3_bucket():
    settings = load_settings(env={"ARTIFACT_BACKEND": "local", "S3_BUCKET": ""})

    assert settings.artifact_backend == "local"
    assert settings.s3_bucket == ""


def test_config_s3_mode_requires_bucket():
    try:
        load_settings(env={"ARTIFACT_BACKEND": "s3"})
    except ConfigError as exc:
        assert "S3_BUCKET" in str(exc)
        return
    raise AssertionError("S3 mode should require S3_BUCKET")


def test_config_s3_mode_accepts_bucket_and_overrides():
    settings = load_settings(
        env={
            "APP_ENV": "prod",
            "ARTIFACT_BACKEND": "S3",
            "AWS_REGION": "ap-southeast-1",
            "S3_BUCKET": "example-artifact-bucket",
            "S3_PREFIX": "mekong/v2/prod",
            "TARGET_STATION": "014501",
            "ACTIVE_MODEL_ID": "seasonal_fno_v1",
            "MODEL_MANIFEST_PATH": "/app/assets/model_manifest.json",
            "MODEL_MANIFEST_KEY": "manifests/model_manifest.json",
            "LOG_LEVEL": "debug",
            "PORT": "8080",
        }
    )

    assert settings.app_env == "prod"
    assert settings.artifact_backend == "s3"
    assert settings.s3_bucket == "example-artifact-bucket"
    assert settings.s3_prefix == "mekong/v2/prod"
    assert settings.log_level == "DEBUG"
    assert settings.port == 8080


def test_config_rejects_invalid_port():
    try:
        load_settings(env={"PORT": "not-a-port"})
    except ConfigError as exc:
        assert "PORT" in str(exc)
        return
    raise AssertionError("invalid PORT should raise ConfigError")


def test_config_import_has_no_heavy_or_cloud_side_effects():
    before = set(sys.modules)

    import src.config as config

    imported = set(sys.modules) - before
    assert "boto3" not in imported
    assert "botocore" not in imported
    assert "tensorflow" not in imported
    assert "gradio" not in imported
    assert "app.app" not in imported
    assert callable(config.load_settings)


def test_env_example_is_safe_to_commit():
    text = Path("env.example").read_text(encoding="utf-8")

    assert "APP_ENV=local" in text
    assert "ARTIFACT_BACKEND=local" in text
    assert "AWS_REGION=ap-southeast-1" in text
    assert "S3_BUCKET=" in text
    assert "S3_BUCKET=mekong-fno-v2-artifacts" not in text
    assert "AWS_ACCESS_KEY_ID" not in text
    assert "AWS_SECRET_ACCESS_KEY" not in text
    assert "AWS_SESSION_TOKEN" not in text


def test_aws_deployment_notes_document_current_ecs_path():
    text = Path("docs/aws_deployment.md").read_text(encoding="utf-8")

    assert "ap-southeast-1" in text
    assert "mekong-fno-api-prod-cluster" in text
    assert "mekong-fno-api-prod" in text
    assert "mekong-fno-api-alb" in text
    assert "mekong-fno-api-tg" in text
    assert "/aws/ecs/mekong-fno-api-prod" in text
    assert "Desired tasks = 0" in text
    assert "Desired tasks = 1" in text
    assert "App Runner was considered earlier" in text
    assert "not the current main deployment path" in text
