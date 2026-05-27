# Mekong v2 Deployment Bootstrap Manifest

`model_manifest.json` is the bootstrap model manifest for the Mekong v2 AWS deployment.

It should be uploaded to S3 under this key:

```text
mekong/v2/prod/manifests/model_manifest.json
```

The full expected S3 location is:

```text
s3://<artifact-bucket>/mekong/v2/prod/manifests/model_manifest.json
```

This manifest only selects the active model and records the expected artifact keys. It does not mean the real model weights have already been uploaded.

Do not store AWS credentials, API keys, tokens, or other secrets in this folder.
