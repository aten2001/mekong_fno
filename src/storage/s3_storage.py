"""S3-backed implementation of the storage boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.storage.base import PathPart, RuntimeArea
from src.storage.s3_keys import S3KeyBuilder, join_key


@dataclass(frozen=True)
class S3ObjectRef:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def join_s3_key(prefix: str = "", *parts: PathPart) -> str:
    """Backward-compatible wrapper around the centralized S3 key joiner."""
    return join_key(prefix, *parts)


class S3StorageBackend:
    """Small S3 adapter mirroring the local storage namespace helpers."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        client: Any | None = None,
        region_name: str | None = None,
    ) -> None:
        bucket = str(bucket).strip()
        if not bucket:
            raise ValueError("bucket is required")
        self.bucket = bucket
        self.key_builder = S3KeyBuilder(prefix=prefix)
        self.prefix = self.key_builder.prefix
        self._provided_client = client
        self.region_name = region_name
        self._lazy_client: Any | None = None

    @property
    def client(self):
        if self._provided_client is not None:
            return self._provided_client
        if self._lazy_client is None:
            try:
                import boto3
            except ModuleNotFoundError as exc:
                raise RuntimeError("boto3 is required when no S3 client is provided") from exc
            self._lazy_client = boto3.client("s3", region_name=self.region_name)
        return self._lazy_client

    def _key(self, *parts: PathPart) -> str:
        return self.key_builder.join_key(*parts)

    def _ref_key(self, key: str) -> S3ObjectRef:
        return S3ObjectRef(bucket=self.bucket, key=key)

    def _ref(self, *parts: PathPart) -> S3ObjectRef:
        return S3ObjectRef(bucket=self.bucket, key=self._key(*parts))

    def _coerce_ref(self, ref_or_key: S3ObjectRef | PathPart) -> S3ObjectRef:
        if isinstance(ref_or_key, S3ObjectRef):
            if ref_or_key.bucket != self.bucket:
                raise ValueError(f"S3ObjectRef bucket {ref_or_key.bucket!r} does not match backend bucket")
            return ref_or_key

        raw = str(ref_or_key)
        prefix = f"s3://{self.bucket}/"
        if raw.startswith(prefix):
            return S3ObjectRef(bucket=self.bucket, key=join_s3_key(raw[len(prefix):]))
        if raw.startswith("s3://"):
            raise ValueError("S3 URI bucket does not match backend bucket")

        key = join_s3_key(raw)
        if self.prefix and not (key == self.prefix or key.startswith(f"{self.prefix}/")):
            key = join_s3_key(self.prefix, key)
        return S3ObjectRef(bucket=self.bucket, key=key)

    def resolve_model_path(self, *parts: PathPart, model_id: str | None = None) -> S3ObjectRef:
        if model_id is None:
            return self._ref("models", *parts)
        return self._ref_key(self.key_builder.model_key(model_id, *parts))

    def resolve_asset_path(self, *parts: PathPart, version: str | None = None) -> S3ObjectRef:
        if version is None:
            return self._ref("assets", *parts)
        return self._ref_key(self.key_builder.asset_key(version, *parts))

    def resolve_data_path(self, *parts: PathPart) -> S3ObjectRef:
        return self._ref("data", *parts)

    def runtime_path(
        self,
        *parts: PathPart,
        station: str | None = None,
        area: RuntimeArea = "root",
    ) -> S3ObjectRef:
        if area == "root":
            if station:
                key = self.key_builder.join_key("runtime", self.key_builder.safe_token(station), *parts)
                return self._ref_key(key)
            return self._ref("runtime", *parts)
        if area == "cache":
            if station:
                return self._ref_key(self.key_builder.runtime_cache_key(station, *parts))
            return self._ref("runtime", "cache", *parts)
        if area == "artifacts":
            if station:
                return self._ref_key(self.key_builder.runtime_artifact_key(station, *parts))
            return self._ref("runtime", "artifacts", *parts)
        raise ValueError(f"Unknown runtime area: {area!r}")

    def backtest_path(
        self,
        *parts: PathPart,
        station: str | None = None,
        model_id: str | None = None,
    ) -> S3ObjectRef:
        if station is not None:
            return self._ref_key(self.key_builder.backtest_artifact_key(station, *parts, model_id=model_id))
        base: list[PathPart] = ["backtests"]
        if model_id is not None:
            base.append(model_id)
        return self._ref(*base, *parts)

    def snapshot_path(self, *parts: PathPart, snapshot_id: str | None = None) -> S3ObjectRef:
        if snapshot_id is None:
            return self._ref("snapshots", *parts)
        return self._ref_key(self.key_builder.snapshot_key(snapshot_id, *parts))

    def read_text(self, ref_or_key: S3ObjectRef | PathPart, *, encoding: str = "utf-8") -> str:
        ref = self._coerce_ref(ref_or_key)
        obj = self.client.get_object(Bucket=ref.bucket, Key=ref.key)
        body = obj["Body"]
        data = body.read() if hasattr(body, "read") else body
        if isinstance(data, str):
            return data
        return bytes(data).decode(encoding)

    def write_text(
        self,
        ref_or_key: S3ObjectRef | PathPart,
        text: str,
        *,
        encoding: str = "utf-8",
    ) -> None:
        ref = self._coerce_ref(ref_or_key)
        self.client.put_object(Bucket=ref.bucket, Key=ref.key, Body=text.encode(encoding))

    def read_json(self, ref_or_key: S3ObjectRef | PathPart, *, encoding: str = "utf-8") -> Any:
        return json.loads(self.read_text(ref_or_key, encoding=encoding))

    def write_json(
        self,
        ref_or_key: S3ObjectRef | PathPart,
        obj: Any,
        *,
        encoding: str = "utf-8",
        indent: int = 2,
    ) -> None:
        self.write_text(
            ref_or_key,
            json.dumps(obj, ensure_ascii=False, indent=indent),
            encoding=encoding,
        )

    def exists(self, ref_or_key: S3ObjectRef | PathPart) -> bool:
        ref = self._coerce_ref(ref_or_key)
        try:
            self.client.head_object(Bucket=ref.bucket, Key=ref.key)
            return True
        except Exception as exc:
            response = getattr(exc, "response", None) or {}
            code = str(response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"} or exc.__class__.__name__ in {
                "NoSuchKey",
                "NotFound",
            }:
                return False
            raise
