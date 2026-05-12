"""Pydantic contracts for the future FastAPI forecast service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    station: str
    horizon: int = Field(default=7, ge=1, le=7)
    mode: Literal["live"] = "live"
    include_backtest: bool = False
    include_uncertainty: bool = True


class ForecastPoint(BaseModel):
    date: str
    y_pred: float
    lower: float | None = None
    upper: float | None = None


class ForecastMetrics(BaseModel):
    rmse: float | None = None
    mae: float | None = None
    mape: float | None = None
    baseline_rmse: float | None = None


class BacktestSummary(BaseModel):
    available: bool
    rmse_model: float | None = None
    rmse_persistence: float | None = None
    samples: int | None = None
    period_start: str | None = None
    period_end: str | None = None


class ForecastResponse(BaseModel):
    station: str
    mode: Literal["live"]
    horizon: int
    generated_at: str
    latest_data_date: str | None = None
    model_id: str | None = None
    assist_enabled: bool = False
    uncertainty_available: bool = False
    predictions: list[ForecastPoint]
    metrics: ForecastMetrics | None = None
    backtest: BacktestSummary | None = None
    warnings: list[str] = Field(default_factory=list)


class UpstreamStationStatus(BaseModel):
    available: bool
    latest_data_date: str | None = None
    used_for_assist: bool = False
    message: str | None = None


class RuntimeStatus(BaseModel):
    cache_available: bool = False
    status_artifact_available: bool = False
    latest_inputs_available: bool = False
    backtest_available: bool = False


class StatusResponse(BaseModel):
    ready: bool
    service_status: Literal["ok", "degraded", "not_ready", "error"]
    generated_at: str
    latest_data_date: str | None = None
    data_freshness_days: int | None = None
    active_model_id: str | None = None
    backend_mode: Literal["local", "s3", "remote"] = "local"
    artifacts_ok: bool = False
    upstream_status: dict[str, UpstreamStationStatus] = Field(default_factory=dict)
    runtime_status: RuntimeStatus | None = None
    warnings: list[str] = Field(default_factory=list)
