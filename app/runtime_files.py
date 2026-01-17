# app/runtime_files.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from app.runtime_paths import RuntimeLayout


def safe_name(s: str) -> str:
    """
    Make a filename-safe token (cross-platform).
    Keep only [A-Za-z0-9_-], replace others with underscore.
    """
    s = str(s)
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in s)


# =========================
# Runtime cache (ephemeral but persistent on HF /data)
# =========================

def live_cache_daily_default(layout: RuntimeLayout) -> Path:
    """Default daily live cache for the target station (Stung Treng)."""
    return layout.cache / "live_recent_daily.json"


def live_cache_path(layout: RuntimeLayout, station_code: str) -> Path:
    """Generic per-station live cache (if you want to key by station code)."""
    return layout.cache / f"live_recent_{safe_name(station_code)}.json"


def live_cache_3s(layout: RuntimeLayout) -> Path:
    """Compatibility name used in your current app.py."""
    return layout.cache / "live_recent_3s.json"


def live_cache_pakse(layout: RuntimeLayout) -> Path:
    """Compatibility name used in your current app.py."""
    return layout.cache / "live_recent_pakse.json"


# =========================
# Runtime artifacts (your current app.py writes backfill here)
# =========================

def backfill_path(layout: RuntimeLayout) -> Path:
    """
    Canonical path for the historical backfill parquet.
    IMPORTANT: kept under layout.artifacts to match current app.py semantics.
    """
    return layout.artifacts / "live_backfill.parquet"


# =========================
# Optional: Assist params & backtest caches (future-proof)
# =========================

def assist_params_dir(layout: RuntimeLayout, station: str, model_id: str) -> Path:
    d = layout.cache / "assist_params" / safe_name(station) / safe_name(model_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def assist_params_path(
    layout: RuntimeLayout,
    station: str,
    upstream: str,
    model_id: str,
    last_date: date,
) -> Path:
    d = assist_params_dir(layout, station, model_id)
    return d / f"{safe_name(upstream)}_assist_{last_date.isoformat()}.json"


def backtest_cache_dir(layout: RuntimeLayout, station: str, model_id: str) -> Path:
    d = layout.cache / "backtests" / safe_name(station) / safe_name(model_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def backtest_cache_path(layout: RuntimeLayout, station: str, model_id: str, year: int, k: int) -> Path:
    d = backtest_cache_dir(layout, station, model_id)
    return d / f"backtest_{int(year)}_ytd_k{int(k)}.parquet"


def backtest_metrics_cache_path(layout: RuntimeLayout, station: str, model_id: str, year: int, k: int) -> Path:
    d = backtest_cache_dir(layout, station, model_id)
    return d / f"backtest_{int(year)}_ytd_k{int(k)}_metrics.json"


# =========================
# Optional: Offline artifacts synced from HF Dataset (if you later add them)
# =========================

def artifact_forecast_latest(layout: RuntimeLayout) -> Path:
    return layout.artifacts / "forecast_latest.parquet"


def artifact_backtest(layout: RuntimeLayout, year: int, k: int) -> Path:
    return layout.artifacts / f"backtest_{int(year)}_ytd_k{int(k)}.parquet"


def artifact_backtest_metrics(layout: RuntimeLayout, year: int, k: int) -> Path:
    return layout.artifacts / f"backtest_{int(year)}_ytd_k{int(k)}_metrics.json"


def artifact_status(layout: RuntimeLayout) -> Path:
    return layout.artifacts / "status.json"