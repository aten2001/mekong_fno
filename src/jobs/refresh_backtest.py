"""Local backtest refresh job boundary."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.core.backtest import backtest_ytd_1day


SUMMARY_FILENAME = "summary.json"


def _horizons(values: Iterable[int]) -> list[int]:
    out = []
    for value in values:
        k = int(value)
        if k not in out:
            out.append(k)
    return out


def _artifact_label(ref) -> str:
    return getattr(ref, "uri", str(ref))


def _summary_ref(storage, station: str, model_id: str | None):
    return storage.backtest_path(SUMMARY_FILENAME, station=station, model_id=model_id)


def _write_backtest_summary(
    *,
    storage,
    station: str,
    model_id: str | None,
    summary: dict[str, Any],
) -> list[str]:
    ref = _summary_ref(storage, station, model_id)
    storage.write_json(ref, summary)
    return [_artifact_label(ref)]


def _build_summary(
    *,
    station: str,
    model_id: str | None,
    year: int,
    horizons: list[int],
    active_model_id: str | None,
    backend_mode: str,
    results: list[dict[str, Any]] | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(summary or {})
    payload.setdefault("available", bool(results))
    payload.update(
        {
            "updated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "station": station,
            "model_id": model_id,
            "active_model_id": active_model_id,
            "year": int(year),
            "horizons": horizons,
            "backend_mode": backend_mode,
            "writer": "refresh_backtest_job",
        }
    )
    if results is not None:
        payload["results"] = results
    return payload


def refresh_backtest_job(
    *,
    runner=None,
    water_daily: pd.Series | None = None,
    year: int = 2025,
    horizons: Iterable[int] = (1,),
    end=None,
    output_dir: str | Path | None = None,
    storage=None,
    station: str = "014501",
    model_id: str | None = None,
    summary: dict[str, Any] | None = None,
    active_model_id: str | None = None,
    backend_mode: str = "local",
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run or describe a local YTD backtest refresh.

    The app still owns model/service loading and concurrency-safe cache writes.
    Until that boundary is extracted, callers must pass an initialized runner and
    water_daily series when they want this job to compute real backtests.
    """
    year = int(year)
    horizon_values = _horizons(horizons)
    output = Path(output_dir) if output_dir is not None else None

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "year": year,
            "horizons": horizon_values,
            "output_dir": str(output) if output is not None else None,
            "requires": ["runner", "water_daily"],
            "station": station,
            "model_id": model_id,
            "active_model_id": active_model_id,
            "backend_mode": backend_mode,
            "written": [],
            "warnings": [],
        }

    if summary is not None:
        payload = _build_summary(
            station=station,
            model_id=model_id,
            year=year,
            horizons=horizon_values,
            active_model_id=active_model_id,
            backend_mode=backend_mode,
            summary=summary,
        )
        written = _write_backtest_summary(
            storage=storage,
            station=station,
            model_id=model_id,
            summary=payload,
        ) if storage is not None else []
        return {
            "ok": True,
            "dry_run": False,
            "year": year,
            "horizons": horizon_values,
            "station": station,
            "model_id": model_id,
            "active_model_id": active_model_id,
            "backend_mode": backend_mode,
            "latest_data_date": payload.get("period_end"),
            "written": written,
            "warnings": [] if storage is not None else ["No storage provided; summary was not persisted."],
            "summary": payload,
        }

    if runner is None or water_daily is None:
        raise ValueError(
            "refresh_backtest_job requires runner and water_daily until app service loading is extracted"
        )

    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    results = []
    for k in horizon_values:
        df, rmse = backtest_ytd_1day(
            runner,
            water_daily,
            start=f"{year}-01-01",
            end=end,
            horizon=k,
        )
        result = {
            "year": year,
            "horizon": int(k),
            "rows": int(len(df)) if df is not None else 0,
            "rmse": float(rmse) if rmse is not None else None,
        }

        if output is not None and df is not None and len(df) > 0:
            parquet_path = output / f"backtest_{year}_ytd_k{int(k)}.parquet"
            metrics_path = output / f"backtest_{year}_ytd_k{int(k)}_metrics.json"
            df.to_parquet(parquet_path, index=False)
            metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            result["parquet_path"] = str(parquet_path)
            result["metrics_path"] = str(metrics_path)

        results.append(result)

    written: list[str] = []
    if storage is not None:
        payload = _build_summary(
            station=station,
            model_id=model_id,
            year=year,
            horizons=horizon_values,
            active_model_id=active_model_id,
            backend_mode=backend_mode,
            results=results,
        )
        written = _write_backtest_summary(
            storage=storage,
            station=station,
            model_id=model_id,
            summary=payload,
        )

    return {
        "ok": True,
        "dry_run": False,
        "year": year,
        "horizons": horizon_values,
        "results": results,
        "station": station,
        "model_id": model_id,
        "active_model_id": active_model_id,
        "backend_mode": backend_mode,
        "written": written,
        "warnings": [],
    }


def main() -> None:
    result = refresh_backtest_job(dry_run=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
