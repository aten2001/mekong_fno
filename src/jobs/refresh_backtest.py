"""Local backtest refresh job boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.core.backtest import backtest_ytd_1day


def _horizons(values: Iterable[int]) -> list[int]:
    out = []
    for value in values:
        k = int(value)
        if k not in out:
            out.append(k)
    return out


def refresh_backtest_job(
    *,
    runner=None,
    water_daily: pd.Series | None = None,
    year: int = 2025,
    horizons: Iterable[int] = (1,),
    end=None,
    output_dir: str | Path | None = None,
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
            "dry_run": True,
            "year": year,
            "horizons": horizon_values,
            "output_dir": str(output) if output is not None else None,
            "requires": ["runner", "water_daily"],
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

    return {
        "dry_run": False,
        "year": year,
        "horizons": horizon_values,
        "results": results,
    }


def main() -> None:
    result = refresh_backtest_job(dry_run=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
