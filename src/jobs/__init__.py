"""Local job-style entrypoints for refresh workflows."""

__all__ = [
    "refresh_backtest_job",
    "refresh_live_job",
]


def __getattr__(name: str):
    if name == "refresh_backtest_job":
        from src.jobs.refresh_backtest import refresh_backtest_job

        return refresh_backtest_job
    if name == "refresh_live_job":
        from src.jobs.refresh_live import refresh_live_job

        return refresh_live_job
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
