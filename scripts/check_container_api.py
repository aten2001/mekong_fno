"""Check a running local Mekong FastAPI container.

This helper does not start Docker. Start the container separately, then run:

    python scripts/check_container_api.py --expected-model-id seasonal_fno_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


def _request_json(url: str, *, method: str = "GET", body: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = response.read().decode("utf-8")
        return response.status, json.loads(payload)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_api(base_url: str, expected_model_id: str | None = None) -> None:
    base = base_url.rstrip("/")

    status_code, health = _request_json(f"{base}/health/live")
    _require(status_code == 200, "/health/live did not return HTTP 200")
    _require(health == {"status": "ok"}, "/health/live did not return {'status': 'ok'}")
    print("PASS /health/live")

    status_code, status = _request_json(f"{base}/status")
    _require(status_code == 200, "/status did not return HTTP 200")
    for field in (
        "ready",
        "service_status",
        "generated_at",
        "latest_data_date",
        "active_model_id",
        "backend_mode",
        "artifacts_ok",
        "upstream_status",
        "warnings",
    ):
        _require(field in status, f"/status missing field: {field}")
    if expected_model_id is not None:
        _require(
            status["active_model_id"] == expected_model_id,
            f"/status active_model_id {status['active_model_id']!r} != {expected_model_id!r}",
        )
    print("PASS /status")

    forecast_body = {
        "station": "014501",
        "horizon": 3,
        "mode": "live",
        "include_backtest": False,
        "include_uncertainty": True,
    }
    status_code, forecast = _request_json(f"{base}/forecast", method="POST", body=forecast_body)
    _require(status_code == 200, "/forecast did not return HTTP 200")
    _require(forecast["station"] == "014501", "/forecast station mismatch")
    _require(forecast["horizon"] == 3, "/forecast horizon mismatch")
    _require(forecast["mode"] == "live", "/forecast mode mismatch")
    _require(len(forecast["predictions"]) == 3, "/forecast prediction count mismatch")
    if expected_model_id is not None:
        _require(
            forecast["model_id"] == expected_model_id,
            f"/forecast model_id {forecast['model_id']!r} != {expected_model_id!r}",
        )
    _require(forecast["warnings"], "/forecast should include placeholder/read-only warnings")
    print("PASS /forecast")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a running local Mekong FastAPI container.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--expected-model-id", default=None)
    args = parser.parse_args()

    try:
        check_api(args.base_url, expected_model_id=args.expected_model_id)
    except (AssertionError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1

    print("PASS container API validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
