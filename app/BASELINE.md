# BASELINE.md

## Purpose

This document captures the **current baseline state** of the Mekong forecasting system before Mekong_Forecasting_System_v2 refactoring.

Its purpose is to:

- preserve the current entrypoints and execution flow
- record the current prediction path and runtime state handling
- document update/backfill workflows
- highlight the main app-data coupling points
- provide a comparison baseline for future refactors

This is a **working engineering baseline**, not a polished architecture document.

---

## 1. Current high-level state

The current system is still fundamentally a **HF Space / Gradio monolithic app**.

At the moment:

- the external serving entrypoint is `app/app.py`
- there is **no separate FastAPI / API server entrypoint yet**
- the app handles UI, model loading, data loading, live/backfill merge, runtime state usage, and prediction in the same application layer
- runtime state is still primarily organized around local runtime directories and synchronized artifacts
- backfill updates are currently handled through GitHub Actions + scripts + app-side artifact sync

---

## 2. Current entrypoints

### HF Space / Gradio entry
- HF Space config points to `app_file: app/app.py`
- `sdk: gradio`
- current public serving entry is therefore `app/app.py`

### Application process entry
The actual application process is launched from `app/app.py` through:

- `build_app()`
- `app.launch(...)`

### Startup sequence
Current startup flow is:

1. sync dataset artifacts
2. warm service via `_load_service()`
3. launch Gradio app

### Important current conclusion
There is **no independent backend service boundary yet**.
The current app entrypoint is still the same file that serves the UI.

---

## 3. Current local startup status

### Confirmed
What is confirmed from the current repo scan:

- the actual running app is `app/app.py`
- the app is launched through Gradio
- startup includes artifact sync + `_load_service()` warmup + `app.launch(...)`

### Not yet explicitly confirmed in this baseline
The following should be filled in after one local validation run:

- exact local startup command
- exact conda environment name used for local startup
- whether any required environment variables must be exported before local run
- whether local startup expects HF-like `/data/runtime` or falls back automatically

### TODO
Fill this section after a clean local launch test.

Suggested fields to fill later:

```text
Conda env:
Startup command:
Required env vars:
Expected startup time:
Expected first successful page: