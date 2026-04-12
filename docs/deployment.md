# Deployment Guide

This page explains how the Mekong FNO system is deployed and operated as a **station-level forecasting and evaluation application**.

The current deployment approach is designed to support:

- local reproducibility
- hosted web delivery through Hugging Face Spaces
- persistent runtime state
- refreshed backfill / cache artifacts
- explicit reload behavior inside the app

This document complements:

- [`../README.md`](../README.md)
- [`architecture.md`](architecture.md)
- [`runtime-design.md`](runtime-design.md)
- [`automation.md`](automation.md)

---

## 1. Deployment Philosophy

The project is not intended to exist only as:

- a notebook
- an offline script collection
- a static benchmark report

Instead, it is deployed as a **lightweight station-level forecasting service** with:

- a live forecasting interface
- a historical backtesting interface
- persistent runtime behavior
- explicit reload controls
- automated artifact update support

The deployment design therefore emphasizes:

- repeatable startup
- stable file paths
- separation between static assets and runtime outputs
- compatibility between local execution and hosted execution

---

## 2. Supported Deployment Modes

The current system supports two main deployment modes:

### 2.1 Local deployment

The app can be run on a local machine using:

- local Python dependencies
- local access to project assets
- a project-local runtime directory

This mode is useful for:

- development
- debugging
- result verification
- local reproduction of hosted behavior

---

### 2.2 Hugging Face Space deployment

The primary hosted deployment target is:

- **Hugging Face Spaces**

In this mode, the same application is exposed as a live web service with:

- interactive UI
- persistent runtime storage
- startup-time model/data loading
- access to refreshed runtime artifacts

This hosted deployment is part of the system’s portfolio value because it demonstrates that the project is not just a model artifact, but a working deployed application.

---

## 3. Deployment Targets and Roles

The current deployment ecosystem includes:

### 3.1 GitHub repository

The GitHub repository serves as the main source-control and documentation home for:

- source code
- README and docs
- workflows
- deployment-relevant configuration
- project assets committed into the repository

### 3.2 Hugging Face Space

The Hugging Face Space serves as the primary hosted delivery surface for:

- live forecasting
- evaluation / backtesting
- user-facing comparison and diagnostics
- reload behavior inside the deployed app

### 3.3 Runtime storage

The runtime storage layer holds operational outputs such as:

- caches
- refreshed backfill artifacts
- evaluation caches
- assist parameter caches
- other generated runtime state

---

## 4. Deployment-Relevant Repository Layout

The current deployment depends on the separation of several repository areas:

```text
app/        -> Gradio app, callbacks, runtime wiring
src/        -> modeling and data-processing logic
assets/     -> static inference/evaluation assets
weights/    -> model checkpoints
data/       -> historical source files
.github/    -> automation workflows
docs/       -> extended documentation
````

This separation is important because deployment stability depends on distinguishing:

* code
* static project resources
* generated operational state

---

## 5. Local Deployment

## 5.1 Prerequisites

A local deployment expects:

* a Python environment with project dependencies installed
* access to required files under:

  * `assets/`
  * `weights/`
  * `data/`
* optional path overrides if custom directories are needed

### Typical dependency installation

```bash
git clone https://github.com/aten2001/mekong_fno.git
cd mekong_fno
pip install -r requirements.txt
```

---

## 5.2 Launching locally

The main application entrypoint is:

```bash
python app/app.py
```

This launches the same Gradio application used in the hosted Space.

---

## 5.3 Local runtime behavior

When running locally:

* the app uses a local runtime root
* runtime-generated files are written to the local runtime directory
* the same reload logic and evaluation logic remain available

This is important because the local deployment is intended to be **behaviorally aligned** with the hosted version, not just a reduced dev-only stub.

---

## 5.4 Why local deployment matters

Local deployment is valuable for:

* confirming reproducibility outside the hosted environment
* debugging path issues
* validating evaluation behavior
* testing changes before pushing to hosted deployment

---

## 6. Hugging Face Space Deployment

## 6.1 Hosted deployment role

Hugging Face Space deployment is the primary public delivery mode for the project.

The hosted app serves as the system’s operational interface for:

* live 7-day station-level forecasting
* 1–7 day-ahead backtesting
* baseline / assist comparison
* availability-aware diagnostics
* routed operational interpretation

---

## 6.2 Why HF Space is suitable here

HF Space is a good fit for this project because it supports:

* Python app hosting
* Gradio-based interfaces
* repository-driven deployment
* persistent storage options
* easy public access for verification

This aligns well with the project’s goal of being a **deployable and verifiable forecasting system**.

---

## 6.3 Space metadata

The root `README.md` in the hosted deployment may include Hugging Face Space metadata at the top, such as:

* title
* emoji
* sdk
* app file
* other Space configuration fields

This metadata is part of Space configuration and must remain compatible with the actual application layout.

---

## 6.4 Hosted startup behavior

On startup, the deployed application typically:

1. resolves runtime paths
2. loads static assets and model checkpoints
3. loads historical and refreshed daily series
4. builds the cached service object
5. exposes the Gradio UI

This warm-up behavior is part of the deployment design, not just a side effect of coding style.

---

## 7. Runtime Root and Persistent Storage

## 7.1 Why runtime storage matters

The system distinguishes between:

* static repository assets
* runtime-generated operational files

This is crucial because the app generates and refreshes data over time.
If runtime outputs were mixed into static project directories without discipline, deployment behavior would become harder to reason about and maintain.

---

## 7.2 Local runtime root

In local deployment, runtime files are typically written to a project-local runtime directory.

This allows:

* iterative local testing
* cache reuse across local runs
* local artifact inspection

---

## 7.3 HF runtime root

In the hosted environment, runtime files are expected to use persistent storage when available.

This supports:

* survival of refreshed artifacts across sessions
* stability of cached state
* repeatable evaluation behavior over time

---

## 7.4 Runtime categories

Typical runtime outputs include:

* live-update caches
* backfill artifacts
* evaluation caches
* assist parameter caches
* runtime reports

These are operational files and should not be confused with repository-committed static resources.

---

## 8. Reload Behavior in Deployment

## 8.1 Why reload exists

The deployed app includes explicit reload behavior so that:

* refreshed runtime data can be picked up
* the cached service state can be rebuilt
* the user does not need to rely on a full application restart for every update

This is especially useful in hosted environments where runtime data may evolve while the app remains deployed.

---

## 8.2 What reload actually does

Conceptually, reload is responsible for:

* rebuilding service state
* re-reading relevant runtime inputs
* refreshing what the app exposes to its callbacks

It is an important operational feature because it turns the app into a **refreshable service**, not just a one-shot loaded interface.

---

## 8.3 Why reload is deployment-relevant

Reload is not just a UI convenience feature.
It is part of deployment architecture because it affects how the application behaves under evolving runtime state.

---

## 9. Backfill and Artifact Publication

## 9.1 Why artifact publication exists

The deployed system depends not only on code, but also on refreshed data products.

Examples include:

* recent daily values
* derived backfill artifacts
* evaluation-related runtime outputs

These need a repeatable path into the deployed environment.

---

## 9.2 Scheduled workflow role

The project includes a scheduled GitHub Actions workflow for publishing backfill-related artifacts.

Typical source location:

* `.github/workflows/publish_backfill.yml`

This helps automate the process of keeping deployed runtime inputs more current.

---

## 9.3 Relationship to deployment

This workflow is part of the deployment story because it helps connect:

* external refreshed data
* generated artifacts
* hosted application behavior

In other words, deployment is not only “host the app,” but also:

> **maintain the operational data path that feeds the app**

---

## 10. Common Deployment Flow

A typical deployment update flow looks like this:

1. code and docs are updated locally
2. changes are committed and merged to `main`
3. `main` is pushed to GitHub
4. `main` is pushed to HF Space
5. the Space rebuilds / refreshes
6. the hosted app loads current code + runtime state
7. reload behavior can be used to pick up newer runtime inputs when appropriate

This flow is especially useful when working with:

* app changes
* README/doc changes
* figure assets
* refreshed operational behavior

---

## 11. Practical Deployment Checklist

Before pushing a deployment update, the following should be checked.

### 11.1 Code and app entrypoint

* `app/app.py` exists
* the app launches locally
* imports resolve correctly

### 11.2 Required static resources

* needed files under `assets/` are present
* model checkpoints under `weights/` are present
* historical source files under `data/` are present

### 11.3 Runtime path behavior

* local runtime paths are valid
* hosted runtime paths are consistent with Space expectations
* persistent storage behavior is understood

### 11.4 Documentation-linked assets

If the README or docs reference files such as:

* figures
* diagrams
* report pages

those files must be:

* tracked by Git
* committed
* merged into `main`
* pushed to the target remote

This is especially important for README images.

### 11.5 Space metadata compatibility

If the root `README.md` includes HF Space front matter, verify that:

* metadata remains present
* fields remain valid
* the application path still matches deployment expectations

---

## 12. Common Deployment Failure Modes

## 12.1 README image exists locally but not on GitHub / HF

Cause:

* image file exists on disk but was never committed
* or was not merged into `main`

Effect:

* README renders broken images remotely

Fix:

* add the file to Git
* commit it
* merge into `main`
* push to GitHub and HF

---

## 12.2 Local files visible after branch switch but not actually tracked

Cause:

* files remain as untracked working-tree files across branch changes

Effect:

* local filesystem appears to contain them
* remote repositories do not

Fix:

* verify with `git status` and `git ls-files`
* add and commit the files explicitly

---

## 12.3 HF push rejected as non-fast-forward

Cause:

* `space/main` has commits not present in local `main`

Effect:

* push to HF is rejected

Fix:

* fetch `space`
* inspect divergence
* merge or rebase as appropriate
* then push again

---

## 12.4 README overwritten without preserving HF metadata

Cause:

* replacing the root `README.md` without keeping HF front matter

Effect:

* Space metadata may be lost or broken

Fix:

* preserve the YAML front matter
* keep one unified `README.md` for both GitHub and HF

---

## 13. Why the Deployment Design Matters for This Project

The deployment strategy is important because it directly supports the project’s strongest portfolio claims:

* it is not notebook-only
* it is not offline-only
* it is not a single static plot
* it is a deployed, refreshable, inspectable forecasting application

The deployment design therefore contributes directly to the project’s credibility as an applied AI / ML system.

---

## 14. Current Deployment Boundaries

The current deployment does not yet attempt to become:

* a large multi-service forecasting platform
* a basin-scale planning system
* a multi-tenant operational forecasting product

The current design is intentionally focused on:

> a **clean, deployable, verifiable station-level forecasting service**

This boundary is deliberate and keeps the system architecture aligned with the current project goals.

---

## 15. Future Deployment Extensions

Potential future deployment-oriented extensions include:

* richer human-readable results pages under `docs/`
* clearer deployment verification pages
* additional artifact-reporting surfaces
* optional external input integration
* stronger monitoring of runtime drift and availability behavior

These are future improvements, not requirements for the current deployed system to be considered complete.

---

## 16. Deployment Summary

The current Mekong FNO deployment can be summarized as:

* **source-controlled on GitHub**
* **hosted on Hugging Face Spaces**
* **backed by persistent runtime behavior**
* **supported by scheduled artifact publication**
* **operated through a reloadable Gradio application**

Its deployment design is one of the reasons the project reads as a **real operational system** rather than only a model-development exercise.

---

## 17. Related Documents

* Main overview: [`../README.md`](../README.md)
* System architecture: [`architecture.md`](architecture.md)
* Evaluation guide: [`evaluation.md`](evaluation.md)
* Runtime details: [`runtime-design.md`](runtime-design.md)
* Automation notes: [`automation.md`](automation.md)