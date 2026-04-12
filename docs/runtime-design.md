# Runtime Design

This page explains how runtime state is organized in the Mekong FNO system and why the runtime layer is a core part of the project’s architecture.

The runtime design is important because this project is not intended to behave like:

- a one-shot notebook
- a static model checkpoint demo
- a script collection with no persistent operational state

Instead, the project is designed to behave like a **deployable station-level forecasting and evaluation service** with:

- refreshable data inputs
- reusable caches
- persisted artifacts
- explicit reload behavior
- consistent local / hosted runtime paths

This document complements:

- [`../README.md`](../README.md)
- [`architecture.md`](architecture.md)
- [`deployment.md`](deployment.md)
- [`automation.md`](automation.md)

---

## 1. Why Runtime Design Matters

A forecasting application is not just:

- source code
- model weights
- a UI

It also depends on evolving operational state such as:

- refreshed daily values
- cached merged series
- backfill artifacts
- evaluation-related outputs
- assist parameter bundles
- runtime reports

If these files are mixed indiscriminately with static project assets, the system becomes harder to:

- reason about
- debug
- refresh
- deploy consistently

The runtime design exists to keep that operational state explicit and manageable.

---

## 2. Core Runtime Principle

The most important runtime principle in this project is:

> **separate stable project assets from generated operational state**

This means the system distinguishes between:

### Static project resources
Files that are expected to be relatively stable across runs, such as:

- source code
- model checkpoints
- committed historical data
- static evaluation / inference assets

### Runtime-generated resources
Files that may be created, refreshed, or replaced during operation, such as:

- live caches
- backfill artifacts
- runtime evaluation caches
- assist parameter caches
- derived operational outputs

This separation is one of the strongest engineering signals in the project.

---

## 3. Runtime vs Static Layers

## 3.1 Static layer

Typical static directories include:

- `app/`
- `src/`
- `assets/`
- `weights/`
- `data/`

These represent the stable repository-defined part of the system.

### Examples of static content
- application code
- modeling utilities
- committed historical station files
- static inference assets
- checkpoint files

These files are version-controlled as project resources.

---

## 3.2 Runtime layer

The runtime layer holds deployment-time or operation-time state.

Typical examples include:

- current live caches
- backfill outputs
- assist fit bundles
- runtime evaluation artifacts
- generated state used by the deployed app

These files are operationally meaningful, but they are not part of the static source bundle in the same sense as committed assets.

---

## 4. Runtime Design Goals

The current runtime design supports five goals.

### 4.1 Stable operational file layout
The application should know where to read and write operational files in a repeatable way.

### 4.2 Refreshability
Newer runtime data should be able to enter the system without redesigning the whole app.

### 4.3 Reusability
Previously generated runtime outputs should be reusable across app runs when appropriate.

### 4.4 Deployment compatibility
The same runtime logic should work both:

- locally
- in Hugging Face Spaces

### 4.5 Explicit state management
The system should make it clear which files belong to:

- repository source content
- deployment-time operational state

---

## 5. Runtime Root

## 5.1 What the runtime root is

The runtime root is the main directory under which operational files are stored.

It acts as the home for runtime-generated state such as:

- caches
- refreshed data products
- derived evaluation outputs
- assist-related parameter files

The exact path may differ between environments, but the conceptual role is the same.

---

## 5.2 Local runtime root

In local execution, the system typically uses a project-local runtime directory.

This is useful because it allows:

- local debugging
- local inspection of generated artifacts
- repeat local testing without polluting static source directories

---

## 5.3 Hosted runtime root

In Hugging Face Spaces, runtime files are expected to use persistent storage when available.

This supports:

- survival of generated files across sessions
- continuity of refreshed operational state
- better alignment between deployment and runtime behavior

This is why runtime design matters directly to deployment.

---

## 6. Main Runtime File Categories

The runtime layer can be understood as several logical file categories.

## 6.1 Live data caches

These are runtime-side files used to hold recently refreshed data or related cache products.

Their purpose is to reduce repeated work and make newer daily values available to the app.

### Role
- avoid redundant retrieval / regeneration
- support refresh workflows
- stabilize current data access

---

## 6.2 Backfill artifacts

These are derived outputs related to refreshed or extended historical data products.

Their role is to help connect:

- source refresh
- evaluation continuity
- deployment-visible operational state

Backfill artifacts are especially important because this project is intended to support ongoing backtesting and not only frozen offline evaluation.

---

## 6.3 Evaluation caches

Evaluation-related intermediate files may also live in the runtime layer.

Their role is to support:

- faster repeated evaluation
- consistent app-side diagnostics
- reuse of expensive or frequently accessed derived outputs

These caches help the system behave like a repeated-use application rather than a one-off script.

---

## 6.4 Assist parameter caches

The current deployed system includes upstream-assisted correction behavior using 3S and Pakse.  
Associated fit bundles or derived parameter files may be stored in runtime-visible locations.

These files are useful because they separate:

- base model behavior
- assisted operational outputs
- derived fitting state

This is especially important for a deployed system that includes:

- source-specific assist paths
- same-date diagnostics
- routing-oriented interpretation

---

## 6.5 Runtime reports and derived outputs

Some runtime-visible outputs may be more report-like in nature, such as:

- evaluation summaries
- generated comparison artifacts
- auxiliary diagnostic outputs

These still belong to runtime state if they are:

- refreshed
- generated
- deployment-facing

rather than static authored documents.

---

## 7. Process-Level Runtime Service

## 7.1 Why a runtime service exists

The app is built around a process-level cached service object that holds loaded system state.

This service helps unify:

- historical and refreshed daily series
- upstream series
- model state
- runtime metadata
- evaluation helpers

Without this layer, each callback would have to rebuild more of the system from scratch.

---

## 7.2 What the service encapsulates

Conceptually, the runtime service encapsulates:

- target-station time series
- upstream time series
- merged operational daily state
- model and related assets
- forecast helpers
- evaluation helpers
- runtime-dependent paths / metadata

This is what allows the UI to behave like an application rather than a loose set of scripts.

---

## 7.3 Why this matters for runtime design

The runtime layer is not only about files on disk.  
It is also about how those files are gathered into a coherent loaded service state.

That is why runtime design in this project includes both:

- filesystem layout
- process-level cached service behavior

---

## 8. Reload Behavior

## 8.1 Purpose of reload

The deployed app exposes an explicit reload behavior.

Its purpose is to allow the application to:

- rebuild service state
- re-read newer runtime inputs
- pick up refreshed operational files

without requiring a full manual restart workflow every time.

---

## 8.2 Why reload is runtime-specific

Reload exists because runtime state can change independently of static code.

For example:

- refreshed daily values may appear
- new backfill artifacts may become available
- cached operational state may need rebuilding

If runtime state never changed, reload would not be necessary.

---

## 8.3 Reload as a runtime boundary signal

The presence of explicit reload behavior is one of the clearest signs that the project is designed as a **runtime-aware service** rather than a static demonstration.

---

## 9. Runtime and Evaluation

## 9.1 Why evaluation depends on runtime state

The current system evaluates not only historical source data, but also the merged runtime-visible state used by the deployed application.

That means evaluation is tied to:

- the current available daily series
- current backfill coverage
- current source availability
- current assist parameters
- current routing behavior

This is why evaluation values may drift slightly as runtime inputs evolve.

---

## 9.2 Why this is acceptable

This project is a deployed forecasting service, not a frozen benchmark package.

So some runtime-linked variation is expected and acceptable, as long as:

- the evaluation scope remains clear
- the interpretation remains honest
- the runtime path remains reproducible

---

## 10. Runtime and Automation

## 10.1 How automation feeds runtime

Automation helps maintain refreshed or generated artifacts that the runtime layer may later consume.

The relationship is:

- automation updates or publishes operationally relevant files
- runtime logic knows where those files live
- reload behavior allows the app to pick them up

This makes the runtime layer the bridge between:

- scheduled update workflows
- deployed application behavior

---

## 10.2 Why runtime is central to the automation story

Without a clean runtime design, automation would have less value because there would be no clear place for refreshed outputs to enter the system.

That is why runtime design is a foundational part of the broader operational architecture.

---

## 11. Runtime and Deployment

## 11.1 Local deployment

In local runs, runtime design supports:

- a project-local runtime root
- repeat local testing
- easier artifact inspection
- local debugging of refresh behavior

## 11.2 Hosted deployment

In Hugging Face Spaces, runtime design supports:

- use of persistent storage when available
- survival of generated state across sessions
- more realistic hosted operational behavior

This means runtime design is not a backend-only concern; it directly affects how the deployed application behaves.

---

## 12. Path Management Principles

The runtime layer works best when path logic is centralized and explicit.

The project should avoid runtime behavior that relies on:

- scattered hardcoded paths
- implicit file locations
- ad hoc write locations
- mixed static/runtime directories without discipline

Instead, runtime path resolution should be:

- centralized
- explicit
- environment-aware
- consistent between local and hosted modes

This makes the system easier to maintain and less fragile during deployment.

---

## 13. Concurrency and Write Safety

Because runtime files may be updated, cached, or regenerated, runtime design should also consider:

- write ordering
- overwrite behavior
- cache validity
- partial-write avoidance
- multi-callback stability

In practical terms, this is why patterns such as:

- file locking
- atomic writes
- explicit runtime file helpers

are valuable in the project.

Even when the deployment is lightweight, runtime correctness still matters.

---

## 14. Common Runtime Failure Modes

## 14.1 File exists locally but is not part of repository or runtime design

This can happen when a file is created manually on disk but is neither:

- committed as a static asset
- nor formally managed as runtime state

This often causes confusion because the file appears locally but not in remote or deployed environments.

---

## 14.2 Static and runtime files mixed together

If generated outputs are written into static project directories without clear intent, it becomes harder to know:

- what should be committed
- what should persist
- what should be regenerated
- what should be ignored

This weakens reproducibility.

---

## 14.3 Reload invoked but runtime inputs have not actually changed

Reload only helps if refreshed runtime-visible inputs are present.  
If automation or upstream data refresh has not changed anything meaningful, reload will rebuild state but not necessarily change outputs.

---

## 14.4 Hosted deployment cannot see refreshed operational state

This can happen if:

- runtime root is misconfigured
- persistent storage is missing or inconsistent
- published artifacts are not reaching the expected path
- repository and hosted remotes diverge

This is why runtime design must be read together with deployment and automation design.

---

## 15. Why Runtime Design Is a Strength of This Project

For portfolio and engineering evaluation, runtime design is one of the clearest signals that the project is more than a model demo.

It shows that the project accounts for:

- state
- refresh
- persistence
- cache reuse
- deployment consistency
- evaluation continuity

These are all properties of systems, not just models.

That is a strong point in favor of the current project architecture.

---

## 16. Current Runtime Boundaries

The runtime layer is intentionally designed for the current scope of the project:

- station-level forecasting
- one main target station
- current assisted variants
- hosted Gradio app
- repository-centric operational flow

It is **not yet** intended as:

- a distributed runtime platform
- a large multi-service data platform
- a basin-scale orchestration framework

This limitation is intentional and keeps the runtime design aligned with the project’s actual operational role.

---

## 17. Future Runtime Extensions

Reasonable future runtime-oriented extensions include:

- richer runtime health checks
- more formal cache invalidation policies
- stronger reporting of artifact freshness
- more explicit runtime drift tracking
- support for additional external covariates
- more structured runtime summaries for diagnostics and results pages

These are future improvements, not requirements for the current runtime design to already be meaningful.

---

## 18. Runtime Summary

The runtime design of the Mekong FNO project can be summarized as:

- a clear separation between static assets and generated operational state
- a dedicated runtime root for refreshable files
- a cached process-level service for loaded system state
- explicit reload behavior for runtime refresh
- deployment-aware handling of persistent operational artifacts

This runtime layer is one of the main reasons the project behaves like a **real forecasting application** rather than only a code artifact.

---

## 19. Related Documents

- Main overview: [`../README.md`](../README.md)
- Architecture: [`architecture.md`](architecture.md)
- Deployment: [`deployment.md`](deployment.md)
- Automation: [`automation.md`](automation.md)
- Evaluation: [`evaluation.md`](evaluation.md)