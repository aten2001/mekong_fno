# Automation Guide

This page explains how automation fits into the Mekong FNO system as part of a **deployable station-level forecasting and evaluation service**.

The automation layer in this project is not just a convenience add-on.  
It plays an operational role by helping the system keep its runtime inputs and derived artifacts more current over time.

This document complements:

- [`../README.md`](../README.md)
- [`deployment.md`](deployment.md)
- [`runtime-design.md`](runtime-design.md)
- [`architecture.md`](architecture.md)

---

## 1. Why Automation Exists in This Project

The project is not intended to remain:

- a one-time offline experiment
- a static model snapshot
- a manually refreshed demo page

Instead, the system is designed to behave like a lightweight operational service with:

- live forecasting
- retrospective evaluation
- refreshable runtime state
- deployment-aware data handling

To support that behavior, the system needs a repeatable path for:

- refreshed data
- regenerated artifacts
- updated runtime inputs
- stable deployment-side consumption

Automation exists to support that path.

---

## 2. Role of Automation in the Current System

The automation layer is currently used to support **scheduled backfill publishing** and related operational artifact refresh behavior.

Typical repository location:

- `.github/workflows/publish_backfill.yml`

In practical terms, automation helps connect:

1. refreshed or newly available data
2. generated backfill/runtime artifacts
3. the deployed application’s runtime behavior

This means automation is part of the operational system boundary, not just a repository maintenance detail.

---

## 3. What the Current Automation Supports

The current automation is most naturally understood as supporting three goals.

### 3.1 Keep runtime-facing artifacts current

The deployed app depends not only on source code, but also on:

- recent daily values
- backfill artifacts
- cached operational products
- evaluation-relevant derived files

Automation helps maintain a repeatable way to update or publish those artifacts.

---

### 3.2 Reduce manual operational steps

Without automation, the operational path would depend more heavily on manual actions such as:

- manually refreshing artifacts
- manually republishing derived files
- manually moving updated outputs into deployment-accessible locations

Automation reduces this burden and makes the system more consistent.

---

### 3.3 Strengthen the project as a deployable system

From a portfolio and engineering perspective, automation is valuable because it shows that the project is not only:

- train model
- launch UI
- inspect output

It also includes an operational update path, which is a strong signal of system maturity.

---

## 4. Current Automation Boundary

The current automation layer should be understood as supporting:

- scheduled artifact refresh / publication
- repository-driven operational updates
- better alignment between refreshed data and deployed behavior

It should **not** be overstated as a full production pipeline with:

- full orchestration
- complex monitoring
- full observability stack
- external workflow platform integration

The current automation is meaningful and useful, but intentionally lightweight.

---

## 5. High-Level Automation Flow

```mermaid
flowchart TD
    A[Scheduled GitHub Actions workflow] --> B[Backfill / update logic]
    B --> C[Generate or refresh artifacts]
    C --> D[Publish updated artifacts]
    D --> E[Repository / deployment-visible state]
    E --> F[HF Space runtime access]
    F --> G[Reloadable forecasting application]
````

This flow captures the intended role of the current automation layer:

* generate or refresh operationally relevant files
* make them available to the deployed system
* allow the deployed app to pick them up through its runtime logic

---

## 6. Main Automation Components

## 6.1 GitHub Actions workflow

The primary automation surface is GitHub Actions.

Typical workflow file:

* `.github/workflows/publish_backfill.yml`

Its role is to provide a scheduled, repository-native update path.

This is a good fit for the current project because:

* the repository already hosts source code and docs
* the deployment model is repository-centric
* the automation needs are currently moderate rather than highly distributed

---

## 6.2 Backfill / artifact generation logic

The workflow is tied to project logic that can generate or refresh artifacts used by the application runtime.

Conceptually, this logic may include:

* collecting refreshed data
* rebuilding backfill products
* updating files used by the runtime service
* preparing deployment-relevant outputs

The exact artifact structure is less important here than the architectural role:

> automation helps move refreshed inputs into a state that the deployed app can consume

---

## 6.3 Published outputs

The outputs of automation are typically not “final user reports” in the business-document sense.
They are operational inputs or intermediates such as:

* refreshed backfill files
* runtime cache inputs
* evaluation-supporting derived artifacts

This makes the automation layer tightly connected to runtime behavior.

---

## 7. How Automation Relates to Deployment

Automation and deployment are separate concepts, but in this project they are tightly linked.

### 7.1 Deployment answers:

* Where does the app run?
* How does the user access it?
* Where does runtime state live?

### 7.2 Automation answers:

* How do refreshed operational artifacts get produced?
* How do deployment-visible inputs get updated?
* How does the system reduce manual maintenance?

So the current system should be thought of as:

> **deployment + runtime design + automation**

not as deployment alone.

---

## 8. How Automation Relates to Runtime Behavior

The deployed app is built around a reloadable runtime service.
Automation helps ensure that there is useful refreshed state for that runtime service to consume.

This means the relationship is:

* automation updates artifacts or deployment-visible inputs
* runtime logic knows where to look for them
* the application reload path can pick them up

This architecture is stronger than having:

* manual artifact drops
* inconsistent local-only updates
* undocumented update paths

---

## 9. Why Automation Matters for a Forecasting Service

For a forecasting application, the usefulness of the deployed system depends partly on how fresh and repeatable its data path is.

Automation matters because it helps bridge the gap between:

* static code
* evolving operational data

That is especially important in a project like this, where the goal is not merely to display a trained model, but to maintain a forecasting service that can:

* be reloaded
* be re-evaluated
* stay tied to newer runtime inputs

---

## 10. Current Operational Story

The current operational story of the project is roughly:

1. source code and docs live in GitHub
2. automation helps refresh / publish relevant artifacts
3. the deployed HF Space consumes repository/runtime-visible state
4. the app exposes reload behavior to pick up newer state
5. the user can inspect:

   * live forecasts
   * historical evaluation
   * advanced diagnostics

This gives the project a much stronger “system” identity than a simple one-time notebook workflow.

---

## 11. Practical Automation Advantages

The current automation layer provides several practical benefits.

### 11.1 Repeatability

A workflow file is easier to inspect and reproduce than a loosely remembered set of manual steps.

### 11.2 Traceability

Because the automation is defined in the repository, it becomes easier to inspect:

* what is supposed to run
* when it runs
* which files or scripts are involved

### 11.3 Lower maintenance burden

Automation reduces the need to repeatedly perform operational update steps by hand.

### 11.4 Stronger project credibility

For a deployed ML portfolio project, automation is one of the clearest signals that the system is intended for repeated use rather than one-off demonstration.

---

## 12. Current Constraints of the Automation Layer

The automation in this project is useful, but it should be described honestly.

### 12.1 It is not a full production orchestration platform

The current workflow does not imply:

* enterprise workflow orchestration
* distributed job scheduling across many services
* heavy-duty observability and alerting
* external pipeline management platforms

### 12.2 It depends on the repository-centric deployment model

The current automation works well because the project is organized around:

* GitHub as source-of-truth
* HF Space as deployment surface
* runtime consumption of repository-visible or generated artifacts

### 12.3 It supports operational refresh, not complete platform integration

The automation layer currently helps maintain deployment-facing artifacts, but it does not yet imply:

* direct ingestion from basin-scale platform forecasts
* cross-system forecast federation
* broader data-platform coupling

---

## 13. Common Automation-Related Failure Modes

## 13.1 Workflow exists but artifacts are not visible to deployment

Possible causes:

* artifact path mismatch
* publication step not writing where expected
* deployment/runtime root not aligned with published outputs

### Practical consequence

The app may continue using stale runtime state even though automation ran.

---

## 13.2 Code updated but artifact-linked docs or figures were not committed

This is especially relevant when README or docs refer to files such as:

* figures
* report pages
* asset paths

If those files are not:

* tracked by Git
* committed
* merged into `main`
* pushed to the target remote

then repository-side rendering may break.

---

## 13.3 Automation updates outputs but app has not reloaded state

Even if automation refreshes files correctly, the deployed app may not immediately reflect changes unless:

* startup re-reads them
* reload behavior is triggered
* runtime state is rebuilt

This is why reload is an important companion feature to automation.

---

## 13.4 Divergence between GitHub and HF remotes

If GitHub `main` and HF `space/main` diverge, automation-related updates may appear inconsistent across surfaces.

This can happen if:

* one remote is updated but not the other
* HF receives direct web edits
* pushes are rejected because of non-fast-forward state

In practice, this means repository synchronization remains part of operational discipline.

---

## 14. Recommended Interpretation in Portfolio Context

For portfolio and hiring purposes, the right way to describe the automation layer is:

> the project includes a lightweight but meaningful automation path for maintaining refreshed operational artifacts that support the deployed forecasting application

This is a strong statement because it is:

* true
* concrete
* engineering-relevant
* appropriately scoped

It is better than either of these two extremes:

### Too weak

* “there is some GitHub Actions file”

### Too strong

* “this is a full production-grade MLOps platform”

The current automation is valuable, but it should be described proportionally.

---

## 15. Future Automation Extensions

Natural future extensions include:

* richer automated report generation
* better human-readable result publication
* stronger drift / availability monitoring
* threshold-based evaluation reporting
* more explicit runtime-health or artifact-health checks
* more formal synchronization between generated artifacts and deployment-visible runtime state

These are reasonable future directions, but they are not required for the current automation layer to already be meaningful.

---

## 16. Practical Reading Guide

If someone wants to inspect the project’s automation story, the most useful reading path is:

1. root [`../README.md`](../README.md)
2. current deployment notes in [`deployment.md`](deployment.md)
3. this page (`automation.md`)
4. the workflow file:

   * `.github/workflows/publish_backfill.yml`
5. the runtime design page:

   * [`runtime-design.md`](runtime-design.md)

This reading order makes it easier to understand how:

* repository workflows
* artifact refresh
* runtime consumption
* deployed application behavior

fit together.

---

## 17. Automation Summary

The automation layer in the Mekong FNO project is best understood as:

* **lightweight**
* **repository-native**
* **operationally meaningful**
* **artifact-oriented**
* **deployment-supporting**

Its role is not to make the project look larger than it is.

Its role is to make the project more real by showing that the deployed forecasting application is supported by a repeatable refresh path for the data products and artifacts it depends on.

---

## 18. Related Documents

* Main overview: [`../README.md`](../README.md)
* Deployment notes: [`deployment.md`](deployment.md)
* Runtime design: [`runtime-design.md`](runtime-design.md)
* System architecture: [`architecture.md`](architecture.md)