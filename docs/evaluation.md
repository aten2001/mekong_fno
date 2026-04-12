# Evaluation Guide

This page explains how the current deployed evaluation workflow should be interpreted for the Mekong FNO system at Stung Treng.

It is intended to answer four questions:

1. **What exactly is being evaluated?**
2. **How are the current comparison tables constructed?**
3. **How should the reported metrics be interpreted?**
4. **What conclusions are valid, and what conclusions should be avoided?**

This document complements:

- [`../README.md`](../README.md)
- [`results.md`](results.md)

---

## 1. Evaluation Philosophy

The evaluation workflow in this project is designed to support a **deployed station-level forecasting service**, not just an offline model benchmark.

That means the current evaluation is built around:

- the same application used for live forecasting
- the same merged runtime data used by the deployed service
- explicit comparison against a strong persistence baseline
- explicit comparison between base and assisted variants
- diagnostics that distinguish:
  - source-specific gain
  - fair same-date comparison
  - source availability
  - routed operational behavior

This is important because many forecasting projects only report model-development metrics in isolation.  
This project instead evaluates the system in a form that is closer to how it would actually be used.

---

## 2. Evaluation Scope

### 2.1 Forecast target

The current evaluation focuses on:

- **station-level daily water-level forecasting**
- target station: **Stung Treng**
- current live forecasting horizon: **next 7 days**
- backtest horizons available in the deployed interface: **1–7 day ahead**

### 2.2 Evaluation period

The current deployed evaluation interface supports retrospective backtesting from:

- **2025-01-01**
- through the **latest available runtime date**

This means the evaluation window is not fixed forever.  
As new runtime data and refreshed API-fetched daily values become available, the latest backtest date may move forward.

### 2.3 What changes over time

Because the application uses refreshed runtime data, some evaluation outputs may drift slightly over time:

- the final backtest end date
- source availability counts
- overlap counts
- routed-policy counts
- RMSE / MAE values

This is normal for a deployed forecasting service and does not by itself indicate a modeling error.

---

## 3. Forecast Variants Evaluated

The deployed evaluation currently compares the following forecast variants.

### 3.1 Persistence

Persistence is treated as an explicit strong baseline.

Conceptually:

- the forecast is anchored to the last available observation at the appropriate horizon
- this is a hard-to-beat benchmark in slowly varying mainstream river-stage prediction

In this project, persistence should be treated as a **serious operational baseline**, not a weak placeholder.

---

### 3.2 Base FNO

This is the core data-driven forecast path before upstream-assisted correction.

It represents:

- the deployed model’s base prediction
- the main learned station-level data-driven forecast component
- the reference forecast to which assisted variants are compared

Base FNO should **not** be interpreted as the whole deployed system by itself.

---

### 3.3 FNO + 3S

This is the base FNO forecast after applying upstream-assisted residual correction using:

- 3S daily level input
- lag-aligned upstream features
- a fitted residual-correction model

This variant is most useful when evaluating:

- whether 3S contributes predictive value
- whether 3S is strong on fair same-date overlap comparison
- whether 3S improves base FNO when its source is available

---

### 3.4 FNO + Pakse

This is the base FNO forecast after applying upstream-assisted residual correction using:

- Pakse daily level input
- lag-aligned upstream features
- a fitted residual-correction model

This variant is particularly important because, in the current deployed results:

- it produces the strongest full-window assisted behavior
- it benefits from broader source coverage than 3S

---

### 3.5 Routed operational policy

The current routed operational policy is:

- **Pakse > 3S > FNO fallback**

Meaning:

- if Pakse-assisted output is available, use it
- otherwise if 3S-assisted output is available, use it
- otherwise fall back to base FNO

This routed configuration is meant to approximate real deployed behavior under source-availability constraints.

---

## 4. Core Metrics

### 4.1 RMSE

The main evaluation metric highlighted in the current deployed interface is **RMSE**.

RMSE is useful because:

- it is sensitive to larger errors
- it is easy to compare across forecast variants
- it is a practical summary for station-level continuous water-level prediction

In this project, RMSE is the primary metric used in:

- headline tables
- horizon-wise comparisons
- routed-performance comparisons

---

### 4.2 MAE

MAE is also reported in selected evaluation outputs.

MAE is useful because:

- it is easier to interpret as average absolute deviation
- it is less dominated by large outliers than RMSE

Where both RMSE and MAE are shown, they should be interpreted together.

---

### 4.3 Why the interface emphasizes RMSE

The deployed UI is designed to remain readable.  
Because of that, the main note and summary views emphasize RMSE, while more detailed interpretation can be supported by:

- advanced diagnostics
- `results.md`
- future extended evaluation pages

---

## 5. Evaluation Views in the Current System

The current evaluation results should be read through **three different contexts**.

This is critical.

### 5.1 Full-window evaluation

This is the most deployment-like comparison.

It evaluates a forecast variant across the full backtest window for a given horizon.

Typical examples:

- persistence baseline (full window)
- base FNO (full window)
- FNO + Pakse (full window)

This is the best place to answer questions like:

- Which variant behaves best over the whole deployed evaluation window?
- Does the strongest assisted version beat persistence on the full window?

---

### 5.2 Common-overlap evaluation

This is the fair same-date comparison view.

It evaluates two assisted variants only on dates where both are available.

This view is essential because source-specific full-window comparisons can be misleading when:

- one source is available on more dates than another
- the available-date subsets differ in difficulty
- a broader-coverage source appears stronger simply because it is evaluated on a different set of dates

This is the right place to answer:

- Which assisted source is stronger on the same dates?

---

### 5.3 Routed operational evaluation

This is the deployment-oriented routing view.

It evaluates what happens when the system uses a real availability-aware decision rule, such as:

- Pakse if available
- otherwise 3S
- otherwise base FNO

This is the right place to answer:

- How does the current deployed policy behave over the full window?
- What actually gets used in practice?
- How much does source coverage affect deployed performance?

---

## 6. Why Source-Specific Tables Must Be Read Carefully

One of the most important interpretation rules in this project is:

> **source-specific subset tables are not automatically direct head-to-head rankings**

This matters because:

- 3S and Pakse are not available on exactly the same dates
- their source-availability windows differ
- the “difficulty” of those subsets may differ
- full-window operational behavior depends on routing and coverage, not only source quality

### 6.1 Example of the interpretation trap

Suppose:

- 3S is stronger on fair overlap dates
- Pakse has much wider availability

Then it is possible for:

- 3S to be the stronger **same-date** source
- while Pakse still produces better **full-window deployed** behavior

That is exactly the kind of distinction this project’s diagnostics are designed to surface.

---

## 7. Horizon-Wise Interpretation

The 1–7 day evaluation is one of the most important parts of the current result interpretation.

### 7.1 Current horizon-dependent structure

The current results show a realistic structure:

- **1–3 day horizons:** persistence remains strongest
- **around 4 days:** Pakse-assisted performance becomes competitive
- **5–7 day horizons:** Pakse-assisted behavior shows clearer gains

### 7.2 Why this matters

This means the current system should not be described as:

- “always stronger than persistence”
- “uniformly better at every horizon”

A more accurate interpretation is:

> the current deployed system is most valuable as a **medium-horizon station-level operational supplement**

rather than as a replacement for very-short-horizon persistence.

### 7.3 Why this is realistic in hydrologic forecasting

For mainstream river-stage prediction, especially at daily resolution:

- short horizons are often dominated by persistence and strong local inertia
- medium horizons are where upstream information begins to provide more useful forward-looking value

So this horizon-dependent structure is not a weakness in itself.  
It is often a more realistic result than a system that appears to outperform every baseline at every horizon.

---

## 8. Interpreting 3S vs Pakse

### 8.1 What can be said confidently

Current diagnostics support the following interpretation:

- **3S** can be stronger on **fair same-date overlap**
- **Pakse** has stronger **source availability / coverage**
- **Pakse-assisted full-window behavior** is currently stronger in deployment-like evaluation
- the routed operational policy is therefore currently driven mainly by **Pakse coverage**

### 8.2 What should not be said

The following simplifications should be avoided:

- “Pakse is better than 3S in every sense”
- “3S is the best deployed source”
- “coverage does not matter”
- “the strongest overlap source is automatically the strongest deployed source”

Those would overstate the results.

---

## 9. What the Current Results Do Support

The current evaluation supports these claims:

### 9.1 Supported claim A
The deployed system is more than a single-model forecast display.

It supports:

- baseline comparison
- assisted correction
- fair same-date diagnostics
- availability-aware reasoning
- routed operational interpretation

### 9.2 Supported claim B
The current strongest deployed configuration can outperform both the base FNO model and the persistence baseline at the 7-day setting.

### 9.3 Supported claim C
The main current value of the system is in **medium-horizon station-level forecasting**, not in replacing short-horizon persistence.

### 9.4 Supported claim D
Operational usefulness depends on:

- source quality
- source coverage
- forecast horizon
- routing policy

not just on the core model alone.

---

## 10. What the Current Results Do Not Yet Support

The current evaluation does **not** justify stronger claims such as:

- “FNO itself is strongest at all horizons”
- “the system universally outperforms simple baselines”
- “the current approach replaces process-based or platform-layer forecasting systems”
- “Pakse is universally stronger than 3S”
- “the current deployed policy is necessarily optimal”

Those would go beyond the evidence currently shown.

---

## 11. Current Evaluation Limitations

### 11.1 Scope limitation
The current evaluation is centered on:

- one target station
- daily water level
- 1–7 day ahead horizons
- current deployed assisted variants

### 11.2 Metric limitation
The current deployed summaries emphasize:

- RMSE
- selected MAE outputs

Future extensions may include richer reporting such as:

- seasonal breakdowns
- threshold-event evaluation
- exceedance-oriented measures
- uncertainty calibration metrics

### 11.3 Availability limitation
Assisted variants depend on source availability.

This means:

- source-specific full-window comparisons are affected by source coverage
- routing results depend on current policy design
- full-window operational behavior is not explained by model quality alone

### 11.4 Platform-layer integration limitation
The current evaluation does not yet directly include:

- platform-layer external forecasts as exogenous inputs
- correction targets
- ensemble members
- residual-model covariates from basin-scale systems

---

## 12. How to Read the Current Tables Correctly

A good evaluation reading order is:

### Step 1: Read the full-window rows
Ask:

- does the strongest assisted variant beat persistence?
- does the strongest assisted variant beat base FNO?

### Step 2: Read the horizon-wise table
Ask:

- at which horizons does persistence dominate?
- at which horizons does Pakse-assisted behavior become useful?

### Step 3: Read the overlap comparison
Ask:

- on fair same-date comparison, which source is actually stronger?

### Step 4: Read the availability summary
Ask:

- how often is each source available?
- how much does coverage differ?

### Step 5: Read the routed operational table
Ask:

- what does the deployed policy actually use?
- how much of the full-window result is driven by coverage and fallback structure?

This reading order helps prevent over-interpretation.

---

## 13. Practical Summary

The current evaluation should be summarized as follows:

- the project is not just a model benchmark; it is a deployable evaluated forecasting system
- persistence remains a strong baseline, especially at very short horizons
- Pakse-assisted behavior is currently the strongest full-window deployed variant
- 3S is stronger on fair same-date overlap comparison
- deployed operational performance depends on source coverage and routing policy
- the current system is most convincing as a **medium-horizon station-level operational supplement**

---

## 14. Related Pages

- Main project overview: [`../README.md`](../README.md)
- Current result summary: [`results.md`](results.md)
- Future model-selection notes: [`model-selection.md`](model-selection.md)

---

## 15. Notes

- Slight numerical drift may occur as refreshed runtime data are incorporated.
- Current evaluation values should always be interpreted in context:
  - full-window
  - common-overlap
  - routed operational
- These contexts answer different questions and should not be collapsed into a single simplistic ranking.