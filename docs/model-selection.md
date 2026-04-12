# Model Selection and Rationale

This page explains why the current Mekong FNO system uses an **FNO-based forecasting core**, how that choice should be interpreted, and how the final deployed system should be understood in relation to:

- persistence
- upstream-assisted correction
- horizon-dependent performance
- operational deployment needs

This document is intended to answer a common but important question:

> **Why use FNO here, and what exactly is the role of FNO in the current deployed system?**

This page complements:

- [`../README.md`](../README.md)
- [`results.md`](results.md)
- [`evaluation.md`](evaluation.md)
- [`architecture.md`](architecture.md)

---

## 1. Short Answer

The current system uses **FNO as the data-driven core forecast model**, not as a claim that FNO by itself is universally strongest for station-level hydrologic forecasting.

A more accurate description of the deployed system is:

> **FNO-based data-driven core + hydrology-informed upstream-assisted correction + availability-aware evaluation and routing**

This distinction matters because the current deployed results show that:

- the **base FNO path** is not the strongest result at all horizons
- **persistence** remains strongest at very short horizons
- the strongest current deployed behavior comes from **assisted variants**, especially **Pakse-assisted** behavior at medium horizons
- the overall deployed interpretation depends not only on the model core, but also on:
  - upstream information
  - source availability
  - forecast horizon
  - routing policy

---

## 2. What “Model Selection” Means in This Project

In this project, model selection should not be interpreted as:

- a claim that one architecture universally dominates all others
- a claim that the system is only a benchmark competition between model families
- a claim that the final deployed value is explained only by the core forecast architecture

Instead, model selection here means:

- choosing a **reasonable data-driven core**
- integrating it into a **deployed forecasting service**
- evaluating it against **strong baselines**
- testing whether **hydrology-informed correction layers** improve it
- determining where the overall system is actually useful in practice

So the central question is not only:

> “Is FNO theoretically advanced?”

It is more practically:

> “Does an FNO-based core support a useful deployed station-level forecasting system when combined with the rest of the pipeline?”

---

## 3. Why FNO Was Chosen

## 3.1 FNO is a reasonable data-driven core for temporal pattern learning

The current forecasting task uses rolling input windows built from daily station-level data.  
Even in a single-station setting, such windows may contain:

- seasonal structure
- long-range smooth variation
- medium-horizon temporal patterns
- slowly evolving background behavior

FNO is a reasonable choice in this context because it can serve as a **global temporal pattern learner** over rolling windows, rather than only relying on very local step-to-step recurrence.

This does **not** mean that FNO is automatically optimal.  
It means that FNO is a defensible and technically coherent choice for the role of **data-driven core model**.

---

## 3.2 FNO fits the project’s medium-horizon ambitions better than a purely one-step framing

The current deployed results suggest that the system’s value is strongest at **medium horizons**, especially around **4–7 days**, rather than at **1-day persistence-dominated forecasting**.

That matters because a model family chosen for this project should not only support:

- next-step continuation

but also:

- broader temporal structure over longer windows
- forecast behavior where longer-horizon information matters more than immediate persistence

In that sense, using FNO as the core is consistent with the project’s **medium-horizon operational supplement** positioning.

---

## 3.3 FNO is used here as a model core, not as the entire system claim

This is one of the most important interpretation rules.

The project is **not** best read as:

> “FNO alone beats all alternatives”

It is better read as:

> “An FNO-based core provides the main data-driven forecast path, which can then be compared, corrected, and operationally interpreted within a deployed system.”

That is a much more accurate description of what the current application actually does.

---

## 4. Why FNO Should Not Be Overclaimed

## 4.1 FNO involving Fourier structure does not automatically imply hydrologic superiority

A common intuition is:

- hydrologic time series have seasonal patterns
- FNO uses Fourier-based structure
- therefore FNO should be better than LSTM, GRU, or similar models

This reasoning is understandable, but too strong.

In practice, whether FNO performs well depends on:

- the task structure
- the data scale
- the horizon
- the input variables available
- the strength of the baseline
- whether useful exogenous information is present

So while FNO can be appropriate, its use does **not** guarantee superior forecasting performance by itself.

---

## 4.2 Single-station forecasting is not necessarily the strongest possible FNO setting

FNO is especially associated with settings where global structure matters, often including:

- continuous fields
- operator-like mappings
- broader spatiotemporal patterns
- richer structured inputs

By contrast, the current project is:

- station-level
- daily
- relatively low-dimensional
- strongly persistence-influenced at short horizons

This does not make FNO inappropriate.  
It simply means that the current task is **not automatically the most favorable possible setting for FNO**.

That is one reason why the current deployed system should be understood through the full system design, not through architecture prestige alone.

---

## 4.3 The current results do not support the claim that “base FNO is universally strongest”

The current deployed evaluation shows that:

- **persistence** remains strongest at **1–3 day horizons**
- the **base FNO** path does not dominate the full-window comparisons
- the strongest deployed behavior is produced by **assisted variants**, especially Pakse-assisted behavior at medium horizons

So any model-selection explanation that implies:

- “FNO itself is already the strongest result”
- or “the project succeeds mainly because FNO dominates the baseline”

would overstate the evidence.

---

## 5. Why Persistence Must Remain Central in Model Selection

## 5.1 Persistence is a serious baseline in this task

For mainstream river-stage prediction at daily resolution, especially in slowly evolving settings, persistence is often a strong benchmark because:

- local inertia is high
- day-to-day continuity is strong
- short-horizon changes can be relatively modest

That means persistence is not just a trivial comparison.  
It is a meaningful operational reference point.

---

## 5.2 Beating persistence at all horizons is not the right expectation

A useful deployed forecasting system does not need to beat persistence uniformly at every horizon in order to be valuable.

The current results are more realistic:

- persistence dominates at **very short horizons**
- assisted medium-horizon behavior becomes more useful later

This is a stronger and more believable result structure than pretending one model should dominate everywhere.

---

## 5.3 Model selection should therefore be judged horizon-wise, not only by one global claim

This project’s model selection is best interpreted through a **horizon-aware lens**:

- at **1–3 days**, short-horizon inertia is dominant
- at **4–7 days**, upstream-assisted forecasting begins to add more practical value

This means the system’s value should be evaluated not just by “who wins overall,” but by:

- **where**
- **when**
- and **under what source conditions**

the system becomes useful.

---

## 6. Why the Deployed System Is Not “Pure FNO”

## 6.1 The current practical value comes from the combined system

The strongest current interpretation of the project is not:

> “FNO is the winner”

It is:

> “The deployed system combines an FNO-based forecast core with hydrology-informed correction and operational evaluation logic.”

That combined design includes:

- base FNO
- persistence comparison
- upstream-assisted variants
- common-overlap diagnostics
- source availability diagnostics
- routed operational interpretation

This is what makes the system credible as a deployed forecasting application.

---

## 6.2 Upstream-assisted correction is part of the deployed design, not a side experiment

3S and Pakse should not be viewed as loose afterthoughts.

In the current deployed system, they function as:

- forecast refinement paths
- diagnostic comparison paths
- operationally relevant assisted variants

This matters for model selection because it means the system is not best described as a **single-model comparison artifact**, but as a **core model plus hydrology-informed correction architecture**.

---

## 6.3 The strongest deployed result currently depends on Pakse-assisted behavior

The current evaluation shows that:

- **3S** can be stronger on fair same-date overlap
- **Pakse** has stronger coverage
- **Pakse-assisted** behavior is the strongest current full-window deployed variant
- routed behavior is therefore strongly influenced by Pakse availability

This tells us that model selection in this project is not only about neural architecture; it is also about:

- information pathways
- source availability
- operational use patterns

---

## 7. How to Interpret 3S vs Pakse in Model Selection

## 7.1 3S and Pakse answer different questions

The current diagnostics show that these two sources should not be collapsed into one simplistic ranking.

### 3S is important because:
- it is stronger on fair same-date overlap comparison
- it may contain stronger signal quality when both sources are simultaneously available

### Pakse is important because:
- it has broader operational availability
- it drives stronger full-window deployed behavior
- it currently supports the strongest routed operational configuration

This means the selection story is not:

> “Which source is always best?”

It is:

> “Which source is best in which evaluation context?”

---

## 8. Horizon-Dependent Interpretation of Model Choice

## 8.1 The current system is not strongest at all horizons

The current horizon-wise pattern is approximately:

- **1–3 days:** persistence strongest
- **4 days:** Pakse-assisted becomes competitive
- **5–7 days:** Pakse-assisted shows clearer gains

This should be understood as a meaningful result, not as a weakness to be hidden.

---

## 8.2 Why this supports the current system design

This pattern suggests that the current deployed system is most useful where:

- pure short-horizon inertia begins to weaken
- upstream information starts to matter more
- medium-horizon guidance becomes more valuable than one-step continuation

That makes the current FNO-based design a reasonable fit for:

> **medium-horizon station-level operational supplementation**

rather than for replacing very-short-horizon persistence.

---

## 9. Why This Is Still a Good Model Selection Story

A good model-selection story for a deployed ML system is not:

- “we picked the fanciest architecture”
- “the core model wins everywhere”
- “the system depends only on one neural design”

A better story is:

- the core model choice is technically coherent
- the baseline is strong and explicitly respected
- the results are interpreted honestly across horizons
- assisted variants are evaluated carefully
- the deployed system’s practical value emerges from the full pipeline, not just from branding the model

That is the story the current project supports.

---

## 10. What Should Not Be Claimed

The current project should **not** be described as proving any of the following:

- FNO is universally best for single-station hydrologic forecasting
- FNO automatically outperforms LSTM/GRU/TCN because it uses Fourier structure
- the base FNO path is strongest at all horizons
- the current deployed system fully replaces persistence
- the current system eliminates the need for mechanistic/platform-layer forecasting

Those claims would go beyond the current evidence.

---

## 11. What Can Be Claimed Confidently

The current project **can** support the following claims:

### 11.1
FNO is used as a **data-driven core model** for learning station-level temporal structure over rolling input windows.

### 11.2
The deployed system is **not** a pure FNO-only path; its practical value comes from combining:

- FNO-based forecasting core
- strong-baseline comparison
- hydrology-informed upstream-assisted correction
- availability-aware operational interpretation

### 11.3
The current strongest practical value of the system is as a **medium-horizon operational supplement**, not as a full replacement for short-horizon persistence.

### 11.4
The current model-selection story is stronger when told as a **system-level design decision** rather than as a neural-architecture competition.

---

## 12. Why This Matters for Portfolio Presentation

For hiring, review, and technical discussion, the current model-selection explanation is strongest when it is:

- technically literate
- honest about the evidence
- specific about the role of FNO
- careful not to overclaim architectural superiority

That means a good portfolio explanation is not:

> “We chose FNO because it is more advanced.”

A better explanation is:

> “We used FNO as the data-driven forecast core because it is a reasonable way to model medium-horizon temporal structure over rolling station-level input windows. The deployed system’s value, however, comes from combining that core with upstream-assisted correction, strong baseline comparison, and availability-aware evaluation.”

This is the most defensible form of the current selection rationale.

---

## 13. Future Model-Selection Directions

The current selection story may evolve if future versions introduce:

- upstream discharge inputs
- upstream/local rainfall inputs
- external platform-layer covariates
- multi-station modeling
- physics-informed extensions
- richer external forecast integration

If those are added, the role of FNO may also need to be re-evaluated in relation to:

- richer exogenous inputs
- stronger spatial structure
- hybrid model designs

That would be a future model-selection problem, not the current one.

---

## 14. Practical Summary

The current model-selection conclusion can be summarized as follows:

- FNO is a **reasonable and coherent choice** for the current project’s data-driven forecast core
- its use does **not** automatically imply universal superiority
- the strongest practical value of the deployed system currently comes from:
  - assisted medium-horizon behavior
  - especially Pakse-assisted full-window performance
  - evaluated against a strong persistence baseline
- the project is best interpreted as a **system-level forecasting design**, not just a claim about one model family

---

## 15. Related Documents

- Main overview: [`../README.md`](../README.md)
- Result summary: [`results.md`](results.md)
- Evaluation guide: [`evaluation.md`](evaluation.md)
- Architecture: [`architecture.md`](architecture.md)