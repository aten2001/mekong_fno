# Results Summary

This page summarizes the current deployed evaluation results for the Mekong FNO system at Stung Treng. It complements the main `README.md` and the live Hugging Face Space by providing a compact, human-readable interpretation of the current backtest outputs.

> **Important note**
> The values below reflect the current deployed evaluation snapshot and may change slightly as runtime data, backfill artifacts, or refreshed API-fetched daily values are updated.

## 1. Scope of Evaluation

The deployed evaluation interface supports:

- retrospective backtesting from **2025-01-01** to the latest available date
- configurable **1–7 day-ahead** forecast horizons
- comparison against:
  - **Persistence**
  - **Base FNO**
  - **FNO + 3S**
  - **FNO + Pakse**
- advanced availability-aware diagnostics for:
  - source-specific gain
  - fair same-date comparison
  - source availability
  - routed operational performance

This means the system is evaluated in the same operational frame in which it is presented, rather than only through offline model-development metrics.

---

## 2. Headline Result

At the **7-day-ahead** setting, the strongest deployed assisted/routed configuration currently outperforms both the base FNO model and the persistence baseline. This is the clearest current evidence that the deployed system has practical value beyond a standalone baseline forecast.

At the same time, the **1–7 day horizon** results show that this advantage is **not uniform across all forecast lengths**:

- **Persistence** remains strongest at **1–3 day horizons**
- **FNO + Pakse** becomes competitive around **4 days**
- **FNO + Pakse** shows clearer gains at **5–7 days**

This suggests that the current deployed system is best interpreted as a **medium-horizon station-level operational supplement**, rather than as a replacement for short-horizon persistence.

---

## 3. 7-Day Headline Metrics

### 3.1 7-day full-window and overlap summary

| Result view | RMSE (m) | MAE (m) | Interpretation |
|---|---:|---:|---|
| Persistence baseline (full window, 7-day setting) | 0.780 | — | Strong baseline over the full backtest window |
| Base FNO (full window, 7-day setting) | 0.852 | 0.524 | Weaker than persistence on the full window |
| FNO + 3S (full window, 7-day setting) | 0.803 | — | Improves base FNO, but not the best deployed full-window result |
| FNO + Pakse (full window, 7-day setting) | 0.689 | — | Best single assisted variant on the full evaluation window |
| FNO + 3S (common overlap dates only, 7-day setting) | 0.552 | 0.388 | Stronger than FNO + Pakse when both sources are available on the same dates |
| FNO + Pakse (common overlap dates only, 7-day setting) | 0.579 | 0.407 | Slightly weaker than 3S on fair same-date comparison |
| Routed operational policy (Pakse > 3S > FNO fallback, 7-day setting) | 0.689 | 0.464 | Best current deployed operational behavior; mainly driven by Pakse coverage |

### 3.2 How to read this table

The rows above serve **different purposes** and should **not** be treated as directly interchangeable rankings:

- **Full-window rows** describe deployed behavior across the whole backtest period
- **Common-overlap rows** describe fair same-date comparison only on dates where both 3S and Pakse are available
- **Routed operational rows** describe the current real deployment behavior under source availability constraints

---

## 4. Horizon-Wise RMSE Summary (1–7 Days)

| Horizon | Persistence RMSE (m) | FNO RMSE (m) | FNO + 3S RMSE (m) | FNO + Pakse RMSE (m) | Best current full-window variant |
|---|---:|---:|---:|---:|---|
| 1 day | 0.165 | 0.492 | 0.439 | 0.432 | Persistence |
| 2 days | 0.302 | 0.529 | 0.468 | 0.442 | Persistence |
| 3 days | 0.423 | 0.583 | 0.520 | 0.472 | Persistence |
| 4 days | 0.531 | 0.652 | 0.591 | 0.524 | FNO + Pakse |
| 5 days | 0.625 | 0.726 | 0.668 | 0.583 | FNO + Pakse |
| 6 days | 0.707 | 0.795 | 0.743 | 0.640 | FNO + Pakse |
| 7 days | 0.780 | 0.852 | 0.803 | 0.689 | FNO + Pakse |

### 4.1 Main takeaway

This horizon-wise structure is one of the most important current conclusions:

- **1–3 days:** persistence dominates
- **4 days:** Pakse-assisted performance begins to edge ahead
- **5–7 days:** Pakse-assisted gains become clearer

This pattern is realistic for mainstream river-stage prediction, where very short horizons are often persistence-dominated and medium horizons are where upstream information begins to provide more practical forecasting value.

---

## 5. Advanced Diagnostics (7-Day Setting)

The advanced diagnostics separate four different questions:

1. Does a source help on the dates where it is available?
2. Which source is stronger on the same overlap dates?
3. How often is each source available?
4. What does the current availability-aware routed system actually do in deployment?

### 5.1 3S-only available-subset comparison

| Window | k (days) | N | From | To | RMSE (FNO) | RMSE (FNO+3S) | ΔRMSE | MAE (FNO) | MAE (FNO+3S) | ΔMAE |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 3S-available (with lag k, h=7) | 3 | 198 | 2025-01-01 | 2026-04-11 | 0.706 | 0.552 | -0.154 | 0.447 | 0.388 | -0.058 |

**Interpretation:**  
On dates where lag-aligned 3S data are available, 3S-assisted correction materially improves over the base FNO forecast.

---

### 5.2 Pakse-only available-subset comparison

| Window | k (days) | N | From | To | RMSE (FNO) | RMSE (FNO+Pakse) | ΔRMSE | MAE (FNO) | MAE (FNO+Pakse) | ΔMAE |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| Pakse-available (with lag k, h=7) | 3 | 319 | 2025-01-01 | 2026-04-11 | 0.948 | 0.729 | -0.218 | 0.560 | 0.473 | -0.088 |

**Interpretation:**  
On dates where lag-aligned Pakse data are available, Pakse-assisted correction also materially improves over the base FNO forecast.

---

### 5.3 Fair same-date comparison on common overlap dates

| Window | k_3S (days) | k_Pakse (days) | N_overlap | From | To | RMSE (FNO) | RMSE (FNO+3S) | RMSE (FNO+Pakse) | MAE (FNO) | MAE (FNO+3S) | MAE (FNO+Pakse) | Best RMSE variant | Best MAE variant |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 3S & Pakse overlap (same dates, h=7) | 3 | 3 | 198 | 2025-01-01 | 2026-04-11 | 0.706 | 0.552 | 0.579 | 0.447 | 0.388 | 0.407 | FNO + 3S | FNO + 3S |

**Interpretation:**  
On a fair same-date comparison, **3S** is stronger than **Pakse**. This is important because it shows that the best common-date signal source is not necessarily the source that delivers the best full-window deployed behavior.

---

### 5.4 Source availability summary

| Window | Horizon | Total backtest days | k_3S (days) | k_Pakse (days) | 3S available days | Pakse available days | Both available days | 3S only days | Pakse only days | Neither available days | 3S availability rate (%) | Pakse availability rate (%) | Overlap rate (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Availability summary | 7 | 466 | 3 | 3 | 198 | 319 | 198 | 0 | 121 | 147 | 42.5 | 68.5 | 42.5 |

**Interpretation:**  

- 3S availability is entirely contained within Pakse availability in the current snapshot
- Pakse covers many more days than 3S
- 3S is stronger on common overlap dates, but Pakse has broader operational coverage

This explains why **Pakse** can produce the strongest full-window deployed behavior even though **3S** is better on fair same-date comparison.

---

### 5.5 Routed operational performance

| Routing policy | Horizon | N_total | RMSE (FNO full) | RMSE (routed) | ΔRMSE | MAE (FNO full) | MAE (routed) | ΔMAE | Use Pakse days | Use 3S days | Fallback FNO days | Use Pakse rate (%) | Use 3S rate (%) | Fallback FNO rate (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pakse > 3S > FNO fallback | 7 | 466 | 0.852 | 0.689 | -0.160 | 0.524 | 0.464 | -0.060 | 319 | 0 | 147 | 68.5 | 0.0 | 31.5 |

**Interpretation:****  
Under the current routing policy, the deployed system behaves effectively as a **Pakse-first operational supplement** with fallback to base FNO when assisted inputs are unavailable. This is why current full-window deployed performance is mainly driven by **Pakse coverage**.

---

## 6. Main Interpretation

### 6.1 What is strong in the current results

The strongest current result is not that the base FNO model beats all baselines.  
The strongest current result is that:

- a **Pakse-assisted deployed configuration**
- built on top of an **FNO-based data-driven core**
- can outperform both:
  - the **base FNO model**
  - and a strong **persistence baseline**
- at the **7-day setting**
- while also showing the clearest advantage at **5–7 day horizons**

This is a meaningful result for station-level operational forecasting.

---

### 6.2 What should be interpreted carefully

These results should **not** be simplified into the claim that:

- “FNO is strongest at all horizons”
- “Pakse is stronger than 3S in every sense”
- “the current system fully replaces persistence”

That would be inaccurate.

A more accurate reading is:

- **Persistence** is strongest at **very short horizons**
- **FNO + Pakse** becomes more useful at **medium horizons**
- **3S** is stronger on **fair same-date comparison**
- **Pakse** is stronger in **full-window deployment** because of better source coverage

---

### 6.3 Operational interpretation

The current system is best described as:

> a **medium-horizon station-level operational supplement**

rather than:

> a **replacement for short-horizon persistence**

This is a realistic and operationally meaningful result structure for mainstream river-stage prediction.

---

## 7. Representative Backtest Figure

A representative Tab 2 backtest figure is shown below.

![Representative Backtest Figure](figures/Backtest_Figure.webp)

This figure should be read together with the tables above:

- the plot shows how the deployed system behaves as a forecast service
- the tables explain how that behavior should be interpreted across horizons, sources, and routing logic

---

## 8. Related Artifacts

- Main project overview: [`../README.md`](../README.md)
- Current raw evaluation artifact: [`../assets/reports/eval_compare.json`](../assets/reports/eval_compare.json)

---

## 9. Notes

- The current values reflect the deployed evaluation snapshot available at the time this document was prepared.
- Slight numerical drift may occur as refreshed daily data are incorporated into the runtime service.
- Full-window, overlap-only, and routed results should always be interpreted in their own evaluation contexts rather than treated as directly interchangeable rankings.