# Market Regime Detection

This document describes the HMM-based regime detection system in `qpat`.

---

## Table of Contents

1. [Overview](#overview)
2. [Usage](#usage)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [HMM Training](#hmm-training)
6. [Regime Labeling](#regime-labeling)
7. [Regime-Filtered Analysis](#regime-filtered-analysis)
8. [JSON Export Schema](#json-export-schema)
9. [Constants Reference](#constants-reference)

---

## Overview

**Module:** `regime.py`

Pattern similarity scores alone can be misleading — a CPI bounce from a 2021 bull market may not apply during a 2022-style high-volatility bear. The regime detector classifies the current market environment into one of four states:

| Regime | Description |
|--------|-------------|
| **Bull-Trend** | Positive returns, low-to-moderate volatility |
| **Bear-Trend** | Negative returns, moderate volatility |
| **Low-Vol-Range** | Flat returns, compressed volatility |
| **High-Vol-Stress** | Negative returns, elevated volatility |

This allows filtering historical events to only those that occurred during a similar market environment, improving signal relevance.

---

## Usage

### Standalone regime detection

```bash
# Basic — detect current regime for SPY
qpat regime SPY

# Longer history for more stable estimates
qpat regime QQQ --lookback 1500

# Export to JSON
qpat regime NVDA -o regime.json

# Use fewer states (2-6)
qpat regime SPY --states 3
```

### Regime-filtered analysis

```bash
# Only compare against FOMC events from the same regime as today
qpat analyze SPY -e fomc --regime-filter

# Custom regime lookback
qpat analyze SPY -e cpi --regime-filter --regime-lookback 1000
```

### Output

The `regime` command displays three panels:

1. **Regime Summary** — Current regime label, confidence (posterior probability), and probability bar chart across all four regimes
2. **State Characteristics Table** — Annualized return, annualized volatility, VIX ratio, and historical frequency for each regime
3. **Price + Regime Chart** — ASCII price chart with a letter strip showing regime transitions (B=Bull, D=Bear, R=Range, S=Stress)

---

## Data Pipeline

**Function:** `fetch_regime_data()` in `regime.py`

The detector uses the primary ticker plus five auxiliary indices to capture broad market context:

| Symbol | Purpose |
|--------|---------|
| Ticker (via provider) | Primary price data |
| `^VIX` | Implied volatility |
| `^VIX3M` | 3-month implied volatility (term structure) |
| `SPY` | Broad market trend |
| `HYG` | High-yield corporate bonds |
| `LQD` | Investment-grade corporate bonds |

The primary ticker is fetched through the configured `DataProvider`. Auxiliary indices are fetched via direct `yfinance.download()` calls, each wrapped in try/except — missing auxiliaries degrade gracefully (features fall back to defaults or are dropped).

---

## Feature Engineering

**Function:** `build_regime_features()` in `regime.py`

### Core Features (always available)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `log_ret` | `ln(Close_t / Close_{t-1})` | Daily log return |
| `rolling_vol_20` | `std(log_ret, window=20)` | 20-day rolling volatility |

### Optional Features (included when data available)

| Feature | Formula | Fallback | Purpose |
|---------|---------|----------|---------|
| `vix_ratio` | `VIX / VIX3M` | 1.0 | Term structure — >1.0 = backwardation (stress), <1.0 = contango (calm) |
| `spy_above_50d` | `1 if SPY > SMA(50) else 0` | Dropped | Short-term trend |
| `spy_above_200d` | `1 if SPY > SMA(200) else 0` | Dropped | Long-term trend |
| `credit_spread` | `HYG / LQD` | Dropped | Credit risk appetite — declining = stress |

NaN rows are dropped after feature construction. The model requires at least 60 observations and warns below 200.

---

## HMM Training

**Function:** `detect_regime()` in `regime.py`

### Model

- **Type:** `GaussianHMM` from `hmmlearn`
- **States:** 4 (configurable via `--states`)
- **Covariance:** Full (captures feature correlations)
- **Iterations:** 500 per fit

### Multiple Random Restarts

HMMs are sensitive to initialization. The training runs `n_fits=5` independent fits with different random seeds and selects the model with the highest log-likelihood:

```
for i in range(n_fits):
    model = GaussianHMM(n_components=4, covariance_type='full',
                        n_iter=500, random_state=42 + i)
    model.fit(X)
    if model.score(X) > best_score:
        best_model = model
```

### Outputs

| Output | Description |
|--------|-------------|
| `model` | Trained GaussianHMM object |
| `states` | Per-observation state assignments |
| `log_likelihood` | Best model's log-likelihood score |
| `converged` | Whether the best model's EM algorithm converged |

---

## Regime Labeling

**Function:** `label_regimes()` in `regime.py`

HMM state IDs are arbitrary integers. The labeler maps them to human-readable names using the model's learned means:

### Algorithm

1. Extract `mean_return` and `mean_volatility` per state from `model.means_`
2. Sort states by mean return descending
3. Assign labels:

| Rule | Assigned Label |
|------|---------------|
| Highest mean return | **Bull-Trend** |
| Lowest mean return + high volatility (above median) | **High-Vol-Stress** |
| Lowest mean return + low volatility (below median) | **Bear-Trend** |
| Remaining two states: lower volatility one | **Low-Vol-Range** |
| Remaining state | Whichever label is unassigned |

### Posterior Probabilities

After labeling, the model computes posterior probabilities for the most recent observation via `model.predict_proba(X)[-1]`, giving a confidence distribution across all four regimes.

---

## Regime-Filtered Analysis

**Flag:** `--regime-filter` on the `analyze` command

### How It Works

1. Before the event comparison loop, run `run_regime_detection()` on the ticker
2. Identify the current regime label
3. For each historical event, look up its regime via `get_regime_at_date()` (snaps to nearest trading day within 5 days)
4. Filter the event list to only those matching the current regime
5. If zero events remain after filtering, fall back to the unfiltered list with a warning

### Regime-Conditional Win Rates

After the pattern profile display, a table shows win rate, average return, and sample size broken out by regime:

```
Regime-Conditional Win Rates — SPY × FOMC
┌──────────────────┬──────────┬────────────┬────────┐
│ Regime           │ Win Rate │ Avg Return │ Sample │
├──────────────────┼──────────┼────────────┼────────┤
│ Bull-Trend       │   72.0%  │   +0.450%  │     7  │
│ Bear-Trend       │   33.3%  │   -0.820%  │     3  │
│ High-Vol-Stress  │   50.0%  │   +0.120%  │     2  │
└──────────────────┴──────────┴────────────┴────────┘
```

This is computed by iterating all events in the category (not just the filtered set), grouping by their historical regime, and computing post-event returns from the similarity result window data.

---

## JSON Export Schema

### `qpat regime SPY -o regime.json`

```json
{
  "ticker": "SPY",
  "current_regime": "Bull-Trend",
  "probabilities": {
    "Bull-Trend": 0.9523,
    "Low-Vol-Range": 0.0301,
    "Bear-Trend": 0.0112,
    "High-Vol-Stress": 0.0064
  },
  "states": [
    {
      "label": "Bull-Trend",
      "mean_return": 0.000842,
      "mean_volatility": 0.007231,
      "mean_vix_ratio": 0.8734,
      "frequency_pct": 62.5
    }
  ],
  "n_observations": 509,
  "log_likelihood": 8542.31,
  "converged": true
}
```

### `export_for_agent()` integration

When `--regime-filter` is used with `--export-json`, the regime data is included in the agent export under the `"regime"` key with the same schema above.

---

## Constants Reference

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| Default lookback | 750 days | `cli.py` | Calendar days of history for regime detection |
| Default states | 4 | `cli.py` / `regime.py` | Number of HMM components |
| Min observations | 60 | `regime.py` | Hard minimum — raises error below this |
| Warn observations | 200 | `regime.py` | Logs warning if below this |
| HMM iterations | 500 | `regime.py` | EM algorithm max iterations per fit |
| Random restarts | 5 | `regime.py` | Number of independent fits |
| Random seed base | 42 | `regime.py` | Starting seed (incremented per restart) |
| Date snap tolerance | 5 days | `regime.py` | Max distance when looking up regime for a date |
| Rolling vol window | 20 days | `regime.py` | Window for `rolling_vol_20` feature |
| SMA(50) min periods | 10 | `regime.py` | Minimum observations for 50-day SMA |
| SMA(200) min periods | 50 | `regime.py` | Minimum observations for 200-day SMA |
| Vol ratio clamp | [0.5, 2.0] | `regime.py` | Bounds for volatility scaling in forecasts |
