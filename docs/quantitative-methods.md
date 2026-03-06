# Quantitative Methods

This document describes every quantitative algorithm, metric, and technique used in `qpat`.

---

## Table of Contents

1. [Data Pipeline](#data-pipeline)
2. [Similarity Metrics](#similarity-metrics)
3. [Composite Scoring](#composite-scoring)
4. [Support & Resistance Detection](#support--resistance-detection)
5. [Volume-Price Authenticity Analysis](#volume-price-authenticity-analysis)
6. [Pattern Profile Aggregation](#pattern-profile-aggregation)
7. [Sliding Window Scan](#sliding-window-scan)
8. [Forecast Methodology](#forecast-methodology)
9. [Constants Reference](#constants-reference)

---

## Data Pipeline

**Modules:** `data.py` (fetching/normalization) → `analysis.py` (scoring) → `cli.py` (forecasting) → `display.py` (rendering)

### Event Window Extraction

`fetch_event_window()` pulls OHLCV data around an event date, producing a DataFrame with a `rel_day` column where `0` is the event day, negative values are days before, and positive values are days after.

### Normalization

`normalize_window()` converts raw prices to percentage change from the event-day close:

```
Close_norm = ((Close / Close_at_rel_day_0) - 1) * 100
```

All OHLC columns are normalized the same way. This allows direct comparison of price behavior across different time periods and price levels.

---

## Similarity Metrics

**Function:** `compare_windows()` in `analysis.py`

Five metrics quantify how similar a target window is to a historical window. Both inputs are normalized `Close_norm` series aligned by `rel_day`.

### 1. Pearson Correlation

Measures linear co-movement between two series.

- **Implementation:** `scipy.stats.pearsonr(target, historical)`
- **Raw range:** [-1, 1]
- **Score conversion:** `score = (correlation + 1) / 2` → maps to [0, 1]
- **Interpretation:** 1.0 = perfectly correlated shapes, 0.5 = uncorrelated, 0.0 = perfectly anti-correlated

### 2. Euclidean Distance

Measures point-by-point magnitude difference between series.

- **Formula:** `distance = scipy.spatial.distance.euclidean(target, historical) / sqrt(length)`
- **Score conversion:** `score = max(0, 1 - distance / 10)`
- **Interpretation:** Division by `sqrt(length)` normalizes for window size. Score of 1.0 = identical series, decaying toward 0 as distance grows.

### 3. Dynamic Time Warping (DTW)

Measures shape similarity allowing for temporal misalignment. Captures patterns that are similar but shifted in time.

- **Algorithm:** Custom implementation (`_simple_dtw()`) using a full cost matrix. For each cell `(i, j)`:
  ```
  cost = |s1[i] - s2[j]|
  dtw[i,j] = cost + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
  ```
- **Normalization:** `dtw_norm = dtw_distance / length`
- **Score conversion:** `score = max(0, 1 - dtw_norm / 10)`
- **Interpretation:** Allows elastic alignment — two series with similar shape but different timing still score well.

### 4. Direction Match

Percentage of days where both series moved in the same direction (up/down).

- **Formula:**
  ```
  target_diff = diff(target)
  hist_diff = diff(historical)
  score = mean(sign(target_diff) == sign(hist_diff))
  ```
- **Range:** [0, 1]
- **Interpretation:** 1.0 = every day moved the same direction. Ignores magnitude, focuses purely on directional agreement.

### 5. Volatility Ratio

Symmetric measure of how similar the two series' volatilities are.

- **Formula:** `ratio = min(std_target, std_hist) / max(std_target, std_hist)`
- **Range:** [0, 1]
- **Interpretation:** 1.0 = identical volatility. Penalizes matches where one period was much more volatile than the other.

---

## Composite Scoring

The five metrics are combined into a single score using fixed weights:

| Metric | Weight | Rationale |
|--------|--------|-----------|
| Pearson Correlation | 0.30 | Primary shape measure |
| Euclidean Distance | 0.20 | Magnitude alignment |
| DTW Distance | 0.20 | Time-flexible shape matching |
| Direction Match | 0.20 | Day-level directional agreement |
| Volatility Ratio | 0.10 | Regime similarity check |

```
composite = 0.30 * corr + 0.20 * euclidean + 0.20 * dtw + 0.20 * direction + 0.10 * volatility
```

### Score Labels

| Range | Label |
|-------|-------|
| 0.80 - 1.00 | Very Similar |
| 0.60 - 0.80 | Similar |
| 0.40 - 0.60 | Moderate |
| 0.00 - 0.40 | Weak |

---

## Support & Resistance Detection

**Function:** `find_support_resistance()` in `analysis.py`

### Step 1: Local Extrema

Uses `scipy.signal.argrelextrema` to find:
- **Support candidates:** Local minima (`np.less_equal`, `order=window`)
- **Resistance candidates:** Local maxima (`np.greater_equal`, `order=window`)

The `order` parameter (default: 5) controls how many neighbors on each side must be ≥ or ≤ the candidate.

### Step 2: Price Clustering

Nearby extrema are grouped into consolidated levels:

1. Compute tolerance: `tolerance = mean(prices) * 0.5%`
2. Sort candidate prices ascending
3. Greedily cluster: each unassigned price starts a new cluster; all unassigned prices within tolerance join it
4. Each cluster produces one level at its average price

### Step 3: Scoring

- **Touches:** Number of extrema in the cluster (more touches = stronger level)
- **Strength:** Normalized to [0, 1] by dividing by the maximum touch count across all levels
- **Date range:** First and last dates of touches in the cluster

### Output

Top N levels per type (support/resistance), sorted by price ascending.

---

## Volume-Price Authenticity Analysis

**Function:** `analyze_volume_price()` in `analysis.py`

Evaluates whether price moves are supported by genuine volume, helping distinguish organic moves from synthetic or manipulated ones.

### Per-Day Metrics

| Metric | Formula |
|--------|---------|
| Relative Volume (RVOL) | `day_volume / 20-day rolling average` |
| Move Efficiency | `abs(price_change_pct) / RVOL` |

### Per-Day Classification

| Condition | Classification |
|-----------|---------------|
| Large move (>0.5%) AND RVOL > 1.2 | Organic |
| Large move (>0.5%) AND RVOL < 0.5 | Synthetic |
| Small move (≤0.5%) AND RVOL > 1.5, price up | Accumulation |
| Small move (≤0.5%) AND RVOL > 1.5, price down | Distribution |
| Everything else | Neutral |

### Volume Confirmation

- **Large moves:** Confirmed if RVOL > 0.8
- **Small moves:** Confirmed if RVOL ≤ 1.5

### Aggregate Authenticity Score

```
authenticity = 0.35 * confirmation_pct/100
             + 0.25 * min(1.0, avg_rvol/2.0)
             + 0.20 * min(1.0, avg_efficiency/5.0)
             + 0.20 * organic_ratio
```

### Overall Classification

| Condition | Label |
|-----------|-------|
| organic_ratio > 0.4 AND authenticity > 0.5 | Organic |
| synthetic > organic AND authenticity < 0.4 | Likely Synthetic |
| accumulation dominates | Accumulation Phase |
| distribution dominates | Distribution Phase |
| Otherwise | Mixed |

---

## Pattern Profile Aggregation

**Function:** `build_pattern_profile()` in `analysis.py`

Aggregates statistics across all event windows for a given ticker and event category.

### Computed Statistics

| Stat | Formula |
|------|---------|
| Avg Return Before | Mean of `(pre_close_last / pre_close_first - 1) * 100` across events |
| Avg Return Event Day | Mean of `(event_close / pre_close_last - 1) * 100` |
| Avg Return After | Mean of `(post_close_last / event_close - 1) * 100` |
| Median Return After | Median of post-event returns |
| Positive After % | `count(post_return > 0) / total_events * 100` |
| Avg Volatility | Mean of `std(daily_returns)` per window |
| Avg Volume Change | Mean of `(event_day_volume / pre_avg_volume - 1) * 100` |

### Trading Signal

```
Direction: BULLISH if avg_return_after > 0, else BEARISH
Confidence: positive_after_pct / 100  (capped at 1.0)
Edge: avg_return_after (in %)
```

---

## Sliding Window Scan

**Function:** `sliding_window_scan()` in `analysis.py`

Scans all historical data for periods that most closely resemble recent price action, without requiring predefined events.

### Algorithm

1. **Target:** Most recent `window_size` trading days, normalized to % change from first day
2. **Scan:** Slide a window of the same size across all historical data, stepping by `step` days
3. **Score:** Run the full 5-metric `compare_windows()` on each historical window vs. target
4. **Rank:** Sort all windows by composite score descending
5. **Deduplicate:** Remove overlapping matches — keep only windows whose start dates are at least `window_size` days apart
6. **Return:** Top N results

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | varies | Trading days per window |
| `step` | 1 | Slide increment (1=daily, 5=weekly) |
| `min_gap` | 0 | Minimum gap between target and history |
| `top_n` | 10 | Max results returned |

---

## Forecast Methodology

**Functions:** `_build_event_forecast()` and `_display_forecast()` in `cli.py`

Produces day-by-day price projections based on what happened after similar historical patterns.

### Step 1: Quality-Gated Filtering

Only matches above a minimum composite score are included. This prevents weak/noisy matches from diluting the forecast.

```
quality_matches = [m for m in matches if m.composite_score >= 0.4]
if len(quality_matches) < 2:
    quality_matches = top 3 matches regardless of score
```

### Step 2: Exponential Score Weighting

Weights are squared to amplify the influence of high-quality matches:

```
weight = composite_score ^ 2
```

| Score | Linear Weight | Squared Weight | Ratio |
|-------|--------------|----------------|-------|
| 0.9 | 0.90 | 0.81 | — |
| 0.7 | 0.70 | 0.49 | 1.65x less |
| 0.5 | 0.50 | 0.25 | 3.24x less |

### Step 3: Volatility Scaling

Each historical match's forward returns are scaled by the ratio of current to historical volatility:

```
current_vol = std(recent daily returns)
hist_vol = std(match window daily returns)
vol_ratio = clamp(current_vol / hist_vol, 0.5, 2.0)
scaled_return = raw_return * vol_ratio
```

If the current market is twice as volatile as the historical period, projected returns are doubled (up to the 2.0 cap). If calmer, they're reduced (down to 0.5 floor).

### Step 4: Weighted Return Aggregation

For each forecast day, compute the weighted average daily return across all contributing matches:

```
avg_return_day_d = sum(return_i * weight_i) / sum(weight_i)
projected_price = previous_price * (1 + avg_return / 100)
```

### Step 5: Confidence Bands

Track each match's individual cumulative projection to compute spread:

- **25th-75th Percentile Range:** Interquartile range of individual match projections at each day
- **Min-Max Range:** Full range across all matches
- **Direction Consensus:** Percentage of matches that agree with the weighted average's direction (up vs. down)

### Step 6: Confidence Decay

Forecast reliability decreases with horizon:

```
confidence = max(0.30, 1.0 - 0.05 * day_number)
```

| Day | Confidence |
|-----|-----------|
| +1 | 95% |
| +5 | 75% |
| +10 | 50% |
| +14+ | 30% (floor) |

### Step 7: Backtesting (Optional)

When a recent event exists, the forecast anchors to the event-day close and compares projections against actual prices for days that have already passed. The "Miss" column shows `(predicted - actual) / actual * 100`.

---

## Constants Reference

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `FORECAST_MIN_SCORE` | 0.4 | `cli.py` | Quality gate threshold |
| `FORECAST_MIN_MATCHES` | 2 | `cli.py` | Minimum matches after quality gate |
| `FORECAST_FALLBACK_N` | 3 | `cli.py` | Fallback top-N if too few pass gate |
| `FORECAST_CONF_DECAY` | 0.05 | `cli.py` | Confidence loss per forecast day |
| `FORECAST_CONF_FLOOR` | 0.30 | `cli.py` | Minimum confidence |
| Correlation weight | 0.30 | `analysis.py` | Composite score weight |
| Euclidean weight | 0.20 | `analysis.py` | Composite score weight |
| DTW weight | 0.20 | `analysis.py` | Composite score weight |
| Direction weight | 0.20 | `analysis.py` | Composite score weight |
| Volatility weight | 0.10 | `analysis.py` | Composite score weight |
| S/R tolerance | 0.5% | `analysis.py` | Clustering tolerance for levels |
| S/R extrema order | 5 | `analysis.py` | `argrelextrema` window parameter |
| RVOL lookback | 20 days | `analysis.py` | Rolling average for relative volume |
| Vol ratio bounds | [0.5, 2.0] | `cli.py` | Forecast volatility scaling clamp |
| Large move threshold | 0.5% | `analysis.py` | Volume-price classification boundary |
| Euclidean/DTW cap | 10 | `analysis.py` | Score normalization denominator |
