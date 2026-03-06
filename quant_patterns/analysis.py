"""
Core analysis engine.

- Support / Resistance level detection
- Pattern similarity scoring (DTW, correlation, Euclidean)
- Event-window comparison
- Aggregated behavior profiles
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


# ── Support & Resistance ────────────────────────────────────────────────────────

@dataclass
class Level:
    price: float
    kind: str  # "support" or "resistance"
    touches: int = 1
    first_date: Optional[date] = None
    last_date: Optional[date] = None
    strength: float = 0.0  # 0-1 normalized

    @property
    def label(self) -> str:
        return f"{'S' if self.kind == 'support' else 'R'} @ {self.price:.2f} (touches={self.touches}, str={self.strength:.2f})"


def find_support_resistance(
    df: pd.DataFrame,
    window: int = 5,
    num_levels: int = 5,
    tolerance_pct: float = 0.5,
) -> list[Level]:
    """
    Detect support and resistance levels using local extrema clustering.

    Args:
        df: OHLCV DataFrame
        window: lookback window for local min/max detection
        num_levels: max levels to return per type
        tolerance_pct: % tolerance for clustering nearby levels
    """
    close = df["Close"].values
    dates = df.index

    if len(close) < window * 2 + 1:
        logger.warning("Not enough data for S/R detection")
        return []

    # Find local minima (support) and maxima (resistance)
    local_min_idx = argrelextrema(close, np.less_equal, order=window)[0]
    local_max_idx = argrelextrema(close, np.greater_equal, order=window)[0]

    levels: list[Level] = []

    # Cluster nearby levels
    def cluster_levels(indices: np.ndarray, kind: str) -> list[Level]:
        if len(indices) == 0:
            return []

        prices = close[indices]
        level_dates = dates[indices]
        tolerance = np.mean(prices) * (tolerance_pct / 100)

        clusters: list[list[int]] = []
        used = set()

        sorted_idx = np.argsort(prices)
        for i in sorted_idx:
            if i in used:
                continue
            cluster = [i]
            used.add(i)
            for j in sorted_idx:
                if j in used:
                    continue
                if abs(prices[i] - prices[j]) <= tolerance:
                    cluster.append(j)
                    used.add(j)
            clusters.append(cluster)

        result = []
        for cluster in clusters:
            cluster_prices = prices[cluster]
            cluster_dates = level_dates[cluster]
            avg_price = float(np.mean(cluster_prices))
            result.append(Level(
                price=avg_price,
                kind=kind,
                touches=len(cluster),
                first_date=cluster_dates.min().date() if hasattr(cluster_dates.min(), 'date') else cluster_dates.min(),
                last_date=cluster_dates.max().date() if hasattr(cluster_dates.max(), 'date') else cluster_dates.max(),
            ))

        # Sort by touches descending and take top N
        result.sort(key=lambda l: l.touches, reverse=True)
        return result[:num_levels]

    supports = cluster_levels(local_min_idx, "support")
    resistances = cluster_levels(local_max_idx, "resistance")

    all_levels = supports + resistances

    # Normalize strength
    if all_levels:
        max_touches = max(l.touches for l in all_levels)
        for l in all_levels:
            l.strength = l.touches / max_touches if max_touches > 0 else 0

    # Sort by price
    all_levels.sort(key=lambda l: l.price)
    return all_levels


# ── Pattern Similarity ──────────────────────────────────────────────────────────

@dataclass
class SimilarityResult:
    """Result of comparing two price windows."""
    event_name: str
    event_date: date
    correlation: float       # Pearson correlation of normalized close prices
    euclidean_dist: float    # Euclidean distance of normalized series
    dtw_distance: float      # Dynamic Time Warping distance
    composite_score: float   # Combined similarity score (0-1, higher = more similar)
    direction_match: float   # % of days where direction matches
    volatility_ratio: float  # Ratio of volatilities
    window_data: Optional[pd.DataFrame] = None

    @property
    def score_label(self) -> str:
        if self.composite_score >= 0.8:
            return "Very Similar"
        elif self.composite_score >= 0.6:
            return "Similar"
        elif self.composite_score >= 0.4:
            return "Moderate"
        else:
            return "Weak"


def _simple_dtw(s1: np.ndarray, s2: np.ndarray) -> float:
    """Simple DTW implementation without external dependencies."""
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )

    return float(dtw_matrix[n, m])


def compare_windows(
    target: pd.DataFrame,
    historical: pd.DataFrame,
    event_name: str = "",
    event_date: Optional[date] = None,
) -> SimilarityResult:
    """
    Compare two normalized price windows.
    Both DataFrames should have 'Close_norm' column from normalize_window().
    """
    # Align to same length
    min_len = min(len(target), len(historical))
    t_close = target["Close_norm"].values[:min_len]
    h_close = historical["Close_norm"].values[:min_len]

    # Handle NaN
    mask = ~(np.isnan(t_close) | np.isnan(h_close))
    t_clean = t_close[mask]
    h_clean = h_close[mask]

    if len(t_clean) < 3:
        return SimilarityResult(
            event_name=event_name,
            event_date=event_date or date.today(),
            correlation=0, euclidean_dist=float("inf"),
            dtw_distance=float("inf"), composite_score=0,
            direction_match=0, volatility_ratio=0,
            window_data=historical,
        )

    # Pearson correlation
    corr, _ = pearsonr(t_clean, h_clean)
    corr = max(-1, min(1, corr))  # clamp

    # Euclidean distance (normalized by length)
    euc = euclidean(t_clean, h_clean) / np.sqrt(len(t_clean))

    # DTW
    dtw_dist = _simple_dtw(t_clean, h_clean)
    dtw_norm = dtw_dist / len(t_clean)

    # Direction match: % of days where both moved same direction
    t_diff = np.diff(t_clean)
    h_diff = np.diff(h_clean)
    if len(t_diff) > 0:
        dir_match = float(np.mean(np.sign(t_diff) == np.sign(h_diff)))
    else:
        dir_match = 0.0

    # Volatility ratio
    t_vol = np.std(t_clean) if np.std(t_clean) > 0 else 1e-10
    h_vol = np.std(h_clean) if np.std(h_clean) > 0 else 1e-10
    vol_ratio = min(t_vol, h_vol) / max(t_vol, h_vol)

    # Composite score: weighted combination
    corr_score = (corr + 1) / 2  # map [-1, 1] to [0, 1]
    euc_score = max(0, 1 - euc / 10)  # higher is better
    dtw_score = max(0, 1 - dtw_norm / 10)

    composite = (
        0.30 * corr_score
        + 0.20 * euc_score
        + 0.20 * dtw_score
        + 0.20 * dir_match
        + 0.10 * vol_ratio
    )

    return SimilarityResult(
        event_name=event_name,
        event_date=event_date or date.today(),
        correlation=corr,
        euclidean_dist=euc,
        dtw_distance=dtw_dist,
        composite_score=composite,
        direction_match=dir_match,
        volatility_ratio=vol_ratio,
        window_data=historical,
    )


# ── Aggregation ─────────────────────────────────────────────────────────────────

@dataclass
class PatternProfile:
    """Aggregated behavior profile across multiple similar events."""
    ticker: str
    category: str
    num_events: int
    avg_return_before: float   # avg % change in days before event
    avg_return_after: float    # avg % change in days after event
    avg_return_event_day: float
    median_return_after: float
    positive_after_pct: float  # % of events with positive return after
    avg_volatility: float
    avg_volume_change: float   # avg % volume change on event day vs prior
    best_match: Optional[SimilarityResult] = None
    all_matches: list[SimilarityResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "category": self.category,
            "num_events": self.num_events,
            "avg_return_before_pct": round(self.avg_return_before, 3),
            "avg_return_after_pct": round(self.avg_return_after, 3),
            "avg_return_event_day_pct": round(self.avg_return_event_day, 3),
            "median_return_after_pct": round(self.median_return_after, 3),
            "positive_after_pct": round(self.positive_after_pct, 1),
            "avg_volatility": round(self.avg_volatility, 4),
            "avg_volume_change_pct": round(self.avg_volume_change, 2),
        }


def build_pattern_profile(
    ticker: str,
    category: str,
    windows: list[pd.DataFrame],
    similarity_results: list[SimilarityResult],
) -> PatternProfile:
    """
    Build aggregated behavior profile from multiple event windows.
    Each window should have rel_day and Close columns.
    """
    returns_before = []
    returns_after = []
    returns_event_day = []
    volatilities = []
    volume_changes = []

    for w in windows:
        if w.empty or "rel_day" not in w.columns:
            continue

        pre = w[w["rel_day"] < 0]["Close"]
        post = w[w["rel_day"] > 0]["Close"]
        event = w[w["rel_day"] == 0]["Close"]

        if len(pre) >= 2:
            ret_before = (pre.iloc[-1] / pre.iloc[0] - 1) * 100
            returns_before.append(ret_before)

        if len(post) >= 2 and not event.empty:
            ret_after = (post.iloc[-1] / event.values[0] - 1) * 100
            returns_after.append(ret_after)

        if not event.empty and len(pre) >= 1:
            ret_ed = (event.values[0] / pre.iloc[-1] - 1) * 100
            returns_event_day.append(ret_ed)

        # Volatility of the window
        daily_returns = w["Close"].pct_change().dropna()
        if len(daily_returns) > 1:
            volatilities.append(float(daily_returns.std()))

        # Volume change on event day vs avg prior
        if "Volume" in w.columns and not event.empty:
            pre_vol = w[w["rel_day"] < 0]["Volume"]
            ev_vol = w[w["rel_day"] == 0]["Volume"]
            if not pre_vol.empty and pre_vol.mean() > 0 and not ev_vol.empty:
                vol_chg = (ev_vol.values[0] / pre_vol.mean() - 1) * 100
                volume_changes.append(vol_chg)

    best = max(similarity_results, key=lambda s: s.composite_score) if similarity_results else None

    return PatternProfile(
        ticker=ticker,
        category=category,
        num_events=len(windows),
        avg_return_before=float(np.mean(returns_before)) if returns_before else 0,
        avg_return_after=float(np.mean(returns_after)) if returns_after else 0,
        avg_return_event_day=float(np.mean(returns_event_day)) if returns_event_day else 0,
        median_return_after=float(np.median(returns_after)) if returns_after else 0,
        positive_after_pct=(sum(1 for r in returns_after if r > 0) / len(returns_after) * 100) if returns_after else 0,
        avg_volatility=float(np.mean(volatilities)) if volatilities else 0,
        avg_volume_change=float(np.mean(volume_changes)) if volume_changes else 0,
        best_match=best,
        all_matches=sorted(similarity_results, key=lambda s: s.composite_score, reverse=True),
    )


# ── Sliding Window Scan ───────────────────────────────────────────────────────


def sliding_window_scan(
    df: pd.DataFrame,
    window_size: int,
    step: int = 1,
    top_n: int = 10,
    min_gap: int = 0,
) -> list[SimilarityResult]:
    """
    Scan historical data with a sliding window to find periods most similar
    to the most recent `window_size` trading days.

    Args:
        df: Full OHLCV DataFrame (must have Close column, DatetimeIndex).
        window_size: Number of trading days in each comparison window.
        step: Slide step in trading days (1 = every day, 5 = weekly).
        top_n: Number of top matches to return.
        min_gap: Minimum trading days between target and historical windows
                 to avoid near-duplicate matches (default: 0 = no gap beyond
                 the target window itself).

    Returns:
        Top N SimilarityResult objects sorted by composite_score descending.
    """
    if len(df) < window_size * 2:
        logger.warning("Not enough data for sliding window scan")
        return []

    close = df["Close"].values
    dates = df.index

    # Target = last window_size days
    target_start = len(df) - window_size
    target_close = close[target_start:]
    target_ref = target_close[0]
    if target_ref == 0:
        return []
    target_norm = ((target_close / target_ref) - 1) * 100

    # Build target DataFrame for compare_windows
    target_df = df.iloc[target_start:].copy()
    target_df["rel_day"] = range(window_size)
    target_df["Close_norm"] = target_norm

    results: list[SimilarityResult] = []
    scan_end = target_start - min_gap

    for i in range(0, scan_end - window_size + 1, step):
        hist_close = close[i : i + window_size]
        hist_ref = hist_close[0]
        if hist_ref == 0:
            continue
        hist_norm = ((hist_close / hist_ref) - 1) * 100

        hist_df = df.iloc[i : i + window_size].copy()
        hist_df["rel_day"] = range(window_size)
        hist_df["Close_norm"] = hist_norm

        start_date = dates[i].date() if hasattr(dates[i], "date") else dates[i]
        end_date = dates[i + window_size - 1].date() if hasattr(dates[i + window_size - 1], "date") else dates[i + window_size - 1]
        label = f"{start_date} → {end_date}"

        result = compare_windows(target_df, hist_df, event_name=label, event_date=start_date)
        result.window_data = hist_df
        results.append(result)

    results.sort(key=lambda r: r.composite_score, reverse=True)

    # Deduplicate overlapping windows: keep highest score per cluster
    if results:
        filtered = [results[0]]
        for r in results[1:]:
            too_close = False
            for kept in filtered:
                if kept.event_date and r.event_date:
                    gap = abs((kept.event_date - r.event_date).days)
                    if gap < window_size:
                        too_close = True
                        break
            if not too_close:
                filtered.append(r)
            if len(filtered) >= top_n:
                break
        results = filtered

    return results[:top_n]


# ── Volume-Price Authenticity ──────────────────────────────────────────────────

@dataclass
class VolumePriceDay:
    """Per-day volume-price metrics."""
    rel_day: int
    date: date
    price_change_pct: float
    relative_volume: float  # RVOL = day_vol / 20-day rolling avg
    move_efficiency: float  # abs(price_chg%) / RVOL
    classification: str     # organic/synthetic/accumulation/distribution/neutral
    volume_confirms_price: bool


@dataclass
class VolumePriceProfile:
    """Aggregate volume-price authenticity assessment."""
    ticker: str
    window_start: date
    window_end: date
    num_days: int
    volume_confirmation_pct: float  # % of days where volume confirms price direction
    authenticity_score: float       # 0-1 weighted composite
    classification: str             # Organic, Likely Synthetic, etc.
    avg_relative_volume: float
    high_volume_days: int
    low_volume_days: int
    daily_metrics: list[VolumePriceDay] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "num_days": self.num_days,
            "volume_confirmation_pct": round(self.volume_confirmation_pct, 1),
            "authenticity_score": round(self.authenticity_score, 3),
            "classification": self.classification,
            "avg_relative_volume": round(self.avg_relative_volume, 2),
            "high_volume_days": self.high_volume_days,
            "low_volume_days": self.low_volume_days,
        }


def analyze_volume_price(
    df: pd.DataFrame,
    lookback_avg: int = 20,
    report_last_n: Optional[int] = None,
) -> Optional[VolumePriceProfile]:
    """
    Analyze volume-price relationship to classify moves as organic vs synthetic.

    Args:
        df: OHLCV DataFrame with Close and Volume columns.
        lookback_avg: Rolling window for average volume baseline.
        report_last_n: If set, compute rolling avg on full df but only report
                       last N days (useful for scan command).
    """
    if df.empty or "Volume" not in df.columns or "Close" not in df.columns:
        return None
    if len(df) < 3:
        return None

    # Compute rolling average volume on full df
    df = df.copy()
    df["_vol_avg"] = df["Volume"].rolling(lookback_avg, min_periods=1).mean()
    df["_price_chg"] = df["Close"].pct_change() * 100

    # Determine reporting slice
    if report_last_n and report_last_n < len(df):
        report_df = df.iloc[-report_last_n:]
    else:
        report_df = df.iloc[1:]  # skip first row (no price change)

    daily_metrics: list[VolumePriceDay] = []
    for idx, row in report_df.iterrows():
        vol = row["Volume"]
        vol_avg = row["_vol_avg"]
        price_chg = row["_price_chg"]

        if pd.isna(price_chg) or pd.isna(vol_avg) or vol_avg == 0:
            continue

        rvol = vol / vol_avg
        efficiency = abs(price_chg) / rvol if rvol > 0 else 0.0
        abs_chg = abs(price_chg)

        # Classification logic
        large_move = abs_chg > 0.5
        small_move = abs_chg <= 0.5

        if large_move and rvol > 1.2:
            classification = "organic"
        elif large_move and rvol < 0.5:
            classification = "synthetic"
        elif small_move and rvol > 1.5 and price_chg >= 0:
            classification = "accumulation"
        elif small_move and rvol > 1.5 and price_chg < 0:
            classification = "distribution"
        else:
            classification = "neutral"

        # Volume confirms price: high volume on moves, low volume on small days
        if large_move:
            confirms = rvol > 0.8
        else:
            confirms = rvol <= 1.5

        dt = idx.date() if hasattr(idx, "date") else idx
        rel_day_val = int(row["rel_day"]) if "rel_day" in df.columns else 0

        daily_metrics.append(VolumePriceDay(
            rel_day=rel_day_val,
            date=dt,
            price_change_pct=round(price_chg, 3),
            relative_volume=round(rvol, 2),
            move_efficiency=round(efficiency, 3),
            classification=classification,
            volume_confirms_price=confirms,
        ))

    if not daily_metrics:
        return None

    # Aggregate
    num_confirms = sum(1 for d in daily_metrics if d.volume_confirms_price)
    confirmation_pct = (num_confirms / len(daily_metrics)) * 100

    avg_rvol = np.mean([d.relative_volume for d in daily_metrics])
    high_vol = sum(1 for d in daily_metrics if d.relative_volume > 1.2)
    low_vol = sum(1 for d in daily_metrics if d.relative_volume < 0.8)
    organic_count = sum(1 for d in daily_metrics if d.classification == "organic")
    organic_ratio = organic_count / len(daily_metrics)
    avg_efficiency = np.mean([d.move_efficiency for d in daily_metrics])

    # Authenticity score: weighted composite
    conf_norm = min(1.0, confirmation_pct / 100)
    rvol_norm = min(1.0, avg_rvol / 2.0)  # cap at 2x average
    eff_norm = min(1.0, avg_efficiency / 5.0)  # normalize efficiency

    authenticity = (
        0.35 * conf_norm
        + 0.25 * rvol_norm
        + 0.20 * eff_norm
        + 0.20 * organic_ratio
    )

    # Overall classification
    synthetic_count = sum(1 for d in daily_metrics if d.classification == "synthetic")
    accum_count = sum(1 for d in daily_metrics if d.classification == "accumulation")
    distrib_count = sum(1 for d in daily_metrics if d.classification == "distribution")

    if organic_ratio > 0.4 and authenticity > 0.5:
        overall = "Organic"
    elif synthetic_count > organic_count and authenticity < 0.4:
        overall = "Likely Synthetic"
    elif accum_count > organic_count and accum_count > synthetic_count:
        overall = "Accumulation Phase"
    elif distrib_count > organic_count and distrib_count > synthetic_count:
        overall = "Distribution Phase"
    else:
        overall = "Mixed"

    start_dt = daily_metrics[0].date
    end_dt = daily_metrics[-1].date

    return VolumePriceProfile(
        ticker=df.attrs.get("ticker", ""),
        window_start=start_dt,
        window_end=end_dt,
        num_days=len(daily_metrics),
        volume_confirmation_pct=round(confirmation_pct, 1),
        authenticity_score=round(authenticity, 3),
        classification=overall,
        avg_relative_volume=round(float(avg_rvol), 2),
        high_volume_days=high_vol,
        low_volume_days=low_vol,
        daily_metrics=daily_metrics,
    )


# ── Quant Agent Export ──────────────────────────────────────────────────────────

def export_for_agent(
    profile: PatternProfile,
    support_resistance: list[Level],
    target_window: pd.DataFrame,
    volume_price: Optional[VolumePriceProfile] = None,
    regime: Optional[dict] = None,
) -> dict:
    """
    Export analysis results in a structured format for downstream quant agent consumption.
    """
    sr_data = [
        {
            "price": l.price,
            "type": l.kind,
            "touches": l.touches,
            "strength": l.strength,
        }
        for l in support_resistance
    ]

    matches_data = [
        {
            "event": m.event_name,
            "date": m.event_date.isoformat() if m.event_date else None,
            "composite_score": round(m.composite_score, 4),
            "correlation": round(m.correlation, 4),
            "direction_match": round(m.direction_match, 4),
            "dtw_distance": round(m.dtw_distance, 4),
            "label": m.score_label,
        }
        for m in profile.all_matches[:10]
    ]

    return {
        "ticker": profile.ticker,
        "event_category": profile.category,
        "analysis_summary": profile.to_dict(),
        "support_resistance": sr_data,
        "top_matches": matches_data,
        "target_window": {
            "start": target_window.index[0].isoformat() if len(target_window) > 0 else None,
            "end": target_window.index[-1].isoformat() if len(target_window) > 0 else None,
            "prices": target_window["Close"].tolist() if "Close" in target_window.columns else [],
        },
        "signal": {
            "direction": "bullish" if profile.avg_return_after > 0 else "bearish",
            "confidence": min(1.0, profile.positive_after_pct / 100 if profile.avg_return_after > 0
                            else (100 - profile.positive_after_pct) / 100),
            "historical_edge_pct": round(profile.avg_return_after, 3),
        },
        "volume_price_authenticity": volume_price.to_dict() if volume_price else None,
        "regime": regime,
    }
