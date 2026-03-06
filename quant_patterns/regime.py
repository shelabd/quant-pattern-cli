"""
HMM-based market regime detection.

Detects Bull-Trend, Bear-Trend, Low-Vol-Range, and High-Vol-Stress regimes
using a Gaussian Hidden Markov Model trained on price/volatility/macro features.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class RegimeState:
    """Per-state summary from HMM."""
    state_id: int
    label: str
    mean_return: float
    mean_volatility: float
    mean_vix_ratio: float
    frequency_pct: float


@dataclass
class RegimeResult:
    """Full regime detection result."""
    ticker: str
    current_regime: str
    probabilities: dict[str, float]  # label -> probability
    states: list[RegimeState]
    regime_history: pd.DataFrame  # Close + regime_label columns
    n_observations: int
    log_likelihood: float
    converged: bool

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "current_regime": self.current_regime,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "states": [
                {
                    "label": s.label,
                    "mean_return": round(s.mean_return, 6),
                    "mean_volatility": round(s.mean_volatility, 6),
                    "mean_vix_ratio": round(s.mean_vix_ratio, 4),
                    "frequency_pct": round(s.frequency_pct, 1),
                }
                for s in self.states
            ],
            "n_observations": self.n_observations,
            "log_likelihood": round(self.log_likelihood, 2),
            "converged": self.converged,
        }


# ── Data Fetching ───────────────────────────────────────────────────────────

def fetch_regime_data(provider, ticker: str, lookback_days: int = 750) -> dict[str, pd.DataFrame]:
    """Fetch ticker OHLCV and auxiliary indices for regime detection."""
    import yfinance as yf

    end = date.today()
    start = end - timedelta(days=lookback_days)

    # Primary ticker via provider
    ticker_df = provider.get_daily_ohlcv(ticker, start, end)

    # Auxiliary indices via direct yfinance (same pattern as fetch_ticker_info)
    aux_tickers = {"vix": "^VIX", "vix3m": "^VIX3M", "spy": "SPY", "hyg": "HYG", "lqd": "LQD"}
    aux_data: dict[str, pd.DataFrame] = {"ticker": ticker_df}

    for key, sym in aux_tickers.items():
        try:
            df = yf.download(sym, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                             progress=False, auto_adjust=True)
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                aux_data[key] = df
            else:
                aux_data[key] = pd.DataFrame()
        except Exception as e:
            logger.debug(f"Could not fetch {sym}: {e}")
            aux_data[key] = pd.DataFrame()

    return aux_data


# ── Feature Engineering ─────────────────────────────────────────────────────

def build_regime_features(
    ticker_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    vix3m_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    hyg_df: pd.DataFrame,
    lqd_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature matrix for HMM training."""
    df = ticker_df[["Close"]].copy()

    # Core features (always available)
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["rolling_vol_20"] = df["log_ret"].rolling(20, min_periods=5).std()

    # VIX ratio (VIX / VIX3M term structure)
    if not vix_df.empty and not vix3m_df.empty and "Close" in vix_df.columns and "Close" in vix3m_df.columns:
        vix_close = vix_df[["Close"]].rename(columns={"Close": "vix"})
        vix3m_close = vix3m_df[["Close"]].rename(columns={"Close": "vix3m"})
        df = df.join(vix_close, how="left").join(vix3m_close, how="left")
        df["vix_ratio"] = df["vix"] / df["vix3m"]
        df["vix_ratio"] = df["vix_ratio"].fillna(1.0)
        df.drop(columns=["vix", "vix3m"], inplace=True)
    else:
        df["vix_ratio"] = 1.0

    # SPY trend indicators
    if not spy_df.empty and "Close" in spy_df.columns:
        spy_close = spy_df[["Close"]].rename(columns={"Close": "spy_close"})
        df = df.join(spy_close, how="left")
        df["spy_close"] = df["spy_close"].ffill()
        spy_50d = df["spy_close"].rolling(50, min_periods=10).mean()
        spy_200d = df["spy_close"].rolling(200, min_periods=50).mean()
        df["spy_above_50d"] = (df["spy_close"] > spy_50d).astype(float)
        df["spy_above_200d"] = (df["spy_close"] > spy_200d).astype(float)
        df.drop(columns=["spy_close"], inplace=True)

    # Credit spread (HYG/LQD ratio)
    if not hyg_df.empty and not lqd_df.empty and "Close" in hyg_df.columns and "Close" in lqd_df.columns:
        hyg_close = hyg_df[["Close"]].rename(columns={"Close": "hyg"})
        lqd_close = lqd_df[["Close"]].rename(columns={"Close": "lqd"})
        df = df.join(hyg_close, how="left").join(lqd_close, how="left")
        df["hyg"] = df["hyg"].ffill()
        df["lqd"] = df["lqd"].ffill()
        mask = df["lqd"] > 0
        df.loc[mask, "credit_spread"] = df.loc[mask, "hyg"] / df.loc[mask, "lqd"]
        df.drop(columns=["hyg", "lqd"], inplace=True)

    # Drop non-feature columns and NaN rows
    feature_cols = [c for c in df.columns if c != "Close"]
    features = df[feature_cols].copy()
    features.dropna(inplace=True)

    return features


# ── HMM Detection ───────────────────────────────────────────────────────────

def detect_regime(
    features: pd.DataFrame,
    n_states: int = 4,
    n_iter: int = 500,
    n_fits: int = 5,
    random_seed: int = 42,
) -> tuple:
    """Train GaussianHMM with multiple random restarts, return best fit.

    Returns:
        (model, states_array, log_likelihood, converged)
    """
    from hmmlearn.hmm import GaussianHMM

    X = features.values
    best_model = None
    best_ll = -np.inf
    best_converged = False

    for i in range(n_fits):
        seed = random_seed + i
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=n_iter,
                random_state=seed,
            )
            try:
                model.fit(X)
                ll = model.score(X)
                if ll > best_ll:
                    best_ll = ll
                    best_model = model
                    best_converged = model.monitor_.converged
            except Exception as e:
                logger.debug(f"HMM fit {i} failed: {e}")
                continue

    if best_model is None:
        raise RuntimeError("All HMM fits failed")

    states = best_model.predict(X)
    return best_model, states, best_ll, best_converged


# ── Regime Labeling ─────────────────────────────────────────────────────────

REGIME_LABELS = ["Bull-Trend", "Bear-Trend", "Low-Vol-Range", "High-Vol-Stress"]


def label_regimes(model, features: pd.DataFrame) -> dict[int, str]:
    """Map HMM state IDs to human-readable regime labels.

    Strategy:
    - Extract mean_return and mean_volatility per state from model.means_
    - Sort by mean_return descending
    - Top return -> Bull-Trend
    - Bottom return: if high vol -> High-Vol-Stress, else Bear-Trend
    - Remaining two: lower vol -> Low-Vol-Range, other gets remaining label
    """
    n_states = model.n_components
    col_names = list(features.columns)

    ret_idx = col_names.index("log_ret") if "log_ret" in col_names else 0
    vol_idx = col_names.index("rolling_vol_20") if "rolling_vol_20" in col_names else 1

    state_stats = []
    for s in range(n_states):
        mean_ret = model.means_[s][ret_idx]
        mean_vol = model.means_[s][vol_idx] if vol_idx < len(model.means_[s]) else 0
        state_stats.append((s, mean_ret, mean_vol))

    # Sort by mean return descending
    state_stats.sort(key=lambda x: x[1], reverse=True)

    labels: dict[int, str] = {}

    if n_states >= 4:
        # Top return -> Bull-Trend
        labels[state_stats[0][0]] = "Bull-Trend"

        # Bottom return: high vol -> High-Vol-Stress, else Bear-Trend
        bottom = state_stats[-1]
        # Compute median vol across states for threshold
        vols = [s[2] for s in state_stats]
        med_vol = np.median(vols)

        if bottom[2] > med_vol:
            labels[bottom[0]] = "High-Vol-Stress"
        else:
            labels[bottom[0]] = "Bear-Trend"

        # Remaining two middle states
        remaining = [s for s in state_stats[1:-1]]
        remaining_labels = [lb for lb in REGIME_LABELS if lb not in labels.values()]

        if len(remaining) >= 2:
            # Lower vol -> Low-Vol-Range
            remaining.sort(key=lambda x: x[2])
            labels[remaining[0][0]] = "Low-Vol-Range"
            # Other gets whatever is left
            leftover = [lb for lb in remaining_labels if lb != "Low-Vol-Range"]
            labels[remaining[1][0]] = leftover[0] if leftover else "Bear-Trend"
        elif len(remaining) == 1:
            labels[remaining[0][0]] = remaining_labels[0] if remaining_labels else "Low-Vol-Range"
    elif n_states == 3:
        labels[state_stats[0][0]] = "Bull-Trend"
        labels[state_stats[-1][0]] = "Bear-Trend"
        labels[state_stats[1][0]] = "Low-Vol-Range"
    elif n_states == 2:
        labels[state_stats[0][0]] = "Bull-Trend"
        labels[state_stats[1][0]] = "Bear-Trend"
    else:
        labels[state_stats[0][0]] = "Bull-Trend"

    return labels


# ── Orchestrator ────────────────────────────────────────────────────────────

def run_regime_detection(
    provider,
    ticker: str,
    lookback_days: int = 750,
    n_states: int = 4,
) -> RegimeResult:
    """End-to-end regime detection pipeline."""
    # Fetch data
    data = fetch_regime_data(provider, ticker, lookback_days)
    ticker_df = data["ticker"]

    # Build features
    features = build_regime_features(
        ticker_df,
        data.get("vix", pd.DataFrame()),
        data.get("vix3m", pd.DataFrame()),
        data.get("spy", pd.DataFrame()),
        data.get("hyg", pd.DataFrame()),
        data.get("lqd", pd.DataFrame()),
    )

    if len(features) < 60:
        raise ValueError(f"Only {len(features)} observations after feature engineering (need >= 60)")
    if len(features) < 200:
        logger.warning(f"Only {len(features)} observations — results may be noisy (recommend >= 200)")

    # Detect regimes
    model, states, log_likelihood, converged = detect_regime(features, n_states=n_states)

    # Label states
    state_labels = label_regimes(model, features)

    # Map states to labels in history
    regime_labels = [state_labels.get(s, f"State-{s}") for s in states]

    # Build regime history DataFrame
    history = ticker_df.loc[features.index, ["Close"]].copy()
    history["regime_label"] = regime_labels

    # Current regime
    current_regime = regime_labels[-1]

    # Posterior probabilities for current day
    proba = model.predict_proba(features.values)[-1]
    probabilities = {}
    for s_id, prob in enumerate(proba):
        label = state_labels.get(s_id, f"State-{s_id}")
        probabilities[label] = float(prob)

    # VIX ratio feature index for state summaries
    col_names = list(features.columns)
    vix_idx = col_names.index("vix_ratio") if "vix_ratio" in col_names else None

    # Build state summaries
    state_summaries = []
    for s_id in range(n_states):
        mask = states == s_id
        label = state_labels.get(s_id, f"State-{s_id}")
        mean_ret = float(model.means_[s_id][0])  # log_ret
        mean_vol = float(model.means_[s_id][1]) if model.means_.shape[1] > 1 else 0.0
        mean_vix = float(model.means_[s_id][vix_idx]) if vix_idx is not None else 1.0
        freq = float(np.sum(mask) / len(mask) * 100)

        state_summaries.append(RegimeState(
            state_id=s_id,
            label=label,
            mean_return=mean_ret,
            mean_volatility=mean_vol,
            mean_vix_ratio=mean_vix,
            frequency_pct=freq,
        ))

    return RegimeResult(
        ticker=ticker,
        current_regime=current_regime,
        probabilities=probabilities,
        states=state_summaries,
        regime_history=history,
        n_observations=len(features),
        log_likelihood=log_likelihood,
        converged=converged,
    )


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_regime_at_date(regime_result: RegimeResult, target_date: date) -> Optional[str]:
    """Lookup regime label for a historical date (snaps to nearest trading day within 5 days)."""
    history = regime_result.regime_history
    ts = pd.Timestamp(target_date)

    if ts in history.index:
        return history.loc[ts, "regime_label"]

    # Snap to nearest within 5 days
    diffs = abs(history.index - ts)
    min_idx = diffs.argmin()
    if diffs[min_idx] <= pd.Timedelta(days=5):
        return history.iloc[min_idx]["regime_label"]

    return None


def filter_events_by_regime(
    events: list,
    regime_result: RegimeResult,
    target_regime: str,
) -> list:
    """Filter MarketEvent list to those occurring during target regime."""
    filtered = []
    for event in events:
        regime = get_regime_at_date(regime_result, event.date)
        if regime == target_regime:
            filtered.append(event)
    return filtered
