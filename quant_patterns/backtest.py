"""
Walk-forward backtester for qpat's directional signals.

Answers the only question that matters for trusting the tool: at each
historical as-of date, using ONLY data available up to that date, would the
signal's direction have predicted the realized next-N-day return better than
the base rate?

Two signal types are replayed:

- **event**: for each historical event of a category, rebuild the event
  signal from the *prior* events only (same Wilson/binomial machinery as
  ``qpat analyze``), then score it against that event's realized post-event
  return.
- **scan**: at regular as-of dates, run the sliding-window pattern scan on
  the trailing history only, derive a direction from the matches' forward
  returns (score²-weighted, as the dashboard does), and score it against the
  realized forward return from the as-of close.

Scoring is deliberately conservative: the hit rate is tested against the
**majority-class base rate** of the same period (if the ticker rose in 63%
of horizon windows, a strategy of "always bullish" already hits 63% — the
signal must beat that, not 50%). This module is pure logic except
:func:`run_backtest`; everything else is unit-testable offline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .analysis import signal_stats_from_returns, sliding_window_scan

logger = logging.getLogger(__name__)

DEFAULT_HORIZON = 10        # trading days the signal is scored over
DEFAULT_MIN_HISTORY = 5     # prior events required before emitting a signal
DEFAULT_SCAN_WINDOW = 10
MIN_HISTORY_ROWS = 120      # rows of history before the first scan signal
MAX_SCAN_LOOKBACK_ROWS = 750


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class SignalOutcome:
    """One walk-forward signal and what actually happened."""
    as_of: date
    signal_type: str            # "event" | "scan"
    direction: str              # "bullish" | "bearish"
    confidence: float           # signal's own confidence at the time
    predicted_edge_pct: float   # what the signal expected over the horizon
    realized_return_pct: float  # what the ticker actually did
    hit: bool                   # direction matched the realized sign
    n_basis: int                # sample behind the signal (prior events / matches)
    label: str = ""             # e.g. the event name

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "signal_type": self.signal_type,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "predicted_edge_pct": round(self.predicted_edge_pct, 3),
            "realized_return_pct": round(self.realized_return_pct, 3),
            "hit": self.hit,
            "n_basis": self.n_basis,
            "label": self.label,
        }


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    n: int
    hit_rate: float

    def to_dict(self) -> dict:
        return {"confidence_range": f"{self.lo:.1f}-{self.hi:.1f}",
                "n": self.n, "hit_rate_pct": round(self.hit_rate * 100, 1)}


@dataclass
class BacktestReport:
    """Out-of-sample scorecard for one signal type."""
    ticker: str
    signal_type: str
    horizon_days: int
    n_signals: int
    hit_rate: float                 # 0-1
    up_rate: float                  # fraction of realized returns > 0
    majority_baseline: float        # max(up_rate, 1-up_rate) — the bar to beat
    p_value: float                  # binomial test of hits vs majority baseline
    avg_signal_return_pct: float    # mean realized return in the signal's direction
    avg_bullish_hit_rate: float
    avg_bearish_hit_rate: float
    n_bullish: int
    n_bearish: int
    calibration: list[CalibrationBin] = field(default_factory=list)
    outcomes: list[SignalOutcome] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def predictive(self) -> bool:
        return self.p_value < 0.10

    @property
    def verdict(self) -> str:
        if self.n_signals == 0:
            return "NO SIGNALS"
        return ("PREDICTIVE at the 10% level" if self.predictive
                else "NOT distinguishable from baseline")

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "signal_type": self.signal_type,
            "horizon_days": self.horizon_days,
            "n_signals": self.n_signals,
            "hit_rate_pct": round(self.hit_rate * 100, 1),
            "up_rate_pct": round(self.up_rate * 100, 1),
            "majority_baseline_pct": round(self.majority_baseline * 100, 1),
            "p_value": round(self.p_value, 4),
            "predictive_at_10pct": self.predictive,
            "verdict": self.verdict,
            "avg_signal_return_pct": round(self.avg_signal_return_pct, 3),
            "bullish": {"n": self.n_bullish, "hit_rate_pct": round(self.avg_bullish_hit_rate * 100, 1)},
            "bearish": {"n": self.n_bearish, "hit_rate_pct": round(self.avg_bearish_hit_rate * 100, 1)},
            "calibration": [b.to_dict() for b in self.calibration],
            "notes": list(self.notes),
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_outcomes(
    ticker: str,
    signal_type: str,
    horizon_days: int,
    outcomes: list[SignalOutcome],
    notes: Optional[list[str]] = None,
) -> BacktestReport:
    """Aggregate walk-forward outcomes into a scorecard.

    The null hypothesis is the majority-class strategy: always predicting
    the period's more common direction. A signal that "predicts" up 100% of
    the time in a bull market matches the majority baseline exactly and is
    correctly reported as non-predictive.
    """
    notes = list(notes or [])
    if not outcomes:
        return BacktestReport(
            ticker=ticker, signal_type=signal_type, horizon_days=horizon_days,
            n_signals=0, hit_rate=0.0, up_rate=0.0, majority_baseline=0.0,
            p_value=1.0, avg_signal_return_pct=0.0,
            avg_bullish_hit_rate=0.0, avg_bearish_hit_rate=0.0,
            n_bullish=0, n_bearish=0, notes=notes,
        )

    n = len(outcomes)
    hits = sum(1 for o in outcomes if o.hit)
    up_rate = sum(1 for o in outcomes if o.realized_return_pct > 0) / n
    majority = max(up_rate, 1 - up_rate)
    majority = min(0.999, max(0.001, majority))
    p_value = float(binomtest(hits, n, majority, alternative="greater").pvalue)

    signed = [o.realized_return_pct if o.direction == "bullish"
              else -o.realized_return_pct for o in outcomes]

    bulls = [o for o in outcomes if o.direction == "bullish"]
    bears = [o for o in outcomes if o.direction == "bearish"]

    if n < 30:
        notes.append(f"Only {n} walk-forward signals — power is low; "
                     "a real edge could hide and a fluke could shine.")

    return BacktestReport(
        ticker=ticker,
        signal_type=signal_type,
        horizon_days=horizon_days,
        n_signals=n,
        hit_rate=hits / n,
        up_rate=up_rate,
        majority_baseline=majority,
        p_value=p_value,
        avg_signal_return_pct=float(np.mean(signed)),
        avg_bullish_hit_rate=(sum(o.hit for o in bulls) / len(bulls)) if bulls else 0.0,
        avg_bearish_hit_rate=(sum(o.hit for o in bears) / len(bears)) if bears else 0.0,
        n_bullish=len(bulls),
        n_bearish=len(bears),
        calibration=calibration_bins(outcomes),
        outcomes=outcomes,
        notes=notes,
    )


def calibration_bins(
    outcomes: list[SignalOutcome],
    edges: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.01),
) -> list[CalibrationBin]:
    """Hit rate per confidence bin. A calibrated signal hits more often when
    it claims more confidence; a flat curve means confidence is decorative."""
    bins: list[CalibrationBin] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        members = [o for o in outcomes if lo <= o.confidence < hi]
        if members:
            bins.append(CalibrationBin(
                lo=lo, hi=min(hi, 1.0), n=len(members),
                hit_rate=sum(o.hit for o in members) / len(members)))
    return bins


# ── Event signal walk-forward ────────────────────────────────────────────────

def walk_forward_event_signals(
    event_returns: list[tuple[date, str, float]],
    min_history: int = DEFAULT_MIN_HISTORY,
) -> list[SignalOutcome]:
    """Replay the event signal through history without lookahead.

    event_returns: chronologically sorted (event_date, event_name,
    post_event_return_pct). For each event with >= min_history priors, the
    signal is rebuilt from the prior returns only and scored against that
    event's own realized return. Neutral signals (no edge in the priors)
    emit nothing — qpat would not have given a direction.
    """
    outcomes: list[SignalOutcome] = []
    for i in range(min_history, len(event_returns)):
        prior = [r for _, _, r in event_returns[:i]]
        stats = signal_stats_from_returns(prior)
        if stats.direction == "neutral":
            continue
        as_of, name, realized = event_returns[i]
        hit = (realized > 0) if stats.direction == "bullish" else (realized < 0)
        outcomes.append(SignalOutcome(
            as_of=as_of,
            signal_type="event",
            direction=stats.direction,
            confidence=stats.confidence,
            predicted_edge_pct=stats.edge_pct,
            realized_return_pct=realized,
            hit=hit,
            n_basis=stats.n,
            label=name,
        ))
    return outcomes


def post_event_return(window: pd.DataFrame, horizon: int) -> Optional[float]:
    """Realized % return from the event-day close to +horizon trading days.

    Requires the full horizon to exist in the window (recent events whose
    aftermath hasn't finished printing are excluded, not truncated)."""
    if window.empty or "rel_day" not in window.columns:
        return None
    event = window[window["rel_day"] == 0]["Close"]
    target = window[window["rel_day"] == horizon]["Close"]
    if event.empty or target.empty or event.values[0] == 0:
        return None
    return float((target.values[0] / event.values[0] - 1) * 100)


# ── Scan signal walk-forward ─────────────────────────────────────────────────

def walk_forward_scan_signals(
    df: pd.DataFrame,
    window_size: int = DEFAULT_SCAN_WINDOW,
    horizon: int = DEFAULT_HORIZON,
    step: Optional[int] = None,
    top_n: int = 5,
    min_history_rows: int = MIN_HISTORY_ROWS,
    max_lookback_rows: int = MAX_SCAN_LOOKBACK_ROWS,
) -> list[SignalOutcome]:
    """Replay the pattern-scan signal through history without lookahead.

    At each as-of index t (every `step` trading days), the scan sees only
    df[:t+1] (capped to a trailing max_lookback_rows). The prediction is the
    score²-weighted mean of the matches' next-`horizon` returns — the same
    aggregation the dashboard uses — and is scored against the realized
    return from the close at t to the close at t+horizon.

    step defaults to the horizon so consecutive outcomes score
    non-overlapping windows: overlapping outcomes are autocorrelated and
    make the binomial p-value optimistic.
    """
    if step is None:
        step = horizon
    close = df["Close"].values
    outcomes: list[SignalOutcome] = []

    for t in range(min_history_rows, len(df) - horizon, step):
        lo = max(0, t + 1 - max_lookback_rows)
        history = df.iloc[lo:t + 1]
        matches = sliding_window_scan(history, window_size=window_size,
                                      step=1, top_n=top_n)
        if not matches:
            continue

        hist_close = history["Close"].values
        weighted: list[tuple[float, float]] = []
        scores: list[float] = []
        for m in matches:
            if m.event_date is None:
                continue
            idx = history.index.searchsorted(pd.Timestamp(m.event_date))
            end = idx + window_size  # first index after the matched window
            fwd_idx = end - 1 + horizon
            if end - 1 < 0 or fwd_idx >= len(hist_close):
                continue
            base_px = hist_close[end - 1]
            if base_px == 0:
                continue
            fwd_ret = (hist_close[fwd_idx] / base_px - 1) * 100
            weighted.append((fwd_ret, m.composite_score ** 2))
            scores.append(m.composite_score)

        total_w = sum(w for _, w in weighted)
        if not weighted or total_w == 0:
            continue
        predicted = sum(r * w for r, w in weighted) / total_w
        if predicted == 0:
            continue
        direction = "bullish" if predicted > 0 else "bearish"
        confidence = float(np.mean(scores))

        realized = (close[t + horizon] / close[t] - 1) * 100
        hit = (realized > 0) if direction == "bullish" else (realized < 0)
        as_of = df.index[t].date() if hasattr(df.index[t], "date") else df.index[t]

        outcomes.append(SignalOutcome(
            as_of=as_of,
            signal_type="scan",
            direction=direction,
            confidence=confidence,
            predicted_edge_pct=float(predicted),
            realized_return_pct=float(realized),
            hit=hit,
            n_basis=len(weighted),
        ))
    return outcomes


# ── Orchestrator (network lives here only) ───────────────────────────────────

def run_backtest(
    ticker: str,
    categories: list,
    horizon: int = DEFAULT_HORIZON,
    mode: str = "both",
    window_size: int = DEFAULT_SCAN_WINDOW,
    step: Optional[int] = None,
    lookback_days: int = 2000,
    min_history: int = DEFAULT_MIN_HISTORY,
    progress_cb=None,
) -> list[BacktestReport]:
    """Fetch data and run the requested walk-forward backtests.

    categories: list of EventCategory for the event leg. mode: "events",
    "scan", or "both". progress_cb, when given, is called with status
    strings for display purposes.
    """
    from datetime import timedelta
    from .data import fetch_event_window, get_provider
    from .events import EventCatalog

    def tick(msg: str):
        if progress_cb:
            progress_cb(msg)

    ticker = ticker.upper()
    today = date.today()
    provider = get_provider("yfinance")
    reports: list[BacktestReport] = []

    if mode in ("events", "both"):
        catalog = EventCatalog()
        events = []
        for cat in categories:
            events.extend(catalog.search(category=cat, ticker=ticker, end=today))
        events.sort(key=lambda e: e.date)

        event_returns: list[tuple[date, str, float]] = []
        skipped = 0
        for evt in events:
            tick(f"event window {evt.date} ({evt.name})")
            try:
                window = fetch_event_window(provider, ticker, evt.date,
                                            days_before=1, days_after=horizon)
            except Exception as e:
                logger.debug(f"Skipping {evt.name}: {e}")
                skipped += 1
                continue
            ret = post_event_return(window, horizon)
            if ret is None:
                skipped += 1
                continue
            event_returns.append((evt.date, evt.name, ret))

        notes = [f"{len(event_returns)} events usable, {skipped} skipped "
                 "(no data near date, or aftermath not finished printing)"]
        gaps = [(b[0] - a[0]).days for a, b in zip(event_returns, event_returns[1:])]
        if any(g < horizon for g in gaps):
            notes.append(
                f"Some events are closer together than the {horizon}d scoring "
                "horizon — outcomes overlap, so the p-value is optimistic.")
        outcomes = walk_forward_event_signals(event_returns, min_history)
        cat_label = "+".join(c.value for c in categories)
        report = score_outcomes(ticker, f"event ({cat_label})", horizon, outcomes, notes)
        reports.append(report)

    if mode in ("scan", "both"):
        tick("fetching scan history")
        df = provider.get_daily_ohlcv(ticker, today - timedelta(days=lookback_days), today)
        tick(f"walking {len(df)} trading days")
        eff_step = step if step is not None else horizon
        outcomes = walk_forward_scan_signals(
            df, window_size=window_size, horizon=horizon, step=eff_step)
        notes = [f"as-of dates every {eff_step} trading days over "
                 f"{len(df)} days of history; horizon {horizon}d"]
        if eff_step < horizon:
            notes.append("step < horizon: scoring windows overlap — "
                         "outcomes are autocorrelated and the p-value is optimistic.")
        reports.append(score_outcomes(ticker, "scan", horizon, outcomes, notes))

    return reports
