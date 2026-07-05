"""Offline tests for the walk-forward backtester. No network access."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from quant_patterns.backtest import (
    SignalOutcome,
    calibration_bins,
    post_event_return,
    score_outcomes,
    walk_forward_event_signals,
    walk_forward_scan_signals,
)
from quant_patterns.analysis import signal_stats_from_returns, compute_signal_stats


def _outcome(direction="bullish", realized=1.0, confidence=0.5, as_of=None):
    hit = (realized > 0) if direction == "bullish" else (realized < 0)
    return SignalOutcome(
        as_of=as_of or date(2025, 1, 1), signal_type="event",
        direction=direction, confidence=confidence,
        predicted_edge_pct=0.5, realized_return_pct=realized,
        hit=hit, n_basis=10,
    )


def _event_returns(returns, start=date(2023, 1, 1), gap_days=30):
    return [(start + timedelta(days=i * gap_days), f"EVT {i}", r)
            for i, r in enumerate(returns)]


# ── signal_stats_from_returns parity ─────────────────────────────────────────


class TestSignalStatsFromReturns:
    def test_matches_compute_signal_stats(self):
        from quant_patterns.analysis import PatternProfile
        rets = [1.0, -0.5, 2.0, 0.3, -1.1, 0.8]
        profile = PatternProfile(
            ticker="SPY", category="fomc", num_events=6,
            avg_return_before=0, avg_return_after=float(np.mean(rets)),
            avg_return_event_day=0, median_return_after=0,
            positive_after_pct=0, avg_volatility=0, avg_volume_change=0,
            returns_after_list=rets,
        )
        a = compute_signal_stats(profile)
        b = signal_stats_from_returns(rets)
        assert a.direction == b.direction
        assert a.confidence == b.confidence
        assert a.p_value == b.p_value


# ── Event walk-forward ────────────────────────────────────────────────────────


class TestWalkForwardEvents:
    def test_min_history_respected(self):
        outcomes = walk_forward_event_signals(_event_returns([1.0] * 10), min_history=5)
        assert len(outcomes) == 5  # events 5..9 scored, 0..4 are warm-up

    def test_no_lookahead_direction(self):
        # Priors all positive, current event negative: the signal must still
        # be bullish (built from priors only) and be scored as a miss.
        rets = [1.0, 1.0, 1.0, 1.0, 1.0, -2.0]
        outcomes = walk_forward_event_signals(_event_returns(rets), min_history=5)
        assert len(outcomes) == 1
        assert outcomes[0].direction == "bullish"
        assert outcomes[0].hit is False
        assert outcomes[0].realized_return_pct == -2.0

    def test_all_positive_all_hits(self):
        outcomes = walk_forward_event_signals(_event_returns([1.0] * 12), min_history=5)
        assert all(o.hit for o in outcomes)
        assert all(o.direction == "bullish" for o in outcomes)

    def test_neutral_priors_emit_nothing(self):
        # Priors where the mean (dragged up by one outlier) and the win
        # majority disagree are neutral — the backtester must abstain
        # rather than score a signal qpat would not have given.
        rets = [10.0, -1.0, -1.0, -1.0, -1.0, 2.0]
        outcomes = walk_forward_event_signals(_event_returns(rets), min_history=5)
        assert outcomes == []

    def test_bearish_priors_bearish_signal(self):
        rets = [-1.0] * 6 + [-0.5]
        outcomes = walk_forward_event_signals(_event_returns(rets), min_history=5)
        assert outcomes[-1].direction == "bearish"
        assert outcomes[-1].hit is True

    def test_n_basis_grows(self):
        outcomes = walk_forward_event_signals(_event_returns([1.0] * 10), min_history=5)
        assert [o.n_basis for o in outcomes] == [5, 6, 7, 8, 9]

    def test_too_few_events_empty(self):
        assert walk_forward_event_signals(_event_returns([1.0] * 4), min_history=5) == []


# ── post_event_return ─────────────────────────────────────────────────────────


class TestPostEventReturn:
    def _window(self, rel_days, closes):
        return pd.DataFrame({"rel_day": rel_days, "Close": closes},
                            index=pd.bdate_range("2024-01-02", periods=len(rel_days)))

    def test_basic(self):
        w = self._window([-1, 0, 1, 2], [99, 100, 101, 103])
        assert post_event_return(w, horizon=2) == pytest.approx(3.0)

    def test_incomplete_horizon_returns_none(self):
        # aftermath not finished printing: only +1 day exists, horizon is 5
        w = self._window([-1, 0, 1], [99, 100, 101])
        assert post_event_return(w, horizon=5) is None

    def test_empty_window(self):
        assert post_event_return(pd.DataFrame(), horizon=5) is None


# ── Scoring ───────────────────────────────────────────────────────────────────


class TestScoreOutcomes:
    def test_empty(self):
        r = score_outcomes("SPY", "event", 10, [])
        assert r.n_signals == 0
        assert r.p_value == 1.0
        assert r.verdict == "NO SIGNALS"

    def test_majority_baseline_is_the_bar(self):
        # 80% of windows up; signal always bullish → hit rate 0.8 == baseline.
        # That must NOT count as predictive.
        outcomes = ([_outcome("bullish", 1.0)] * 8) + ([_outcome("bullish", -1.0)] * 2)
        r = score_outcomes("SPY", "event", 10, outcomes)
        assert r.hit_rate == pytest.approx(0.8)
        assert r.majority_baseline == pytest.approx(0.8)
        assert r.p_value > 0.10
        assert not r.predictive

    def test_genuine_skill_detected(self):
        # 50/50 market, signal right 28/30 times → far beyond majority baseline
        outcomes = ([_outcome("bullish", 1.0)] * 14 +
                    [_outcome("bearish", -1.0)] * 14 +
                    [_outcome("bullish", -1.0)] * 1 +
                    [_outcome("bearish", 1.0)] * 1)
        r = score_outcomes("SPY", "event", 10, outcomes)
        assert r.hit_rate == pytest.approx(28 / 30)
        assert r.p_value < 0.01
        assert r.predictive

    def test_avg_signal_return_signs(self):
        outcomes = [_outcome("bullish", 2.0), _outcome("bearish", -3.0)]
        r = score_outcomes("SPY", "event", 10, outcomes)
        # bullish +2 counts +2; bearish -3 counts +3 → avg 2.5
        assert r.avg_signal_return_pct == pytest.approx(2.5)

    def test_small_sample_note(self):
        r = score_outcomes("SPY", "event", 10, [_outcome()] * 10)
        assert any("power is low" in n for n in r.notes)

    def test_directional_breakdown(self):
        outcomes = [_outcome("bullish", 1.0), _outcome("bullish", -1.0),
                    _outcome("bearish", -1.0)]
        r = score_outcomes("SPY", "event", 10, outcomes)
        assert r.n_bullish == 2 and r.n_bearish == 1
        assert r.avg_bullish_hit_rate == pytest.approx(0.5)
        assert r.avg_bearish_hit_rate == pytest.approx(1.0)

    def test_to_dict_schema(self):
        r = score_outcomes("SPY", "event", 10, [_outcome()] * 6)
        d = r.to_dict()
        for key in ("hit_rate_pct", "majority_baseline_pct", "p_value",
                    "predictive_at_10pct", "verdict", "calibration", "outcomes"):
            assert key in d


class TestCalibrationBins:
    def test_bins_partition_by_confidence(self):
        outcomes = [_outcome(confidence=0.1, realized=1.0),
                    _outcome(confidence=0.1, realized=-1.0),
                    _outcome(confidence=0.9, realized=1.0)]
        bins = calibration_bins(outcomes)
        assert len(bins) == 2
        assert bins[0].n == 2 and bins[0].hit_rate == pytest.approx(0.5)
        assert bins[1].n == 1 and bins[1].hit_rate == pytest.approx(1.0)

    def test_empty(self):
        assert calibration_bins([]) == []


# ── Scan walk-forward ─────────────────────────────────────────────────────────


def _trending_df(n=300, drift=0.002, vol=0.01, seed=42, start="2023-01-02"):
    dates = pd.bdate_range(start, periods=n)
    np.random.seed(seed)
    close = 100 * np.cumprod(1 + np.random.normal(drift, vol, n))
    return pd.DataFrame({
        "Open": close, "High": close * 1.005, "Low": close * 0.995,
        "Close": close, "Volume": np.random.randint(1e6, 5e6, n),
    }, index=dates)


class TestWalkForwardScan:
    def test_produces_outcomes(self):
        df = _trending_df(300)
        outcomes = walk_forward_scan_signals(df, window_size=10, horizon=10,
                                             step=10, min_history_rows=120)
        assert len(outcomes) > 0
        for o in outcomes:
            assert o.signal_type == "scan"
            assert o.direction in ("bullish", "bearish")
            assert o.n_basis > 0

    def test_realized_return_is_true_forward_return(self):
        df = _trending_df(300)
        horizon = 10
        outcomes = walk_forward_scan_signals(df, window_size=10, horizon=horizon,
                                             step=10, min_history_rows=120)
        close = df["Close"].values
        dates = [d.date() for d in df.index]
        for o in outcomes:
            t = dates.index(o.as_of)
            expected = (close[t + horizon] / close[t] - 1) * 100
            assert o.realized_return_pct == pytest.approx(expected)

    def test_no_outcome_within_horizon_of_data_end(self):
        # the last `horizon` days cannot be scored — no realized return yet
        df = _trending_df(300)
        horizon = 10
        outcomes = walk_forward_scan_signals(df, window_size=10, horizon=horizon,
                                             step=10, min_history_rows=120)
        last_scoreable = df.index[len(df) - horizon - 1].date()
        assert all(o.as_of <= last_scoreable for o in outcomes)

    def test_step_defaults_to_horizon(self):
        df = _trending_df(300)
        a = walk_forward_scan_signals(df, window_size=10, horizon=10,
                                      min_history_rows=120)
        b = walk_forward_scan_signals(df, window_size=10, horizon=10, step=10,
                                      min_history_rows=120)
        assert [o.as_of for o in a] == [o.as_of for o in b]

    def test_too_short_history_empty(self):
        df = _trending_df(100)
        outcomes = walk_forward_scan_signals(df, window_size=10, horizon=10,
                                             step=10, min_history_rows=120)
        assert outcomes == []
