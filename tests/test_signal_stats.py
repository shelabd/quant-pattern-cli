"""Tests for the statistical significance layer: Wilson confidence,
binomial p-values, and baseline comparison."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_patterns.analysis import (
    BaselineStats,
    PatternProfile,
    _wilson_lower,
    compute_baseline_stats,
    compute_signal_stats,
    export_for_agent,
    build_pattern_profile,
    SimilarityResult,
)


def _make_profile(returns_after):
    rets = list(returns_after)
    n = len(rets)
    wins = sum(1 for r in rets if r > 0)
    return PatternProfile(
        ticker="SPY", category="fomc", num_events=n,
        avg_return_before=0.0,
        avg_return_after=float(np.mean(rets)) if rets else 0.0,
        avg_return_event_day=0.0,
        median_return_after=float(np.median(rets)) if rets else 0.0,
        positive_after_pct=(wins / n * 100) if n else 0.0,
        avg_volatility=0.01, avg_volume_change=0.0,
        returns_after_list=rets,
    )


# ── Wilson lower bound ────────────────────────────────────────────────────────


class TestWilsonLower:
    def test_zero_n(self):
        assert _wilson_lower(0, 0) == 0.0

    def test_small_sample_shrinks(self):
        """7/10 wins must claim far less confidence than 70/100 wins."""
        small = _wilson_lower(7, 10)
        large = _wilson_lower(70, 100)
        assert small < large
        assert small < 0.45  # raw rate is 0.70; shrunk well below

    def test_bounds(self):
        for wins, n in [(0, 10), (5, 10), (10, 10), (1, 1), (99, 100)]:
            v = _wilson_lower(wins, n)
            assert 0.0 <= v <= 1.0

    def test_perfect_record_not_certain(self):
        assert _wilson_lower(10, 10) < 1.0

    def test_monotonic_in_wins(self):
        vals = [_wilson_lower(w, 20) for w in range(21)]
        assert vals == sorted(vals)


# ── Baseline stats ────────────────────────────────────────────────────────────


class TestBaselineStats:
    def _df(self, n=300, drift=0.001, seed=42):
        dates = pd.bdate_range("2022-01-03", periods=n)
        np.random.seed(seed)
        close = 100 * np.cumprod(1 + np.random.normal(drift, 0.01, n))
        return pd.DataFrame({"Close": close}, index=dates)

    def test_basic(self):
        b = compute_baseline_stats(self._df(), horizon_days=10)
        assert b is not None
        assert 0 <= b.win_rate <= 1
        assert b.n > 200
        assert b.horizon_days == 10

    def test_upward_drift_high_win_rate(self):
        b = compute_baseline_stats(self._df(drift=0.005), horizon_days=10)
        assert b.win_rate > 0.7
        assert b.mean_return_pct > 0

    def test_too_short_returns_none(self):
        assert compute_baseline_stats(self._df(n=15), horizon_days=10) is None

    def test_empty_returns_none(self):
        assert compute_baseline_stats(pd.DataFrame(), horizon_days=10) is None
        assert compute_baseline_stats(None, horizon_days=10) is None


# ── Signal stats ──────────────────────────────────────────────────────────────


class TestComputeSignalStats:
    def test_empty_profile_neutral(self):
        stats = compute_signal_stats(_make_profile([]))
        assert stats.direction == "neutral"
        assert stats.confidence == 0.0
        assert stats.p_value == 1.0

    def test_bullish_direction(self):
        stats = compute_signal_stats(_make_profile([1.0, 2.0, -0.5, 1.5, 0.8]))
        assert stats.direction == "bullish"
        assert stats.wins == 4
        assert stats.n == 5

    def test_bearish_direction(self):
        stats = compute_signal_stats(_make_profile([-1.0, -2.0, 0.5, -1.5, -0.8]))
        assert stats.direction == "bearish"
        assert 0 < stats.confidence < 1

    def test_outlier_win_among_losses_is_neutral(self):
        """Mean up but majority down: one +10% outlier against four small
        losses is not coherent directional evidence — must abstain, not
        call 'bullish' with a contradictory win count."""
        stats = compute_signal_stats(_make_profile([10.0, -1.0, -1.0, -1.0, -1.0]))
        assert stats.direction == "neutral"
        assert stats.confidence == 0.0
        assert stats.p_value == 1.0
        assert stats.edge_pct > 0  # the raw mean is still reported

    def test_outlier_loss_among_wins_is_neutral(self):
        stats = compute_signal_stats(_make_profile([-10.0, 1.0, 1.0, 1.0, 1.0]))
        assert stats.direction == "neutral"

    def test_split_wins_is_neutral(self):
        """Exactly half wins: no majority, no direction."""
        stats = compute_signal_stats(_make_profile([2.0, 1.0, -1.0, -0.5]))
        assert stats.direction == "neutral"

    def test_small_sample_low_confidence(self):
        """The old raw win rate said 70% on 7/10; Wilson must shrink it."""
        rets = [1.0] * 7 + [-1.0] * 3
        stats = compute_signal_stats(_make_profile(rets))
        assert stats.confidence < 0.45

    def test_coin_flip_not_significant(self):
        rets = [1.0] * 6 + [-1.0] * 4
        stats = compute_signal_stats(_make_profile(rets))
        assert stats.p_value > 0.10
        assert not stats.significant

    def test_strong_edge_significant(self):
        rets = [1.0] * 28 + [-1.0] * 2
        stats = compute_signal_stats(_make_profile(rets))
        assert stats.p_value < 0.10
        assert stats.significant

    def test_baseline_raises_bar(self):
        """A 70% win rate is meaningless when the ticker is up 70% of the
        time anyway — p-value vs that baseline must exceed p-value vs 0.5."""
        rets = [1.0] * 7 + [-1.0] * 3
        profile = _make_profile(rets)
        no_base = compute_signal_stats(profile)
        high_base = compute_signal_stats(
            profile, BaselineStats(win_rate=0.70, mean_return_pct=0.9, n=500, horizon_days=10))
        assert high_base.p_value > no_base.p_value
        assert not high_base.significant

    def test_excess_edge(self):
        rets = [2.0, 1.0, 3.0, -1.0]
        base = BaselineStats(win_rate=0.55, mean_return_pct=0.5, n=500, horizon_days=10)
        stats = compute_signal_stats(_make_profile(rets), base)
        assert stats.excess_edge_pct == pytest.approx(np.mean(rets) - 0.5)

    def test_to_dict_schema(self):
        stats = compute_signal_stats(_make_profile([1.0, -1.0, 2.0]))
        d = stats.to_dict()
        for key in ("direction", "confidence", "historical_edge_pct", "n_events",
                    "win_rate_pct", "p_value", "significant_at_10pct"):
            assert key in d


# ── Export integration ────────────────────────────────────────────────────────


class TestExportSignal:
    def test_export_signal_enriched(self):
        windows = [
            pd.DataFrame({
                "Close": [100, 99, 100, 103, 105],
                "Volume": [1e6] * 5,
                "rel_day": [-2, -1, 0, 1, 2],
            }, index=pd.bdate_range("2024-01-02", periods=5))
        ]
        results = [SimilarityResult("T", date(2024, 1, 4), 0.9, 0.5, 1.0, 0.8, 0.7, 0.95)]
        profile = build_pattern_profile("SPY", "fomc", windows, results)
        export = export_for_agent(profile, [], windows[0])

        sig = export["signal"]
        assert sig["direction"] == "bullish"
        assert 0 <= sig["confidence"] <= 1
        assert sig["n_events"] == 1
        assert "p_value" in sig
        assert "significant_at_10pct" in sig

    def test_export_with_precomputed_stats(self):
        windows = [
            pd.DataFrame({
                "Close": [100, 99, 100, 103, 105],
                "Volume": [1e6] * 5,
                "rel_day": [-2, -1, 0, 1, 2],
            }, index=pd.bdate_range("2024-01-02", periods=5))
        ]
        results = [SimilarityResult("T", date(2024, 1, 4), 0.9, 0.5, 1.0, 0.8, 0.7, 0.95)]
        profile = build_pattern_profile("SPY", "fomc", windows, results)
        base = BaselineStats(win_rate=0.6, mean_return_pct=0.4, n=500, horizon_days=2)
        stats = compute_signal_stats(profile, base)
        export = export_for_agent(profile, [], windows[0], signal_stats=stats)
        assert export["signal"]["baseline"]["win_rate_pct"] == 60.0
        assert export["signal"]["excess_edge_pct"] is not None
