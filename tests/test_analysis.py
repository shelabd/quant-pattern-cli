"""Tests for the analysis engine: similarity scoring, S/R, profiles."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_patterns.analysis import (
    Level,
    PatternProfile,
    SimilarityResult,
    _simple_dtw,
    build_pattern_profile,
    compare_windows,
    export_for_agent,
    find_support_resistance,
    sliding_window_scan,
)
from quant_patterns.data import normalize_window


# ── DTW ───────────────────────────────────────────────────────────────────────


class TestDTW:
    def test_identical_series(self):
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _simple_dtw(s, s) == 0.0

    def test_different_series(self):
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([4.0, 5.0, 6.0])
        assert _simple_dtw(s1, s2) > 0

    def test_symmetric(self):
        s1 = np.array([1.0, 3.0, 2.0, 5.0])
        s2 = np.array([2.0, 4.0, 1.0, 3.0])
        assert _simple_dtw(s1, s2) == _simple_dtw(s2, s1)

    def test_shifted_series_lower_than_reversed(self):
        """A shifted version should be closer than a reversed version."""
        s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s_shifted = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        s_reversed = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert _simple_dtw(s1, s_shifted) < _simple_dtw(s1, s_reversed)

    def test_different_lengths(self):
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _simple_dtw(s1, s2)
        assert result >= 0


# ── compare_windows ───────────────────────────────────────────────────────────


def _make_normalized_window(close_values, rel_days=None):
    """Helper to create a minimal normalized DataFrame."""
    n = len(close_values)
    if rel_days is None:
        half = n // 2
        rel_days = list(range(-half, n - half))
    dates = pd.bdate_range("2024-01-02", periods=n)
    ref_price = close_values[n // 2]  # event day price
    close_norm = [(c / ref_price - 1) * 100 for c in close_values]
    return pd.DataFrame(
        {
            "Close": close_values,
            "Close_norm": close_norm,
            "rel_day": rel_days,
        },
        index=dates[:n],
    )


class TestCompareWindows:
    def test_identical_windows_high_score(self):
        values = [100, 101, 102, 101, 103, 104, 103, 105, 106, 107, 108]
        w1 = _make_normalized_window(values)
        w2 = _make_normalized_window(values)
        result = compare_windows(w1, w2, "Test", date(2024, 1, 8))
        assert result.composite_score > 0.8
        assert result.correlation > 0.99

    def test_opposite_windows_low_score(self):
        up = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        down = [110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
        w1 = _make_normalized_window(up)
        w2 = _make_normalized_window(down)
        result = compare_windows(w1, w2, "Test", date(2024, 1, 8))
        assert result.correlation < 0
        assert result.composite_score < 0.4

    def test_result_fields(self):
        values = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        w = _make_normalized_window(values)
        result = compare_windows(w, w, "TestEvent", date(2024, 6, 15))
        assert result.event_name == "TestEvent"
        assert result.event_date == date(2024, 6, 15)
        assert 0 <= result.direction_match <= 1
        assert result.volatility_ratio > 0
        assert result.euclidean_dist >= 0
        assert result.dtw_distance >= 0

    def test_score_label_thresholds(self):
        r = SimilarityResult("", date.today(), 0, 0, 0, 0.85, 0, 0)
        assert r.score_label == "Very Similar"
        r.composite_score = 0.65
        assert r.score_label == "Similar"
        r.composite_score = 0.45
        assert r.score_label == "Moderate"
        r.composite_score = 0.2
        assert r.score_label == "Weak"

    def test_short_series_returns_zero_score(self):
        w1 = _make_normalized_window([100, 101], rel_days=[-1, 0])
        w2 = _make_normalized_window([100, 102], rel_days=[-1, 0])
        result = compare_windows(w1, w2)
        assert result.composite_score == 0

    def test_different_lengths_handled(self):
        short = _make_normalized_window([100, 101, 102, 103, 104])
        long = _make_normalized_window([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        result = compare_windows(short, long, "Test", date(2024, 1, 1))
        assert result.composite_score > 0


# ── find_support_resistance ───────────────────────────────────────────────────


class TestSupportResistance:
    def test_returns_levels(self, sample_ohlcv):
        levels = find_support_resistance(sample_ohlcv, window=3)
        assert isinstance(levels, list)
        assert all(isinstance(l, Level) for l in levels)

    def test_level_types(self, sample_ohlcv):
        levels = find_support_resistance(sample_ohlcv, window=3)
        types = {l.kind for l in levels}
        assert types <= {"support", "resistance"}

    def test_strength_normalized(self, sample_ohlcv):
        levels = find_support_resistance(sample_ohlcv, window=3)
        if levels:
            assert max(l.strength for l in levels) == 1.0
            assert all(0 <= l.strength <= 1 for l in levels)

    def test_sorted_by_price(self, sample_ohlcv):
        levels = find_support_resistance(sample_ohlcv, window=3)
        prices = [l.price for l in levels]
        assert prices == sorted(prices)

    def test_too_short_data(self):
        dates = pd.bdate_range("2024-01-02", periods=3)
        df = pd.DataFrame(
            {"Close": [100, 101, 102]},
            index=dates,
        )
        levels = find_support_resistance(df, window=5)
        assert levels == []

    def test_level_label(self):
        l = Level(price=150.50, kind="support", touches=3, strength=0.75)
        assert "S @" in l.label
        assert "150.50" in l.label

        l2 = Level(price=200.0, kind="resistance", touches=2, strength=0.5)
        assert "R @" in l2.label

    def test_num_levels_cap(self):
        # Create data with many extrema
        dates = pd.bdate_range("2024-01-02", periods=100)
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.normal(0, 2, 100))
        df = pd.DataFrame({"Close": close}, index=dates)
        levels = find_support_resistance(df, window=3, num_levels=3)
        supports = [l for l in levels if l.kind == "support"]
        resistances = [l for l in levels if l.kind == "resistance"]
        assert len(supports) <= 3
        assert len(resistances) <= 3


# ── build_pattern_profile ─────────────────────────────────────────────────────


class TestBuildPatternProfile:
    def _make_window(self, seed=42):
        dates = pd.bdate_range("2024-01-08", periods=21)
        np.random.seed(seed)
        close = 100 + np.cumsum(np.random.normal(0, 1, 21))
        return pd.DataFrame(
            {
                "Close": close,
                "Volume": np.random.randint(1_000_000, 5_000_000, 21),
                "rel_day": list(range(-10, 11)),
            },
            index=dates,
        )

    def _make_sim_result(self, score=0.7):
        return SimilarityResult(
            event_name="Test", event_date=date(2024, 1, 15),
            correlation=0.8, euclidean_dist=1.0, dtw_distance=2.0,
            composite_score=score, direction_match=0.6, volatility_ratio=0.9,
        )

    def test_basic_profile(self):
        windows = [self._make_window(seed=i) for i in range(3)]
        results = [self._make_sim_result(0.5 + i * 0.1) for i in range(3)]
        profile = build_pattern_profile("SPY", "fomc", windows, results)

        assert profile.ticker == "SPY"
        assert profile.category == "fomc"
        assert profile.num_events == 3
        assert profile.best_match is not None
        assert profile.best_match.composite_score == 0.7

    def test_all_matches_sorted(self):
        windows = [self._make_window(seed=i) for i in range(5)]
        results = [self._make_sim_result(0.3 + i * 0.1) for i in range(5)]
        profile = build_pattern_profile("SPY", "cpi", windows, results)
        scores = [m.composite_score for m in profile.all_matches]
        assert scores == sorted(scores, reverse=True)

    def test_profile_to_dict(self):
        windows = [self._make_window()]
        results = [self._make_sim_result()]
        profile = build_pattern_profile("SPY", "fomc", windows, results)
        d = profile.to_dict()
        assert d["ticker"] == "SPY"
        assert "avg_return_after_pct" in d
        assert "positive_after_pct" in d

    def test_empty_windows(self):
        profile = build_pattern_profile("SPY", "fomc", [], [])
        assert profile.num_events == 0
        assert profile.avg_return_after == 0
        assert profile.best_match is None


# ── export_for_agent ──────────────────────────────────────────────────────────


class TestExportForAgent:
    def test_export_structure(self):
        windows = [
            pd.DataFrame({
                "Close": [100, 101, 102, 103, 104],
                "Volume": [1e6] * 5,
                "rel_day": [-2, -1, 0, 1, 2],
            }, index=pd.bdate_range("2024-01-02", periods=5))
        ]
        results = [SimilarityResult(
            "Test", date(2024, 1, 4), 0.9, 0.5, 1.0, 0.8, 0.7, 0.95,
        )]
        profile = build_pattern_profile("SPY", "fomc", windows, results)
        sr = [Level(100.0, "support", 3, strength=0.8)]
        target = windows[0]

        export = export_for_agent(profile, sr, target)

        assert export["ticker"] == "SPY"
        assert export["event_category"] == "fomc"
        assert "analysis_summary" in export
        assert "support_resistance" in export
        assert "top_matches" in export
        assert "signal" in export
        assert export["signal"]["direction"] in ("bullish", "bearish")
        assert 0 <= export["signal"]["confidence"] <= 1

    def test_signal_direction(self):
        # Bullish: positive avg_return_after
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
        assert export["signal"]["direction"] == "bullish"


# ── sliding_window_scan ───────────────────────────────────────────────────────


class TestSlidingWindowScan:
    def _make_long_df(self, n=200, seed=42):
        dates = pd.bdate_range("2023-01-02", periods=n)
        np.random.seed(seed)
        close = 100 + np.cumsum(np.random.normal(0, 1, n))
        return pd.DataFrame(
            {
                "Open": close - 0.5,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": np.random.randint(1_000_000, 5_000_000, n),
            },
            index=dates,
        )

    def test_returns_results(self):
        df = self._make_long_df(200)
        results = sliding_window_scan(df, window_size=10, top_n=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_results_are_sorted_by_score(self):
        df = self._make_long_df(200)
        results = sliding_window_scan(df, window_size=10, top_n=10)
        scores = [r.composite_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_results_are_similarity_results(self):
        df = self._make_long_df(200)
        results = sliding_window_scan(df, window_size=10, top_n=3)
        for r in results:
            assert isinstance(r, SimilarityResult)
            assert r.event_name  # should have a date range label
            assert r.event_date is not None
            assert r.composite_score > 0

    def test_step_reduces_results(self):
        df = self._make_long_df(200)
        r1 = sliding_window_scan(df, window_size=10, step=1, top_n=50)
        r5 = sliding_window_scan(df, window_size=10, step=5, top_n=50)
        # Step=5 examines fewer windows, may find fewer unique matches
        assert len(r5) <= len(r1)

    def test_too_short_data(self):
        df = self._make_long_df(10)
        results = sliding_window_scan(df, window_size=10)
        assert results == []

    def test_no_overlap_with_target(self):
        """Historical matches should not overlap with the target window."""
        df = self._make_long_df(200)
        window_size = 10
        results = sliding_window_scan(df, window_size=window_size, top_n=20)
        target_start = df.index[-window_size].date()
        for r in results:
            assert r.event_date < target_start

    def test_deduplication(self):
        """Matches should be spaced apart (no overlapping windows)."""
        df = self._make_long_df(300)
        window_size = 10
        results = sliding_window_scan(df, window_size=window_size, top_n=10)
        for i, r1 in enumerate(results):
            for r2 in results[i + 1:]:
                gap = abs((r1.event_date - r2.event_date).days)
                assert gap >= window_size

    def test_top_n_respected(self):
        df = self._make_long_df(200)
        for n in [1, 3, 7]:
            results = sliding_window_scan(df, window_size=10, top_n=n)
            assert len(results) <= n

    def test_window_data_attached(self):
        df = self._make_long_df(200)
        results = sliding_window_scan(df, window_size=10, top_n=3)
        for r in results:
            assert r.window_data is not None
            assert "Close_norm" in r.window_data.columns
            assert len(r.window_data) == 10
