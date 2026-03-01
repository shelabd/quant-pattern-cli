"""Tests for display utilities."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_patterns.analysis import Level, SimilarityResult
from quant_patterns.display import (
    _pct_color,
    _score_color,
    _sparkline,
    ascii_price_chart,
)


# ── _sparkline ────────────────────────────────────────────────────────────────


class TestSparkline:
    def test_basic(self):
        result = _sparkline([1, 2, 3, 4, 5])
        assert len(result) > 0
        assert isinstance(result, str)

    def test_empty(self):
        assert _sparkline([]) == ""

    def test_single_value(self):
        assert _sparkline([5]) == ""

    def test_constant_values(self):
        result = _sparkline([5, 5, 5, 5, 5])
        assert len(result) > 0

    def test_respects_width(self):
        values = list(range(100))
        result = _sparkline(values, width=20)
        assert len(result) == 20

    def test_short_input(self):
        result = _sparkline([1, 10], width=30)
        assert len(result) == 2  # Not resampled since input < width


# ── _pct_color ────────────────────────────────────────────────────────────────


class TestPctColor:
    def test_positive_large(self):
        assert _pct_color(1.0) == "green"

    def test_positive_small(self):
        assert _pct_color(0.3) == "bright_green"

    def test_negative_small(self):
        assert _pct_color(-0.3) == "bright_red"

    def test_negative_large(self):
        assert _pct_color(-1.0) == "red"


# ── _score_color ──────────────────────────────────────────────────────────────


class TestScoreColor:
    def test_thresholds(self):
        assert _score_color(0.9) == "bright_green"
        assert _score_color(0.7) == "green"
        assert _score_color(0.5) == "yellow"
        assert _score_color(0.2) == "red"


# ── ascii_price_chart ─────────────────────────────────────────────────────────


class TestAsciiPriceChart:
    def test_basic_chart(self, sample_ohlcv):
        chart = ascii_price_chart(sample_ohlcv, title="Test")
        assert "Test" in chart
        assert "●" in chart  # Price points
        assert len(chart.split("\n")) > 10

    def test_empty_df(self):
        df = pd.DataFrame()
        assert ascii_price_chart(df) == "[No data]"

    def test_with_sr_levels(self, sample_ohlcv):
        mid = sample_ohlcv["Close"].mean()
        levels = [
            Level(mid - 1, "support", 2, strength=0.5),
            Level(mid + 1, "resistance", 3, strength=0.8),
        ]
        chart = ascii_price_chart(sample_ohlcv, support_resistance=levels)
        assert "S" in chart or "R" in chart  # S/R markers

    def test_custom_dimensions(self, sample_ohlcv):
        chart = ascii_price_chart(sample_ohlcv, height=10, width=40)
        lines = chart.split("\n")
        assert len(lines) > 5
