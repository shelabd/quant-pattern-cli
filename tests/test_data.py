"""Tests for data providers and window extraction."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_patterns.data import (
    DataProvider,
    YFinanceProvider,
    fetch_event_window,
    get_provider,
    normalize_window,
)


# ── get_provider factory ──────────────────────────────────────────────────────


class TestGetProvider:
    def test_yfinance(self):
        p = get_provider("yfinance")
        assert isinstance(p, YFinanceProvider)
        assert p.name() == "Yahoo Finance"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")

    def test_ibkr_without_install(self):
        # IBKRProvider can be instantiated but will fail on use without ib_insync
        p = get_provider("ibkr")
        assert p.name() == "Interactive Brokers"


# ── fetch_event_window ────────────────────────────────────────────────────────


class TestFetchEventWindow:
    def test_returns_dataframe(self, mock_provider):
        df = fetch_event_window(mock_provider, "SPY", date(2024, 6, 15), 5, 5)
        assert isinstance(df, pd.DataFrame)
        assert "Close" in df.columns
        assert "rel_day" in df.columns

    def test_has_rel_day_column(self, mock_provider):
        df = fetch_event_window(mock_provider, "SPY", date(2024, 6, 15), 5, 5)
        assert "rel_day" in df.columns
        assert 0 in df["rel_day"].values

    def test_window_size(self, mock_provider):
        df = fetch_event_window(mock_provider, "SPY", date(2024, 6, 15), 10, 10)
        # Should have approximately 21 trading days (10 before + event + 10 after)
        assert len(df) >= 15  # Allow some slack for date alignment
        assert len(df) <= 25

    def test_ohlcv_columns_present(self, mock_provider):
        df = fetch_event_window(mock_provider, "SPY", date(2024, 6, 15), 5, 5)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns


# ── normalize_window ──────────────────────────────────────────────────────────


class TestNormalizeWindow:
    def test_produces_norm_columns(self, sample_window):
        result = normalize_window(sample_window)
        assert "Close_norm" in result.columns
        assert "Open_norm" in result.columns
        assert "High_norm" in result.columns
        assert "Low_norm" in result.columns

    def test_event_day_close_norm_is_zero(self, sample_window):
        result = normalize_window(sample_window)
        event_day = result[result["rel_day"] == 0]
        assert len(event_day) == 1
        assert abs(event_day["Close_norm"].values[0]) < 1e-10

    def test_norm_values_are_percentages(self, sample_window):
        result = normalize_window(sample_window)
        # Values should be reasonable percentage changes (not raw prices)
        assert result["Close_norm"].abs().max() < 100  # Less than 100% change

    def test_requires_rel_day(self, sample_ohlcv):
        with pytest.raises(ValueError, match="rel_day"):
            normalize_window(sample_ohlcv)

    def test_preserves_original_columns(self, sample_window):
        result = normalize_window(sample_window)
        assert "Close" in result.columns
        assert "Volume" in result.columns
        assert "rel_day" in result.columns

    def test_handles_no_exact_event_day(self):
        """When rel_day=0 is missing, should use closest."""
        dates = pd.bdate_range("2024-01-02", periods=10)
        df = pd.DataFrame(
            {
                "Close": np.linspace(100, 110, 10),
                "Open": np.linspace(99, 109, 10),
                "High": np.linspace(101, 111, 10),
                "Low": np.linspace(98, 108, 10),
                "rel_day": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],  # no 0
            },
            index=dates,
        )
        result = normalize_window(df)
        assert "Close_norm" in result.columns
