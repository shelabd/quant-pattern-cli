"""Shared fixtures for qpat tests."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_patterns.analysis import SimilarityResult
from quant_patterns.data import DataProvider
from quant_patterns.events import EventCatalog, EventCategory, MarketEvent


# ── Mock Data Provider ────────────────────────────────────────────────────────


class MockProvider(DataProvider):
    """In-memory data provider for tests. Returns synthetic OHLCV data."""

    def __init__(self, base_price: float = 100.0, volatility: float = 0.02):
        self.base_price = base_price
        self.volatility = volatility

    def name(self) -> str:
        return "Mock"

    def get_daily_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        dates = pd.bdate_range(start, end)
        if len(dates) == 0:
            raise ValueError(f"No data for {ticker} between {start} and {end}")
        n = len(dates)
        np.random.seed(42)
        returns = np.random.normal(0, self.volatility, n)
        close = self.base_price * np.cumprod(1 + returns)
        high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
        opn = (close + low) / 2
        volume = np.random.randint(1_000_000, 10_000_000, n)

        df = pd.DataFrame(
            {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )
        df.index.name = "Date"
        return df


@pytest.fixture
def mock_provider():
    return MockProvider()


# ── Sample DataFrames ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_ohlcv():
    """20 trading days of synthetic OHLCV data."""
    dates = pd.bdate_range("2024-01-02", periods=20)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.normal(0, 1, 20))
    df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 5_000_000, 20),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def sample_window():
    """A normalized event window with rel_day column."""
    dates = pd.bdate_range("2024-01-08", periods=21)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.normal(0, 1, 21))
    df = pd.DataFrame(
        {
            "Open": close - 0.3,
            "High": close + 0.8,
            "Low": close - 0.8,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 5_000_000, 21),
            "rel_day": list(range(-10, 11)),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


# ── Event Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_events():
    return [
        MarketEvent("Test CPI", date(2024, 1, 11), EventCategory.CPI, "Test CPI release"),
        MarketEvent("Test FOMC", date(2024, 1, 31), EventCategory.FOMC, "Test FOMC decision"),
        MarketEvent("NVDA Earnings", date(2024, 2, 21), EventCategory.EARNINGS, "Test earnings", "NVDA"),
        MarketEvent("AAPL Earnings", date(2024, 10, 31), EventCategory.EARNINGS, "Test earnings", "AAPL"),
    ]


@pytest.fixture
def catalog(tmp_path):
    """EventCatalog with custom events path pointing to tmp_path."""
    return EventCatalog(custom_events_path=tmp_path / "custom_events.json")
