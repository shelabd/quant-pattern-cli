"""
Data providers for historical price data.

Primary: yfinance (free, no auth)
Optional: IBKR via ib_insync (requires TWS/Gateway running)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Base class for price data providers."""

    @abstractmethod
    def get_daily_ohlcv(
        self, ticker: str, start: date, end: date
    ) -> pd.DataFrame:
        """
        Return DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (date only, no tz)
        """
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class YFinanceProvider(DataProvider):
    """Yahoo Finance data via yfinance.

    Caches responses in-process: commands like dashboard/analyze fetch
    overlapping ranges for the same ticker many times per run.
    """

    def __init__(self):
        self._cache: dict[tuple[str, date, date, bool], pd.DataFrame] = {}

    def name(self) -> str:
        return "Yahoo Finance"

    def get_daily_ohlcv(
        self, ticker: str, start: date, end: date, auto_adjust: bool = True
    ) -> pd.DataFrame:
        """auto_adjust=False returns raw traded prices — required when
        scoring journaled levels: adjusted history shifts under raw journal
        prices at every ex-dividend date."""
        cache_key = (ticker, start, end, auto_adjust)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        # yfinance end is exclusive, add 1 day
        end_adj = end + timedelta(days=1)
        logger.info(f"Fetching {ticker} from yfinance: {start} → {end}")

        tk = yf.Ticker(ticker)
        df = tk.history(start=start.isoformat(), end=end_adj.isoformat(), auto_adjust=auto_adjust)

        if df.empty:
            raise ValueError(f"No data returned for {ticker} between {start} and {end}")

        # Normalize
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        self._cache[cache_key] = df
        return df.copy()


class IBKRProvider(DataProvider):
    """
    Interactive Brokers data via ib_insync.
    Requires TWS or IB Gateway running locally.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id

    def name(self) -> str:
        return "Interactive Brokers"

    def get_daily_ohlcv(
        self, ticker: str, start: date, end: date
    ) -> pd.DataFrame:
        try:
            from ib_insync import IB, Stock, util
        except ImportError:
            raise ImportError(
                "ib_insync not installed. Install with: pip install ib_insync\n"
                "Also requires TWS or IB Gateway running."
            )

        ib = IB()
        try:
            ib.connect(self.host, self.port, clientId=self.client_id)

            contract = Stock(ticker, "SMART", "USD")
            ib.qualifyContracts(contract)

            duration_days = (end - start).days + 1
            duration_str = f"{duration_days} D"

            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end.strftime("%Y%m%d 23:59:59"),
                durationStr=duration_str,
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )

            df = util.df(bars)
            if df.empty:
                raise ValueError(f"No IBKR data for {ticker}")

            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            })
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df = df[(df.index.date >= start) & (df.index.date <= end)]
            df.index.name = "Date"
            return df
        finally:
            ib.disconnect()


def fetch_ticker_info(ticker: str) -> dict:
    """
    Fetch ticker metadata (name, description, type, sector, etc.) via yfinance.
    Returns a dict with keys: name, description, quote_type, sector, industry,
    exchange, market_cap, currency. Missing fields are None.
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
    except Exception as e:
        logger.warning(f"Could not fetch info for {ticker}: {e}")
        return {"name": ticker, "description": None, "quote_type": None,
                "sector": None, "industry": None, "exchange": None,
                "market_cap": None, "currency": None}

    # Build a short description: first two sentences of the summary
    summary = info.get("longBusinessSummary") or ""
    short_desc = ". ".join(summary.split(". ")[:2]).strip()
    if short_desc and not short_desc.endswith("."):
        short_desc += "."

    return {
        "name": info.get("longName") or info.get("shortName") or ticker,
        "description": short_desc or None,
        "quote_type": info.get("quoteType"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "exchange": info.get("exchange"),
        "market_cap": info.get("marketCap"),
        "currency": info.get("currency"),
    }


def get_provider(name: str = "yfinance", **kwargs) -> DataProvider:
    """Factory for data providers."""
    providers = {
        "yfinance": YFinanceProvider,
        "ibkr": IBKRProvider,
    }
    if name not in providers:
        raise ValueError(f"Unknown provider: {name}. Available: {list(providers.keys())}")
    return providers[name](**kwargs)


def fetch_event_window(
    provider: DataProvider,
    ticker: str,
    event_date: date,
    days_before: int = 10,
    days_after: int = 10,
) -> pd.DataFrame:
    """
    Fetch price data around an event date with buffer for weekends/holidays.
    Returns exactly the trading days within the window.
    """
    # Add buffer for non-trading days
    buffer = max(days_before, days_after) + 15
    start = event_date - timedelta(days=buffer)
    end = event_date + timedelta(days=buffer)

    df = provider.get_daily_ohlcv(ticker, start, end)

    # Find nearest trading day to event_date
    event_idx = df.index.searchsorted(pd.Timestamp(event_date))
    if event_idx >= len(df):
        event_idx = len(df) - 1

    # The anchor must actually be near the event date. When it isn't — the
    # event is in the future, or predates the ticker's history — the clamped
    # index silently re-anchors the window to the data boundary, producing a
    # window that has nothing to do with the event (and, for future events,
    # a near-duplicate of "today" that scores ~1.0 similarity).
    anchor_date = df.index[event_idx].date()
    if abs((anchor_date - event_date).days) > 7:
        raise ValueError(
            f"No trading data near event date {event_date} for {ticker} "
            f"(nearest available: {anchor_date})"
        )

    # Get window indices
    start_idx = max(0, event_idx - days_before)
    end_idx = min(len(df), event_idx + days_after + 1)

    window = df.iloc[start_idx:end_idx].copy()

    # Add relative day column (0 = event day)
    window["rel_day"] = range(-event_idx + start_idx, -event_idx + start_idx + len(window))

    return window


def normalize_window(df: pd.DataFrame, ref_col: str = "Close") -> pd.DataFrame:
    """
    Normalize prices relative to the event day (rel_day=0) price.
    Returns percentage change from event day price.
    """
    if "rel_day" not in df.columns:
        raise ValueError("DataFrame must have 'rel_day' column")

    event_row = df[df["rel_day"] == 0]
    if event_row.empty:
        # Use closest to 0
        event_row = df.iloc[(df["rel_day"].abs()).argmin(): (df["rel_day"].abs()).argmin() + 1]

    ref_price = event_row[ref_col].values[0]
    if ref_price == 0:
        raise ValueError("Reference price is 0, cannot normalize")

    result = df.copy()
    for col in ["Open", "High", "Low", "Close"]:
        if col in result.columns:
            result[f"{col}_norm"] = ((result[col] / ref_price) - 1) * 100

    return result
