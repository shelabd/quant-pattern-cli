"""Tests for universe parsing, filtering, and loading (universe.py)."""

import json

import numpy as np
import pandas as pd

from quant_patterns.universe import (
    _parse_directory,
    load_universe,
    refresh_universe,
    universe_age_days,
)

NASDAQ_FIXTURE = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
ZAZZT|Tick Pilot Test Stock|G|Y|N|100|N|N
QQQ|Invesco QQQ Trust|G|N|N|100|Y|N
ACHRW|Archer Aviation Inc. - Warrant|Q|N|N|100|N|N
FAKEU|Fake Acquisition Corp - Unit|Q|N|N|100|N|N
GOOGL|Alphabet Inc. - Class A Common Stock|Q|N|N|100|N|N
File Creation Time: 0717202621:30|||||||
"""

OTHER_FIXTURE = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
BRK.B|Berkshire Hathaway Inc. Class B|N|BRK/B|N|100|N|BRK=B
WMT|Walmart Inc. Common Stock|N|WMT|N|100|N|WMT
BAC$K|Bank of America Preferred Series K|N|BAC-K|N|100|N|BAC-K
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
"""


def test_parse_directory_filters_junk():
    rows = _parse_directory(NASDAQ_FIXTURE)
    tickers = {r["ticker"] for r in rows}
    assert tickers == {"AAPL", "GOOGL"}  # no test issue, ETF, warrant, unit


def test_parse_directory_other_listed_format():
    rows = _parse_directory(OTHER_FIXTURE)
    tickers = {r["ticker"] for r in rows}
    # BRK.B (class suffix) and BAC$K (preferred) and SPY (ETF) filtered.
    assert tickers == {"WMT"}


def test_parse_directory_garbage_returns_empty():
    assert _parse_directory("") == []
    assert _parse_directory("not a pipe file") == []


def test_refresh_universe_ranks_by_dollar_volume(tmp_path, monkeypatch):
    import quant_patterns.universe as universe_mod
    monkeypatch.setattr(
        universe_mod, "fetch_symbol_directory",
        lambda timeout=30: [{"ticker": t, "name": t}
                            for t in ("BIG", "SMALL", "TINY", "NODATA")])

    def frame(price, volume, n=30):
        dates = pd.bdate_range("2026-06-01", periods=n)
        return pd.DataFrame({"Open": price, "High": price, "Low": price,
                             "Close": np.full(n, float(price)),
                             "Volume": np.full(n, float(volume))}, index=dates)

    frames = {"BIG": frame(100, 10_000_000),      # $1B/day
              "SMALL": frame(50, 1_000_000),      # $50M/day
              "TINY": frame(10, 100_000)}         # $1M/day — below floor

    path = tmp_path / "universe.json"
    universe = refresh_universe(lambda tickers: frames, size=10,
                                min_dollar_vol=20e6, path=path)
    tickers = [row["ticker"] for row in universe]
    assert tickers == ["BIG", "SMALL"]  # ranked, TINY floored, NODATA dropped

    loaded, note = load_universe(path=path)
    assert loaded == ["BIG", "SMALL"]
    assert "universe.json" in note


def test_load_universe_falls_back_to_bundled(tmp_path):
    tickers, note = load_universe(path=tmp_path / "missing.json")
    assert note == "bundled snapshot"
    assert len(tickers) > 1000
    assert all(t == t.upper() for t in tickers[:50])


def test_load_universe_max_tickers(tmp_path):
    tickers, _ = load_universe(path=tmp_path / "missing.json", max_tickers=25)
    assert len(tickers) == 25


def test_universe_age_days(tmp_path):
    path = tmp_path / "universe.json"
    assert universe_age_days(path) is None
    path.write_text(json.dumps({"as_of": "2026-07-01T21:45:00",
                                "tickers": [{"ticker": "AAPL"}]}))
    age = universe_age_days(path)
    assert age is not None and age >= 0
