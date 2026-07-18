"""Tests for the bulk-fetch disk cache (screener_data.py) — offline via a
monkeypatched yf.download."""

from datetime import date

import numpy as np
import pandas as pd

import quant_patterns.screener_data as sd


def make_frame(start: str, periods: int, price: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    closes = np.full(periods, price)
    df = pd.DataFrame({"Open": closes, "High": closes * 1.01,
                       "Low": closes * 0.99, "Close": closes,
                       "Volume": np.full(periods, 1e6)}, index=dates)
    df.index.name = "Date"
    return df


def fake_batch(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble a group_by='ticker' style multi-column frame."""
    return pd.concat(frames, axis=1)


def test_cache_round_trip(tmp_path):
    path = tmp_path / "ohlcv.pkl"
    frames = {"AAPL": make_frame("2026-01-02", 50)}
    sd.save_ohlcv_cache(frames, path)
    loaded = sd.load_ohlcv_cache(path)
    assert list(loaded) == ["AAPL"]
    pd.testing.assert_frame_equal(loaded["AAPL"], frames["AAPL"])


def test_cache_corrupt_file_returns_empty(tmp_path):
    path = tmp_path / "ohlcv.pkl"
    path.write_bytes(b"not a pickle")
    assert sd.load_ohlcv_cache(path) == {}


def test_cache_version_mismatch_returns_empty(tmp_path, monkeypatch):
    path = tmp_path / "ohlcv.pkl"
    sd.save_ohlcv_cache({"AAPL": make_frame("2026-01-02", 5)}, path)
    monkeypatch.setattr(sd, "CACHE_VERSION", 2)
    assert sd.load_ohlcv_cache(path) == {}


def test_cold_fetch_populates_cache(tmp_path, monkeypatch):
    path = tmp_path / "ohlcv.pkl"
    calls = []

    def fake_download(tickers, **kwargs):
        symbols = tickers.split()
        calls.append(symbols)
        return fake_batch({t: make_frame("2026-01-02", 120) for t in symbols})

    monkeypatch.setattr(sd.yf, "download", fake_download)
    frames, warnings = sd.fetch_universe_ohlcv(
        ["AAPL", "MSFT"], cache_path=path, today=date(2026, 6, 19))
    assert set(frames) == {"AAPL", "MSFT"}
    assert calls and calls[0] == ["AAPL", "MSFT"]
    assert warnings == []
    # Cache persisted for the next run.
    assert set(sd.load_ohlcv_cache(path)) == {"AAPL", "MSFT"}


def test_warm_fetch_appends_tail_only(tmp_path, monkeypatch):
    path = tmp_path / "ohlcv.pkl"
    today = date(2026, 6, 19)  # a Friday
    # Cache reaches Wednesday 2026-06-17; expected last session is Friday.
    cached = make_frame("2025-06-02", 273)
    assert cached.index[-1].date() == date(2026, 6, 17)
    sd.save_ohlcv_cache({"AAPL": cached}, path)

    fetch_starts = []

    def fake_download(tickers, start=None, **kwargs):
        fetch_starts.append(start)
        return fake_batch({t: make_frame("2026-06-18", 2, price=111.0)
                           for t in tickers.split()})

    monkeypatch.setattr(sd.yf, "download", fake_download)
    frames, _ = sd.fetch_universe_ohlcv(["AAPL"], cache_path=path, today=today)
    # Tail fetch started from the day after the cached end, not the full window.
    assert fetch_starts == ["2026-06-18"]
    df = frames["AAPL"]
    assert df.index[-1].date() == date(2026, 6, 19)
    assert float(df["Close"].iloc[-1]) == 111.0
    # No duplicate index rows after the append.
    assert not df.index.duplicated().any()


def test_fresh_cache_skips_network(tmp_path, monkeypatch):
    path = tmp_path / "ohlcv.pkl"
    today = date(2026, 6, 21)  # a Sunday -> last expected session Friday 19th
    cached = make_frame("2025-06-02", 275)
    assert cached.index[-1].date() == date(2026, 6, 19)
    sd.save_ohlcv_cache({"AAPL": cached}, path)

    def boom(*a, **k):
        raise AssertionError("network hit on a fresh cache")

    monkeypatch.setattr(sd.yf, "download", boom)
    frames, warnings = sd.fetch_universe_ohlcv(["AAPL"], cache_path=path,
                                               today=today)
    assert set(frames) == {"AAPL"}
    assert warnings == []


def test_dead_tickers_dropped_with_warning(tmp_path, monkeypatch):
    path = tmp_path / "ohlcv.pkl"

    def fake_download(tickers, **kwargs):
        return fake_batch({"AAPL": make_frame("2026-01-02", 120)})

    monkeypatch.setattr(sd.yf, "download", fake_download)
    frames, warnings = sd.fetch_universe_ohlcv(
        ["AAPL", "DEADX"], cache_path=path, today=date(2026, 6, 19))
    assert set(frames) == {"AAPL"}
    assert warnings and "1/2" in warnings[0]


def test_force_ignores_cache(tmp_path, monkeypatch):
    path = tmp_path / "ohlcv.pkl"
    sd.save_ohlcv_cache({"AAPL": make_frame("2025-06-02", 275, price=1.0)}, path)

    def fake_download(tickers, **kwargs):
        return fake_batch({t: make_frame("2026-01-02", 120, price=222.0)
                           for t in tickers.split()})

    monkeypatch.setattr(sd.yf, "download", fake_download)
    frames, _ = sd.fetch_universe_ohlcv(["AAPL"], cache_path=path,
                                        force=True, today=date(2026, 6, 19))
    assert float(frames["AAPL"]["Close"].iloc[-1]) == 222.0
