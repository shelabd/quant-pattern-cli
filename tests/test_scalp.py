"""Offline tests for intraday scalp levels. No network access."""

from datetime import datetime

import pandas as pd
import pytest

from quant_patterns.scalp import (
    ET,
    LevelCandidate,
    _pick_level,
    compute_scalp_levels,
    format_message,
    intraday_candidates,
    is_market_open,
    oi_wall_candidates,
    remaining_sigma,
    session_minutes_remaining,
)


def et(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=ET)


def bars(prices, volume=1_000_000, start_h=9, start_m=30, day=6):
    """5-minute session bars from a list of closes (July 2026, a Monday=6th)."""
    idx = pd.date_range(et(2026, 7, day, start_h, start_m), periods=len(prices),
                        freq="5min")
    df = pd.DataFrame({
        "Open": prices, "High": [p + 0.2 for p in prices],
        "Low": [p - 0.2 for p in prices], "Close": prices,
        "Volume": [volume] * len(prices),
    }, index=idx)
    return df


def chain_df(strikes, put_oi, call_oi, gamma=None):
    return pd.DataFrame({
        "strike": strikes,
        "put_oi": put_oi, "call_oi": call_oi,
        "put_vol": [0] * len(strikes), "call_vol": [0] * len(strikes),
        "iv": [0.15] * len(strikes),
        "gamma": gamma if gamma is not None else [0.05] * len(strikes),
    })


class TestSessionClock:
    def test_open_hours(self):
        assert is_market_open(et(2026, 7, 6, 9, 30))       # Monday open
        assert is_market_open(et(2026, 7, 6, 15, 59))
        assert not is_market_open(et(2026, 7, 6, 16, 0))   # close is exclusive
        assert not is_market_open(et(2026, 7, 6, 9, 29))
        assert not is_market_open(et(2026, 7, 4, 12, 0))   # Saturday

    def test_minutes_remaining(self):
        assert session_minutes_remaining(et(2026, 7, 6, 15, 30)) == 30
        assert session_minutes_remaining(et(2026, 7, 6, 9, 30)) == 390
        assert session_minutes_remaining(et(2026, 7, 6, 17, 0)) == 0
        # pre-open clamps to the full session, not wall-clock minutes
        assert session_minutes_remaining(et(2026, 7, 6, 4, 45)) == 390

    def test_remaining_sigma_shrinks_into_close(self):
        early = remaining_sigma(750.0, 0.15, 390)
        late = remaining_sigma(750.0, 0.15, 30)
        assert early > late > 0
        assert remaining_sigma(750.0, 0.15, 0) == 0.0


class TestIntradayCandidates:
    def test_vwap_and_session_levels(self):
        today = bars([744, 745, 746, 745, 744, 745, 746, 747])
        cands, vwap = intraday_candidates(today, None)
        srcs = {c.source for c in cands}
        assert {"VWAP", "session low", "session high",
                "opening range low", "opening range high"} <= srcs
        assert vwap == pytest.approx(745.25, abs=0.1)

    def test_opening_range_needs_30_minutes(self):
        today = bars([744, 745, 746])  # only 15 minutes in
        cands, _ = intraday_candidates(today, None)
        srcs = {c.source for c in cands}
        assert "opening range low" not in srcs

    def test_prior_session_levels(self):
        today = bars([745] * 7)
        prior = bars([740, 742, 741], day=2)
        cands, _ = intraday_candidates(today, prior)
        by_src = {c.source: c.price for c in cands}
        assert by_src["prior low"] == pytest.approx(739.8)
        assert by_src["prior high"] == pytest.approx(742.2)
        assert by_src["prior close"] == pytest.approx(741.0)


class TestOiWalls:
    def test_put_and_call_walls(self):
        chain = chain_df([740, 745, 750, 755],
                         put_oi=[20_000, 5_000, 1_000, 500],
                         call_oi=[500, 1_000, 22_000, 4_000])
        cands, magnet, detail = oi_wall_candidates(chain, 746.0)
        by_src = {c.source: c.price for c in cands}
        assert by_src["put wall"] == 740.0
        assert by_src["call wall"] == 750.0
        assert magnet == 750.0  # highest gamma-weighted total OI
        assert "total OI" in detail

    def test_band_excludes_far_strikes(self):
        chain = chain_df([700, 745, 790],
                         put_oi=[900_000, 5_000, 0],
                         call_oi=[0, 5_000, 900_000])
        cands, _, _ = oi_wall_candidates(chain, 746.0)
        prices = {c.price for c in cands}
        assert 700.0 not in prices and 790.0 not in prices

    def test_empty_chain(self):
        assert oi_wall_candidates(None, 746.0) == ([], None, "")


class TestPickLevel:
    def test_confluence_beats_single_source(self):
        # Two weak sources clustered at ~744 outweigh one at 742.
        cands = [
            LevelCandidate(744.0, "session low"),
            LevelCandidate(744.3, "prior close"),
            LevelCandidate(742.0, "prior low"),
        ]
        level, sources = _pick_level(cands, 746.0)
        assert 743.9 <= level <= 744.4
        assert len(sources) == 2

    def test_snaps_to_wall_strike(self):
        cands = [
            LevelCandidate(744.0, "put wall", "18,000 put OI @ 744"),
            LevelCandidate(744.4, "VWAP"),
        ]
        level, sources = _pick_level(cands, 746.0)
        assert level == 744.0  # the strike, not the weighted mean
        assert sources[0].startswith("18,000")

    def test_empty(self):
        assert _pick_level([], 746.0) == (None, [])


class TestComputeScalpLevels:
    def test_end_to_end_offline(self):
        now = et(2026, 7, 6, 11, 0)
        today = bars([745, 746, 745.5, 745, 746, 746.5, 746, 745.5])
        prior = bars([743, 744, 743.5], day=2)
        chain = chain_df([740, 744, 750, 752],
                         put_oi=[8_000, 15_000, 1_000, 0],
                         call_oi=[0, 1_000, 18_000, 3_000])
        lv = compute_scalp_levels("SPY", 745.5, now, today, prior, chain,
                                  iv=0.15, chain_expiry="2026-07-06")
        assert lv.floor is not None and lv.floor < 745.5
        assert lv.ceiling is not None and lv.ceiling > 745.5
        assert lv.sigma_remaining and lv.sigma_remaining > 0
        assert lv.minutes_left == 300
        d = lv.to_dict()
        assert d["ticker"] == "SPY" and d["floor"] is not None

    def test_no_chain_still_produces_levels(self):
        now = et(2026, 7, 6, 11, 0)
        today = bars([745, 746, 745.5, 745, 746, 746.5, 746, 745.5])
        lv = compute_scalp_levels("SPY", 745.5, now, today, None, None, iv=None)
        assert lv.floor is not None and lv.ceiling is not None
        assert lv.magnet is None

    def test_message_format(self):
        now = et(2026, 7, 6, 11, 0)
        today = bars([745, 746, 745.5, 745, 746, 746.5, 746, 745.5])
        lv = compute_scalp_levels("SPY", 745.5, now, today, None, None, iv=0.15)
        msg = format_message(lv)
        assert "SPY scalp" in msg and "Floor" in msg and "Ceiling" in msg
        assert "not financial advice" in msg
