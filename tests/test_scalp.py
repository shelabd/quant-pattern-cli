"""Offline tests for intraday scalp levels. No network access."""

from datetime import datetime

import pandas as pd
import pytest

from quant_patterns.scalp import (
    ET,
    MIN_RR,
    LevelCandidate,
    ScalpLevels,
    _pick_level,
    compute_scalp_levels,
    format_message,
    intraday_candidates,
    is_market_open,
    oi_wall_candidates,
    remaining_sigma,
    scalp_setups,
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
        level, sources, near, near_srcs = _pick_level(cands, 746.0)
        assert 743.9 <= level <= 744.4
        assert len(sources) == 2
        # the lone 742 candidate is beyond the level, not between: no near
        assert near is None

    def test_snaps_to_wall_strike(self):
        cands = [
            LevelCandidate(744.0, "put wall", "18,000 put OI @ 744"),
            LevelCandidate(744.4, "VWAP"),
        ]
        level, sources, _, _ = _pick_level(cands, 746.0)
        assert level == 744.0  # the strike, not the weighted mean
        assert sources[0].startswith("18,000")

    def test_near_level_between_spot_and_main(self):
        # Wall wins at 755; session high at 751.1 sits between spot and it.
        cands = [
            LevelCandidate(755.0, "call wall", "12,000 call OI @ 755"),
            LevelCandidate(754.5, "expected move"),
            LevelCandidate(751.1, "session high"),
        ]
        level, _, near, near_srcs = _pick_level(cands, 751.0)
        assert level == 755.0
        assert near == pytest.approx(751.1)
        assert near_srcs == ["session high"]

    def test_empty(self):
        assert _pick_level([], 746.0) == (None, [], None, [])


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

    def test_near_ceiling_end_to_end(self):
        # Spot pinned at the session high with a far call wall: the wall is
        # the ceiling, the session high surfaces as the near ceiling.
        now = et(2026, 7, 6, 12, 0)
        today = bars([748, 749, 750, 750.5, 751, 751.1, 751.0])
        chain = chain_df([745, 750, 755],
                         put_oi=[15_000, 5_000, 0],
                         call_oi=[0, 2_000, 12_000])
        lv = compute_scalp_levels("SPY", 751.0, now, today, None, chain, iv=0.10)
        assert lv.ceiling == 755.0
        assert lv.near_ceiling is not None and 751.0 < lv.near_ceiling < 755.0
        msg = format_message(lv)
        assert "near ceiling" in msg and "break = room to 755.00" in msg
        d = lv.to_dict()
        assert d["near_ceiling"] == round(lv.near_ceiling, 2)


def snapshot(spot=745.0, floor=740.0, ceiling=752.0, vwap=744.0,
             magnet=None, sigma=4.0):
    return ScalpLevels(
        ticker="SPY", asof=et(2026, 7, 6, 11, 0), spot=spot,
        floor=floor, ceiling=ceiling, vwap=vwap, magnet=magnet,
        sigma_remaining=sigma, minutes_left=300,
    )


class TestScalpSetups:
    def test_two_sided_plan(self):
        setups = scalp_setups(snapshot())
        assert {s.side for s in setups} == {"long", "short"}
        long = next(s for s in setups if s.side == "long")
        short = next(s for s in setups if s.side == "short")
        # geometry: stop beyond the level, targets inside the range
        assert long.stop < long.entry_lo <= 740.0 <= long.entry_hi
        assert long.target1 < long.target2 < 752.0
        assert short.stop > short.entry_hi >= 752.0 >= short.entry_lo
        assert short.target1 > short.target2 > 740.0

    def test_nearest_trigger_first(self):
        setups = scalp_setups(snapshot(spot=750.0))  # closer to the ceiling
        assert setups[0].side == "short"

    def test_rr_math(self):
        long = next(s for s in scalp_setups(snapshot()) if s.side == "long")
        risk = long.trigger - long.stop
        assert long.rr1 == pytest.approx((long.target1 - long.trigger) / risk)
        assert long.rr2 == pytest.approx((long.target2 - long.trigger) / risk)
        assert long.rr2 >= MIN_RR and not long.skip_reason

    def test_trend_bias_from_vwap(self):
        # spot above VWAP: long is with-trend, short counter-trend
        setups = scalp_setups(snapshot(spot=745.0, vwap=744.0))
        assert next(s for s in setups if s.side == "long").with_trend
        short = next(s for s in setups if s.side == "short")
        assert not short.with_trend
        assert any("counter-trend" in n for n in short.notes)

    def test_vwap_is_t1_when_inside_range(self):
        long = next(s for s in scalp_setups(snapshot(floor=740.0, vwap=744.0))
                    if s.side == "long")
        assert long.target1_label == "VWAP" and long.target1 == 744.0

    def test_midrange_t1_without_vwap(self):
        long = next(s for s in scalp_setups(snapshot(vwap=None))
                    if s.side == "long")
        assert long.target1_label == "mid-range"

    def test_tight_range_is_skipped(self):
        setups = scalp_setups(snapshot(floor=744.5, ceiling=745.5))
        assert setups and all(s.skip_reason for s in setups)

    def test_target_beyond_sigma_noted(self):
        long = next(s for s in scalp_setups(snapshot(sigma=2.0))
                    if s.side == "long")  # ceiling is 12 pts away, 1σ only 2
        assert any("1σ" in n for n in long.notes)

    def test_missing_level_means_no_setups(self):
        assert scalp_setups(snapshot(ceiling=None)) == []

    def test_setups_in_message_and_dict(self):
        lv = snapshot()
        lv.setups = scalp_setups(lv)
        msg = format_message(lv)
        assert "LONG" in msg and "SHORT" in msg and "stop" in msg
        d = lv.to_dict()
        assert len(d["setups"]) == 2
        assert {"side", "entry_zone", "stop", "target1", "rr2"} <= set(d["setups"][0])
