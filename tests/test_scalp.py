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
    check_level_hits,
    filter_new_hits,
    format_alert,
    oi_wall_candidates,
    relative_volume,
    remaining_sigma,
    rvol_regime,
    scalp_setups,
    session_minutes_remaining,
    volume_profile_candidates,
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


class TestVolumeProfile:
    def test_poc_at_heaviest_price(self):
        today = bars([744.0] * 3 + [747.0] * 2 + [750.0] * 5)
        today["Volume"] = [1_000_000] * 5 + [5_000_000] * 5
        cands = volume_profile_candidates(today, 750.0)
        poc = next(c for c in cands if c.source == "volume POC")
        assert abs(poc.price - 750.0) < 1.2
        assert "of session" in poc.detail

    def test_secondary_node_survives(self):
        # Heavy acceptance at 750 (POC) and a distinct node at 744.
        today = bars([744.0] * 4 + [747.0] * 2 + [750.0] * 4)
        today["Volume"] = [3_000_000] * 4 + [500_000] * 2 + [4_000_000] * 4
        cands = volume_profile_candidates(today, 750.0)
        nodes = [c for c in cands if c.source == "volume node"]
        assert nodes and abs(nodes[0].price - 744.0) < 1.2

    def test_too_little_dispersion(self):
        today = bars([745.0] * 8)  # everything in one bin
        assert volume_profile_candidates(today, 745.0) == []

    def test_zero_volume(self):
        today = bars([744.0, 747.0, 750.0], volume=0)
        assert volume_profile_candidates(today, 750.0) == []


class TestRvol:
    def test_same_elapsed_bars(self):
        today = bars([745.0] * 4)
        today["Volume"] = [2_000_000] * 4
        prior = bars([744.0] * 8, day=2)  # full prior session, 1M/bar
        # baseline = first 4 prior bars (same elapsed), not the full day
        assert relative_volume(today, [prior]) == pytest.approx(2.0)

    def test_averages_multiple_sessions(self):
        today = bars([745.0] * 4)  # 1M/bar -> 4M
        p1 = bars([744.0] * 8, day=1)
        p2 = bars([744.0] * 8, day=2)
        p2["Volume"] = [3_000_000] * 8
        # baseline = mean(4M, 12M) = 8M -> rvol 0.5
        assert relative_volume(today, [p1, p2]) == pytest.approx(0.5)

    def test_no_baseline(self):
        assert relative_volume(bars([745.0] * 4), []) is None

    def test_regime_bands(self):
        assert "trend-day" in rvol_regime(1.8)
        assert "quiet" in rvol_regime(0.5)
        assert rvol_regime(1.0) == ""
        assert rvol_regime(None) == ""


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
        today = bars([748, 748.5, 749, 749.2, 749.5, 749.8,
                      750.5, 751, 751.1, 751.05])
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

    def test_trend_day_rvol_flags_counter_trend_fade(self):
        now = et(2026, 7, 6, 12, 0)
        today = bars([748, 749, 750, 750.5, 751, 751.1, 751.0])
        today["Volume"] = [3_000_000] * len(today)  # 3x the prior pace
        priors = [bars([747.0] * 78, day=d) for d in (1, 2)]
        lv = compute_scalp_levels("SPY", 751.0, now, today, priors[-1], None,
                                  iv=0.10, prior_sessions=priors)
        assert lv.rvol == pytest.approx(3.0)
        assert lv.rvol_sessions == 2
        assert any("trend-day" in w for w in lv.warnings)
        counter = [s for s in lv.setups if not s.with_trend]
        assert counter and any("get-run-over" in n for n in counter[0].notes)
        assert lv.to_dict()["rvol"] == 3.0

    def test_rvol_falls_back_to_single_prior(self):
        now = et(2026, 7, 6, 12, 0)
        today = bars([748, 749, 750, 750.5])
        prior = bars([747.0] * 78, day=2)
        lv = compute_scalp_levels("SPY", 750.5, now, today, prior, None, iv=None)
        assert lv.rvol == pytest.approx(1.0)
        assert lv.rvol_sessions == 1


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


def watch_snapshot():
    """A journaled snapshot as scalp-watch reads it (post-to_dict JSON)."""
    lv = snapshot()  # floor 740, ceiling 752, spot 745
    lv.setups = scalp_setups(lv)
    return lv.to_dict()


class TestLevelWatch:
    def test_no_hit_mid_range(self):
        assert check_level_hits(watch_snapshot(), 746.0) == []

    def test_floor_touch_inside_entry_zone(self):
        snap = watch_snapshot()
        zone_hi = next(s for s in snap["setups"] if s["side"] == "long")["entry_zone"][1]
        hits = check_level_hits(snap, zone_hi - 0.01)
        assert [(h["side"], h["kind"]) for h in hits] == [("floor", "touch")]
        assert hits[0]["setup"]["side"] == "long"

    def test_floor_break_below_stop(self):
        snap = watch_snapshot()
        stop = next(s for s in snap["setups"] if s["side"] == "long")["stop"]
        hits = check_level_hits(snap, stop - 0.05)
        assert [(h["side"], h["kind"]) for h in hits] == [("floor", "break")]

    def test_ceiling_touch_and_break(self):
        snap = watch_snapshot()
        short = next(s for s in snap["setups"] if s["side"] == "short")
        assert check_level_hits(snap, short["entry_zone"][0] + 0.01)[0]["kind"] == "touch"
        assert check_level_hits(snap, short["stop"] + 0.05)[0]["kind"] == "break"

    def test_falls_back_to_levels_without_setups(self):
        snap = {"floor": 740.0, "ceiling": 752.0, "setups": []}
        assert check_level_hits(snap, 740.1)[0]["kind"] == "touch"
        assert check_level_hits(snap, 738.0)[0]["kind"] == "break"

    def test_dedup_once_per_day(self):
        hits = check_level_hits(watch_snapshot(), 740.0)
        fresh, state = filter_new_hits(hits, {}, "2026-07-07")
        assert len(fresh) == 1
        again, state = filter_new_hits(hits, state, "2026-07-07")
        assert again == []
        # a break at the same level is a different alert
        breaks = check_level_hits(watch_snapshot(), 738.0)
        fresh2, state = filter_new_hits(breaks, state, "2026-07-07")
        assert len(fresh2) == 1

    def test_dedup_resets_next_day(self):
        hits = check_level_hits(watch_snapshot(), 740.0)
        _, state = filter_new_hits(hits, {}, "2026-07-07")
        fresh, _ = filter_new_hits(hits, state, "2026-07-08")
        assert len(fresh) == 1

    def test_moved_level_rearms(self):
        hits = check_level_hits(watch_snapshot(), 740.0)
        _, state = filter_new_hits(hits, {}, "2026-07-07")
        moved = [dict(hits[0], level=741.5)]
        fresh, _ = filter_new_hits(moved, state, "2026-07-07")
        assert len(fresh) == 1

    def test_alert_text(self):
        snap = watch_snapshot()
        touch = check_level_hits(snap, 740.0)[0]
        text = format_alert("SPY", 740.0, touch, asof="15:00 ET")
        assert "🔔" in text and "FLOOR 740.00" in text
        assert "stop" in text and "15:00 ET" in text
        brk = check_level_hits(snap, 738.0)[0]
        text = format_alert("SPY", 738.0, brk)
        assert "BROKEN" in text and "invalidated" in text
