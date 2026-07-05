"""Offline tests for the pin butterfly engine. No network access."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from quant_patterns.butterfly import (
    DEFAULT_IV_FALLBACK,
    DEFAULT_TARGET_POP,
    EVENT_VOL_ADDON,
    UnpriceableStrikeError,
    adaptive_width,
    atm_iv,
    band_bounds,
    bs_gamma,
    candidate_widths,
    choose_expiry,
    detect_drift,
    evaluate_ratio,
    event_vol_addon,
    event_warnings,
    expected_move,
    floor_to_cent,
    fly_expected_value,
    max_debit_for,
    mid_price,
    normalize_chain,
    prob_in_profit,
    price_fly,
    score_pins,
    select_pin,
    select_width_by_pop,
)

EXPIRY = date(2026, 6, 17)
TODAY = date(2026, 6, 12)


def make_chain(strikes, oi=None, mid_fn=None, iv=0.2, volume=None):
    """Synthetic normalized chain. mid_fn maps strike -> per-share option mid;
    bid/ask straddle it by a cent."""
    n = len(strikes)
    oi = oi if oi is not None else [1000] * n
    volume = volume if volume is not None else [0] * n
    mid_fn = mid_fn or (lambda k: 1.0)
    mids = [mid_fn(k) for k in strikes]
    return pd.DataFrame({
        "strike": [float(k) for k in strikes],
        "call_oi": oi, "put_oi": oi,
        "call_vol": volume, "put_vol": volume,
        "call_bid": [m - 0.01 for m in mids],
        "call_ask": [m + 0.01 for m in mids],
        "call_last": mids,
        "put_bid": [m - 0.01 for m in mids],
        "put_ask": [m + 0.01 for m in mids],
        "put_last": mids,
        "iv": [iv] * n,
    })


# ── Pin detection ─────────────────────────────────────────────────────────────


class TestPinDetection:
    def test_picks_max_oi_strike(self):
        strikes = [578, 579, 580, 581, 582]
        oi = [100, 200, 5000, 300, 150]
        chain = make_chain(strikes, oi=oi)
        pin = select_pin(chain, spot=578.0, dte=3, band_pct=1.5, drift="bullish")
        assert pin is not None
        assert pin["strike"] == 580.0
        assert pin["total_oi"] == 10000  # calls + puts

    def test_band_direction_bullish_excludes_below_spot(self):
        lo, hi = band_bounds(580.0, 1.5, "bullish")
        assert lo == 580.0
        assert hi == pytest.approx(588.7)

        # Huge OI below spot must not be selected in a bullish band
        chain = make_chain([575, 582], oi=[50000, 100])
        pin = select_pin(chain, spot=580.0, dte=3, band_pct=1.5, drift="bullish")
        assert pin["strike"] == 582.0

    def test_band_direction_bearish(self):
        lo, hi = band_bounds(580.0, 1.5, "bearish")
        assert hi == 580.0
        assert lo == pytest.approx(571.3)

    def test_band_neutral_symmetric(self):
        lo, hi = band_bounds(580.0, 1.5, "neutral")
        assert lo == pytest.approx(571.3)
        assert hi == pytest.approx(588.7)

    def test_empty_band_returns_none(self):
        chain = make_chain([400, 401])
        assert select_pin(chain, spot=580.0, dte=3, drift="bullish") is None

    def test_round_number_bonus_breaks_near_ties(self):
        # 583 and 585 nearly tied on raw OI; the +10% multiple-of-5 bonus and
        # comparable gamma must tip the score to 585
        chain = make_chain([583, 585], oi=[1000, 980])
        scored = score_pins(chain, spot=582.0, dte=3, band_pct=1.5, drift="bullish")
        assert scored.iloc[0]["strike"] == 585.0

    def test_volume_fallback_when_band_oi_zero(self):
        chain = make_chain([580, 582], oi=[0, 0], volume=[500, 9000])
        pin = select_pin(chain, spot=579.0, dte=3, band_pct=1.5, drift="bullish")
        assert pin["used_volume"] is True
        assert pin["strike"] == 582.0
        assert pin["concentration"] == 18000


# ── Drift detection ───────────────────────────────────────────────────────────


class TestDriftDetection:
    def _series(self, values):
        return pd.Series(values, dtype=float)

    def test_uptrend_bullish(self):
        close = self._series(np.linspace(100, 120, 40))
        assert detect_drift(close) == "bullish"

    def test_downtrend_bearish(self):
        close = self._series(np.linspace(120, 100, 40))
        assert detect_drift(close) == "bearish"

    def test_flat_neutral(self):
        close = self._series([100.0] * 40)
        assert detect_drift(close) == "neutral"

    def test_disagreement_is_neutral(self):
        # Long uptrend (EMA5 > EMA20) but sharp 3-session pullback
        values = list(np.linspace(100, 130, 37)) + [126, 124, 122]
        assert detect_drift(self._series(values)) == "neutral"

    def test_short_series_neutral(self):
        assert detect_drift(self._series([100, 101, 102])) == "neutral"


# ── Pricing & ratio arithmetic ────────────────────────────────────────────────


class TestPricing:
    def test_mid_from_bid_ask(self):
        assert mid_price(1.00, 1.10, 5.0) == pytest.approx(1.05)

    def test_mid_falls_back_to_last(self):
        assert mid_price(0.0, 0.0, 0.85) == 0.85

    def test_unpriceable_raises(self):
        with pytest.raises(UnpriceableStrikeError):
            mid_price(0.0, 0.0, 0.0)

    def test_price_fly_debit_and_legs(self):
        # mids: wings 1.00 and 0.20, body 0.50 -> debit = 1.0 - 2*0.5 + 0.2 = 0.20
        mids = {577.0: 1.00, 580.0: 0.50, 583.0: 0.20}
        chain = make_chain(list(mids), mid_fn=lambda k: mids[k])
        debit, legs = price_fly(chain, body=580.0, width=3.0, right="PUT", expiry=EXPIRY)
        assert debit == pytest.approx(0.20)
        assert [(leg.action, leg.quantity, leg.strike) for leg in legs] == [
            ("BUY", 1, 577.0), ("SELL", 2, 580.0), ("BUY", 1, 583.0)]
        assert all(leg.right == "PUT" for leg in legs)

    def test_missing_wing_raises(self):
        chain = make_chain([580, 583])  # no 577
        with pytest.raises(UnpriceableStrikeError):
            price_fly(chain, body=580.0, width=3.0, right="CALL", expiry=EXPIRY)

    def test_ratio_arithmetic_spec_case(self):
        # debit 0.23, width 3 -> max profit 2.77, RR ~ 12.0, BE body -/+ 2.77
        r = evaluate_ratio(0.23, 3.0, body=580.0)
        assert r["max_profit"] == pytest.approx(2.77)
        assert r["risk_reward"] == pytest.approx(12.04, abs=0.05)
        assert r["breakeven_low"] == pytest.approx(580.0 - 2.77)
        assert r["breakeven_high"] == pytest.approx(580.0 + 2.77)

    def test_debit_ceiling(self):
        assert max_debit_for(3.0, 12.0) == pytest.approx(3.0 / 13.0)
        assert max_debit_for(5.0, 15.0) == pytest.approx(0.3125)

    def test_floor_to_cent_never_exceeds(self):
        ceiling = max_debit_for(3.0, 12.0)  # 0.2307...
        assert floor_to_cent(ceiling) == 0.23
        assert floor_to_cent(ceiling) <= ceiling

    def test_gamma_peaks_at_the_money(self):
        atm = bs_gamma(580, 580, 0.2, 3 / 365)
        otm = bs_gamma(580, 588, 0.2, 3 / 365)
        assert atm > otm > 0


# ── Adaptive width ────────────────────────────────────────────────────────────


def _convex_chain(body=580.0, body_mid=0.50, slope_5=0.40, slope_3=0.23, slope_2=0.10):
    """Chain where the fly debit per width is controlled directly:
    debit(w) = mid(body-w) - 2*mid(body) + mid(body+w)."""
    mids = {
        body - 5: body_mid + slope_5, body + 5: body_mid,
        body - 3: body_mid + slope_3, body + 3: body_mid,
        body - 2: body_mid + slope_2, body + 2: body_mid,
        body: body_mid,
    }
    return make_chain(sorted(mids), mid_fn=lambda k: mids[k])


class TestAdaptiveWidth:
    def test_width_5_fails_width_3_passes(self):
        # ceilings: w5 -> 0.385, w3 -> 0.231, w2 -> 0.154
        chain = _convex_chain(slope_5=0.50, slope_3=0.20, slope_2=0.20)
        result = adaptive_width(chain, 580.0, "CALL", EXPIRY, min_rr=12.0)
        assert result["selected_width"] == 3.0
        assert result["debit"] == pytest.approx(0.20)
        assert result["adaptive"] is True
        assert "selected" in result["attempts"][-1]["result"]
        assert result["attempts"][0]["width"] == 5.0
        assert ">" in result["attempts"][0]["result"]  # failed ceiling

    def test_no_trade_when_no_width_passes(self):
        chain = _convex_chain(slope_5=2.0, slope_3=1.5, slope_2=1.0)
        result = adaptive_width(chain, 580.0, "CALL", EXPIRY, min_rr=12.0)
        assert result["selected_width"] is None
        assert result["legs"] == []
        assert len(result["attempts"]) == 3

    def test_fixed_width_disables_ladder(self):
        chain = _convex_chain(slope_5=2.0, slope_3=0.20, slope_2=0.10)
        result = adaptive_width(chain, 580.0, "CALL", EXPIRY,
                                min_rr=12.0, fixed_width=5.0)
        assert result["selected_width"] is None
        assert result["adaptive"] is False
        assert len(result["attempts"]) == 1  # only width 5 tried


# ── Event warnings ────────────────────────────────────────────────────────────


class _Evt:
    def __init__(self, name, d):
        self.name = name
        self.date = d


class TestEventWarnings:
    def test_event_inside_window_fires(self):
        events = [_Evt("CPI Release", TODAY + timedelta(days=2))]
        warnings, half = event_warnings(events, EXPIRY + timedelta(days=30), TODAY, EXPIRY)
        assert half is True
        assert any("half size" in w for w in warnings)

    def test_event_outside_window_silent(self):
        events = []  # caller pre-filters to the window; nothing inside
        warnings, half = event_warnings(events, EXPIRY + timedelta(days=30), TODAY, EXPIRY)
        assert half is False
        assert warnings == []

    def test_incomplete_calendar_notice(self):
        warnings, half = event_warnings([], EXPIRY - timedelta(days=10), TODAY, EXPIRY)
        assert half is False
        assert any("incomplete" in w for w in warnings)

    def test_no_coverage_at_all(self):
        warnings, _ = event_warnings([], None, TODAY, EXPIRY)
        assert any("incomplete" in w for w in warnings)


# ── OI-aware expiry selection ─────────────────────────────────────────────────


class TestExpirySelection:
    def test_prefers_high_oi_short_expiry(self):
        heavy_short = make_chain([580, 585], oi=[11000, 8000])   # 22K wall
        thin_long = make_chain([580, 585], oi=[1000, 900])       # 2K
        candidates = [
            (TODAY + timedelta(days=2), heavy_short),
            (TODAY + timedelta(days=5), thin_long),
        ]
        best = choose_expiry(candidates, spot=579.0, today=TODAY,
                             band_pct=1.5, drift="bullish")
        assert best["dte"] == 2
        assert best["pin"]["total_oi"] == 22000

    def test_high_oi_long_expiry_beats_thin_short(self):
        thin_short = make_chain([580], oi=[500])
        heavy_long = make_chain([580], oi=[20000])
        candidates = [
            (TODAY + timedelta(days=2), thin_short),
            (TODAY + timedelta(days=5), heavy_long),
        ]
        best = choose_expiry(candidates, spot=579.0, today=TODAY,
                             band_pct=1.5, drift="bullish")
        assert best["dte"] == 5

    def test_tie_breaks_toward_shorter_dte(self):
        chain_a = make_chain([580], oi=[5000])
        chain_b = make_chain([580], oi=[5000])
        candidates = [
            (TODAY + timedelta(days=5), chain_b),
            (TODAY + timedelta(days=2), chain_a),
        ]
        best = choose_expiry(candidates, spot=579.0, today=TODAY,
                             band_pct=1.5, drift="bullish")
        assert best["dte"] == 2

    def test_no_candidates_in_band(self):
        chain = make_chain([400])
        best = choose_expiry([(TODAY + timedelta(days=3), chain)],
                             spot=580.0, today=TODAY, drift="bullish")
        assert best is None


# ── Chain normalization ───────────────────────────────────────────────────────


class TestNormalizeChain:
    def _raw(self, strikes, oi, vol=None):
        n = len(strikes)
        return pd.DataFrame({
            "strike": strikes,
            "openInterest": oi,
            "volume": vol or [0] * n,
            "bid": [1.0] * n,
            "ask": [1.1] * n,
            "lastPrice": [1.05] * n,
            "impliedVolatility": [0.22] * n,
        })

    def test_merges_calls_and_puts(self):
        calls = self._raw([580, 585], [100, 200])
        puts = self._raw([580, 590], [300, 400])
        chain = normalize_chain(calls, puts)
        assert list(chain["strike"]) == [580.0, 585.0, 590.0]
        row580 = chain[chain.strike == 580].iloc[0]
        assert row580["call_oi"] == 100 and row580["put_oi"] == 300
        # one-sided strikes filled with zero, not NaN
        assert chain[chain.strike == 585].iloc[0]["put_oi"] == 0

    def test_nan_oi_becomes_zero(self):
        calls = self._raw([580], [np.nan])
        puts = self._raw([580], [50])
        chain = normalize_chain(calls, puts)
        assert chain.iloc[0]["call_oi"] == 0

    def test_iv_fallback_when_missing(self):
        calls = self._raw([580], [10])
        puts = self._raw([580], [10])
        calls["impliedVolatility"] = 0.0
        puts["impliedVolatility"] = np.nan
        chain = normalize_chain(calls, puts)
        assert chain.iloc[0]["iv"] > 0  # DEFAULT_IV_FALLBACK


# ── Expected-move / macro-uncertainty model ───────────────────────────────────


class _MacroEvt:
    """Minimal MarketEvent stand-in with the .category.value the addon reads."""
    def __init__(self, name, d, cat):
        self.name = name
        self.date = d
        self.category = type("C", (), {"value": cat})()


class TestAtmIv:
    def test_interpolates_between_strikes(self):
        chain = make_chain([580, 590], iv=0.2)
        chain.loc[chain.strike == 580, "iv"] = 0.18
        chain.loc[chain.strike == 590, "iv"] = 0.28
        # spot halfway -> iv halfway
        assert atm_iv(chain, 585.0) == pytest.approx(0.23, abs=1e-9)

    def test_nearest_when_spot_outside_range(self):
        chain = make_chain([580, 590], iv=0.2)
        assert atm_iv(chain, 999.0) == pytest.approx(0.2)

    def test_fallback_when_no_usable_iv(self):
        chain = make_chain([580], iv=0.0)  # zeroed -> filtered out
        assert atm_iv(chain, 580.0) == DEFAULT_IV_FALLBACK


class TestEventVolAddon:
    def test_future_event_adds_in_quadrature(self):
        evts = [_MacroEvt("FOMC", TODAY + timedelta(days=1), "fomc"),
                _MacroEvt("CPI", TODAY + timedelta(days=2), "cpi")]
        pct, breakdown = event_vol_addon(evts, TODAY, EXPIRY)
        expected = (EVENT_VOL_ADDON["fomc"] ** 2 + EVENT_VOL_ADDON["cpi"] ** 2) ** 0.5
        assert pct == pytest.approx(expected)
        assert len(breakdown) == 2

    def test_event_today_is_excluded(self):
        # an event dated today already printed — no forward uncertainty
        evts = [_MacroEvt("FOMC", TODAY, "fomc")]
        pct, breakdown = event_vol_addon(evts, TODAY, EXPIRY)
        assert pct == 0.0
        assert breakdown == []

    def test_unknown_category_ignored(self):
        evts = [_MacroEvt("Mystery", TODAY + timedelta(days=1), "gdp")]
        pct, breakdown = event_vol_addon(evts, TODAY, EXPIRY)
        assert pct == 0.0


class TestExpectedMove:
    def test_diffusion_only(self):
        em = expected_move(750.0, 0.20, dte=4, event_pct=0.0)
        manual = 750.0 * 0.20 * (4 / 365.0) ** 0.5
        assert em["total"] == pytest.approx(manual)
        assert em["pct"] == pytest.approx(manual / 750.0)
        assert em["event"] == 0.0

    def test_event_priced_in_iv_does_not_add(self):
        # A healthy ATM IV already carries the event's variance — the macro
        # component is a floor, so it must NOT widen the band on top of IV.
        base = expected_move(750.0, 0.20, dte=4, event_pct=0.0)["total"]
        same = expected_move(750.0, 0.20, dte=4, event_pct=0.011)["total"]
        assert same == pytest.approx(base)

    def test_event_floor_binds_when_iv_understates(self):
        # Near-zero IV (stale/missing chain vol): the macro floor takes over.
        em = expected_move(750.0, 0.01, dte=1, event_pct=0.011)
        assert em["total"] == pytest.approx(750.0 * 0.011)
        assert em["total"] > em["diffusion"]


class TestProbInProfit:
    def test_symmetric_band_centered(self):
        # breakevens symmetric about center -> POP = P(|Z| < be/sigma)
        pop = prob_in_profit(740.0, 760.0, sigma=10.0, center=750.0)
        assert pop == pytest.approx(0.6827, abs=2e-3)  # ~1 sigma each side

    def test_degenerate_sigma(self):
        assert prob_in_profit(740.0, 760.0, sigma=0.0, center=750.0) == 1.0
        assert prob_in_profit(740.0, 760.0, sigma=0.0, center=800.0) == 0.0


class TestFlyExpectedValue:
    def test_zero_sigma_is_point_payoff(self):
        # settle pinned at body -> max profit = width - debit
        ev = fly_expected_value(750.0, 3.0, 0.5, sigma=0.0, center=750.0)
        assert ev == pytest.approx(2.5)

    def test_wide_sigma_loses_the_debit(self):
        # huge move -> fly almost always expires worthless -> EV ~ -debit
        ev = fly_expected_value(750.0, 3.0, 0.5, sigma=200.0, center=750.0)
        assert ev == pytest.approx(-0.5, abs=0.05)

    def test_tighter_sigma_beats_wider(self):
        tight = fly_expected_value(750.0, 3.0, 0.5, sigma=3.0, center=750.0)
        wide = fly_expected_value(750.0, 3.0, 0.5, sigma=15.0, center=750.0)
        assert tight > wide


# ── POP-based width selection ────────────────────────────────────────────────

# Convex option mids so a 1-2-1 fly always prices to a positive debit that
# grows with width: debit(w) = f(b-w) - 2f(b) + f(b+w) = 0.02·w².
def _convex_mid(body=740.0):
    return lambda k: 0.01 * (k - body) ** 2 + 1.0


class TestCandidateWidths:
    def test_only_symmetric_listed_widths(self):
        strikes = [720, 725, 730, 735, 740, 745, 750, 755, 760]
        chain = make_chain(strikes, mid_fn=_convex_mid())
        assert candidate_widths(chain, body=740.0) == [5.0, 10.0, 15.0, 20.0]

    def test_min_width_filters_tight_wings(self):
        strikes = [738, 739, 740, 741, 742]
        chain = make_chain(strikes, mid_fn=_convex_mid())
        # widths 1 and 2 listed; default MIN_WIDTH=2 drops the 1-wide
        assert candidate_widths(chain, body=740.0) == [2.0]

    def test_max_width_caps_search(self):
        strikes = [720, 725, 730, 735, 740, 745, 750, 755, 760]
        chain = make_chain(strikes, mid_fn=_convex_mid())
        assert candidate_widths(chain, body=740.0, max_width=12.0) == [5.0, 10.0]

    def test_asymmetric_strike_dropped(self):
        # 730 has no mirror (750 missing) -> width 10 excluded
        strikes = [730, 735, 740, 745]
        chain = make_chain(strikes, mid_fn=_convex_mid())
        assert candidate_widths(chain, body=740.0) == [5.0]


class TestSelectWidthByPop:
    def _chain(self):
        strikes = [720, 725, 730, 735, 740, 745, 750, 755, 760]
        return make_chain(strikes, mid_fn=_convex_mid())

    def test_prefers_widest_high_pop_fly(self):
        # Tight sigma -> every listed fly is +EV, so the highest-POP (widest)
        # wins. The R:R-ladder would have picked the narrowest (5) instead.
        sel = select_width_by_pop(self._chain(), body=740.0, right="CALL",
                                  expiry=EXPIRY, sigma=6.0, center=740.0)
        assert sel["selected"] is not None
        assert sel["selected"]["width"] == 20.0
        assert sel["had_positive"] is True
        assert sel["reached_target"] is True
        # selected POP is the max POP across all scored candidates
        assert sel["selected"]["pop"] == max(c["pop"] for c in sel["scored"])

    def test_no_positive_ev_is_no_trade(self):
        # A move that dwarfs every tent -> all candidates negative-EV.
        sel = select_width_by_pop(self._chain(), body=740.0, right="CALL",
                                  expiry=EXPIRY, sigma=300.0, center=740.0)
        assert sel["selected"] is None
        assert sel["had_positive"] is False
        assert sel["best_pop_attempt"] is not None  # carried for the reason text

    def test_selected_is_positive_ev(self):
        sel = select_width_by_pop(self._chain(), body=740.0, right="CALL",
                                  expiry=EXPIRY, sigma=8.0, center=740.0,
                                  target_pop=DEFAULT_TARGET_POP)
        assert sel["selected"]["ev"] > 0
        # never picks a -EV fly even if it has a higher POP
        assert all(c["ev"] > 0 or c["pop"] <= sel["selected"]["pop"]
                   for c in sel["scored"])

    def test_reached_target_flag_gates_on_pop(self):
        # An unreachable target still returns a selection, but flags the miss
        # so the orchestrator can downgrade it to NO TRADE.
        miss = select_width_by_pop(self._chain(), body=740.0, right="CALL",
                                   expiry=EXPIRY, sigma=6.0, center=740.0,
                                   target_pop=0.99)
        assert miss["selected"] is not None
        assert miss["reached_target"] is False
        hit = select_width_by_pop(self._chain(), body=740.0, right="CALL",
                                  expiry=EXPIRY, sigma=6.0, center=740.0,
                                  target_pop=0.50)
        assert hit["reached_target"] is True

    def test_selected_clears_rr_floor(self):
        sel = select_width_by_pop(self._chain(), body=740.0, right="CALL",
                                  expiry=EXPIRY, sigma=6.0, center=740.0, min_rr=1.0)
        assert sel["selected"]["ratio"]["risk_reward"] >= 1.0

    def test_rr_floor_blocks_capital_torching_wides(self):
        # A tight floor keeps only the narrow, capital-efficient fly; loosening
        # it lets the wide high-POP fly through. Guards against deep-ITM boxes.
        chain = self._chain()
        strict = select_width_by_pop(chain, body=740.0, right="CALL", expiry=EXPIRY,
                                     sigma=6.0, center=740.0, min_rr=5.0)
        loose = select_width_by_pop(chain, body=740.0, right="CALL", expiry=EXPIRY,
                                    sigma=6.0, center=740.0, min_rr=1.0)
        assert strict["selected"]["width"] == 5.0      # only 1:9 fly clears 1:5
        assert loose["selected"]["width"] == 20.0      # widest +EV fly clears 1:1
        assert loose["selected"]["pop"] > strict["selected"]["pop"]
