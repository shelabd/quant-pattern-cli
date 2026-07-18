"""Tests for the rally-potential screener's pure logic (screener.py)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_patterns.screener import (
    FAMILIES,
    FACTOR_DIRECTIONS,
    PROFILE_WEIGHTS,
    ScreenResult,
    build_factor_panel,
    build_results,
    compute_factors,
    composite_scores,
    cross_rank,
    family_scores,
    format_screen_message,
    load_screen_journal,
    log_screen,
    options_flow_note,
    reasons_for,
    score_screen_journal,
)


# ── Synthetic frames ─────────────────────────────────────────────────────────

def make_frame(closes: np.ndarray, volumes: np.ndarray = None,
               spread: float = 0.01) -> pd.DataFrame:
    n = len(closes)
    if volumes is None:
        volumes = np.full(n, 1_000_000.0)
    dates = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({
        "Open": closes * (1 - spread / 2),
        "High": closes * (1 + spread),
        "Low": closes * (1 - spread),
        "Close": closes,
        "Volume": volumes,
    }, index=dates)
    df.index.name = "Date"
    return df


def trending_up(n=300, daily=0.003):
    return make_frame(100 * np.cumprod(np.full(n, 1 + daily)))


def flat(n=300):
    return make_frame(np.full(n, 100.0))


def downtrend(n=300, daily=0.003):
    return make_frame(100 * np.cumprod(np.full(n, 1 - daily)))


def squeezing(n=300):
    """Volatile early history, dead-quiet recent tape near its high."""
    rng = np.random.default_rng(7)
    early = 100 + np.cumsum(rng.normal(0, 2.0, n - 30))
    late = np.full(30, early[-1]) + rng.normal(0, 0.05, 30)
    return make_frame(np.concatenate([early, late]))


# ── compute_factors ──────────────────────────────────────────────────────────

def test_compute_factors_rejects_short_history():
    assert compute_factors(trending_up(60)) is None
    assert compute_factors(None) is None


def test_factor_keys_cover_all_families():
    factors = compute_factors(trending_up())
    for fam, cols in FAMILIES.items():
        for col in cols:
            assert col in factors, f"{fam}/{col} missing"
            assert col in FACTOR_DIRECTIONS


def test_momentum_signs():
    up = compute_factors(trending_up())
    down = compute_factors(downtrend())
    assert up["ret_12_1"] > 0 > down["ret_12_1"]
    assert up["ret_3m"] > 0 > down["ret_3m"]


def test_uptrend_sits_at_its_52w_high_with_full_stack():
    f = compute_factors(trending_up())
    assert f["pct_off_52w_high"] == pytest.approx(0, abs=1.5)
    assert f["days_since_52w_high"] <= 1
    assert f["ema_stack"] == 4.0
    d = compute_factors(downtrend())
    assert d["pct_off_52w_high"] < -30
    assert d["ema_stack"] <= 1.0


def test_squeeze_detects_compression():
    tight = compute_factors(squeezing())
    loose = compute_factors(make_frame(
        100 + np.cumsum(np.random.default_rng(3).normal(0, 2.0, 300))))
    assert tight["bb_width_pctile"] < 15
    assert tight["atr_contraction"] < loose["atr_contraction"]


def test_updown_volume_ratio_reflects_accumulation():
    n = 300
    closes = 100 + np.cumsum(np.tile([1.0, -0.5], n // 2))
    vols = np.tile([3_000_000.0, 1_000_000.0], n // 2)  # heavy up days
    f = compute_factors(make_frame(closes, vols))
    assert f["updown_vol"] > 2.0


# ── ranking & composites ─────────────────────────────────────────────────────

def make_panel():
    return build_factor_panel({
        "UP": trending_up(), "FLAT": flat(),
        "DOWN": downtrend(), "SQZ": squeezing(),
    })


def test_cross_rank_direction_awareness():
    pct = cross_rank(make_panel())
    # Higher momentum ranks higher...
    assert pct.loc["UP", "ret_12_1"] > pct.loc["DOWN", "ret_12_1"]
    # ...but LOWER bb width ranks higher (direction -1).
    panel = make_panel()
    tightest = panel["bb_width_pctile"].idxmin()
    assert pct.loc[tightest, "bb_width_pctile"] == pct["bb_width_pctile"].max()


def test_cross_rank_keeps_nan():
    panel = make_panel()
    panel.loc["UP", "ret_12_1"] = np.nan
    pct = cross_rank(panel)
    assert np.isnan(pct.loc["UP", "ret_12_1"])


def test_composite_prefers_uptrend_for_position_profile():
    panel = make_panel()
    fams = family_scores(cross_rank(panel))
    comp = composite_scores(fams, "position")
    assert comp["UP"] > comp["DOWN"]
    assert comp["UP"] > comp["FLAT"]


def test_composite_renormalizes_missing_families():
    fams = pd.DataFrame({"momentum": [80.0], "hi52": [np.nan],
                         "trend": [np.nan], "squeeze": [np.nan],
                         "trigger": [np.nan], "volume": [np.nan]},
                        index=["X"])
    comp = composite_scores(fams, "position")
    assert comp["X"] == pytest.approx(80.0)


def test_profile_weights_sum_to_one():
    for profile, weights in PROFILE_WEIGHTS.items():
        assert sum(weights.values()) == pytest.approx(1.0), profile
        assert set(weights) == set(FAMILIES)


def test_build_results_orders_by_score_and_carries_reasons():
    panel = make_panel()
    pct = cross_rank(panel)
    fams = family_scores(pct)
    closes = {t: 100.0 for t in panel.index}
    results = build_results("position", date(2026, 7, 17), panel, pct, fams,
                            closes, top=3)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].ticker == "UP"
    assert results[0].reasons  # non-empty why-strings


def test_reasons_reference_top_weighted_families():
    panel = make_panel()
    pct = cross_rank(panel)
    fams = family_scores(pct)
    reasons = reasons_for("swing", fams.loc["SQZ"], panel.loc["SQZ"],
                          pct.loc["SQZ"])
    assert any("BB width" in r for r in reasons)


# ── options flow note ────────────────────────────────────────────────────────

def make_chain(call_vol, put_vol, strikes=(90, 100, 110), spot=100):
    n = len(strikes)
    return pd.DataFrame({
        "strike": strikes,
        "call_oi": [100] * n, "put_oi": [100] * n,
        "call_vol": [call_vol / n] * n, "put_vol": [put_vol / n] * n,
        "call_bid": [1.0] * n, "call_ask": [1.2] * n, "call_last": [1.1] * n,
        "put_bid": [1.0] * n, "put_ask": [1.2] * n, "put_last": [1.1] * n,
        "iv": [0.3] * n,
    })


def test_options_flow_note_fires_on_call_skew():
    chains = [(date(2026, 8, 1), make_chain(9000, 1500))]
    note = options_flow_note(chains, spot=100, as_of=date(2026, 7, 17))
    assert note is not None and "C/P vol" in note


def test_options_flow_note_quiet_or_balanced_returns_none():
    assert options_flow_note([(date(2026, 8, 1), make_chain(500, 400))],
                             100, date(2026, 7, 17)) is None
    assert options_flow_note([(date(2026, 8, 1), make_chain(5000, 5000))],
                             100, date(2026, 7, 17)) is None
    assert options_flow_note([], 100, date(2026, 7, 17)) is None


# ── journal ──────────────────────────────────────────────────────────────────

def make_result(ticker="UP", profile="swing", score=88.0):
    return ScreenResult(ticker=ticker, profile=profile,
                        as_of=date(2026, 7, 17), score=score,
                        families={"momentum": 90.0}, reasons=["r"], close=100.0)


def test_log_screen_dedups_on_asof_profile_ticker(tmp_path):
    path = tmp_path / "screen_journal.jsonl"
    assert log_screen([make_result(), make_result(profile="position")], path) == 2
    # Re-fire same night: nothing appended.
    assert log_screen([make_result()], path) == 0
    assert len(load_screen_journal(path)) == 2


def test_score_screen_journal_forward_returns():
    entries = [
        {"as_of": "2026-07-17", "profile": "swing", "ticker": "WIN", "score": 85.0},
        {"as_of": "2026-07-17", "profile": "swing", "ticker": "LOSE", "score": 55.0},
    ]
    n = 70
    dates = pd.bdate_range("2026-07-20", periods=n)

    def bars(mult):
        closes = 100 * np.cumprod(np.full(n, mult))
        return pd.DataFrame({"Open": np.concatenate([[100.0], closes[:-1]]),
                             "High": closes, "Low": closes,
                             "Close": closes, "Volume": np.full(n, 1e6)},
                            index=dates)

    frames = {"WIN": bars(1.01), "LOSE": bars(0.99), "SPY": bars(1.0)}

    stats = score_screen_journal(entries, lambda t, a: frames.get(t))
    assert stats["n_picks"] == 2
    swing = stats["by_profile"]["swing"]
    assert swing["overall"]["+5d"]["n"] == 2
    hi = swing["by_score_band"]["80-100"]["+5d"]
    lo = swing["by_score_band"]["0-59"]["+5d"]
    assert hi["avg_pct"] > 0 > lo["avg_pct"]
    assert hi["avg_vs_spy_pct"] > 0  # WIN beats flat SPY


def test_score_screen_journal_pending_when_no_bars():
    entries = [{"as_of": "2026-07-17", "profile": "swing",
                "ticker": "NEW", "score": 90.0}]
    stats = score_screen_journal(entries, lambda t, a: None)
    assert stats["n_picks"] == 0
    assert stats["pending"] == 1


# ── message formatting ───────────────────────────────────────────────────────

def test_format_screen_message_lists_profiles_and_disclaimer():
    msg = format_screen_message(
        {"swing": [make_result()], "position": [make_result(profile="position")]},
        date(2026, 7, 17), regime="SPY above its 200-DMA (risk-on)")
    assert "SWING" in msg and "POSITION" in msg
    assert "UP" in msg and "risk-on" in msg
    assert "not financial advice" in msg
