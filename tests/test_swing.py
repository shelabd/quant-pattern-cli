"""Offline tests for the swing signal engine (no network)."""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_patterns.swing import (
    MIN_RR,
    STOP_ATR,
    TARGET_ATR,
    SwingSignal,
    bs_delta,
    detect_trend,
    evaluate_swing,
    format_swing_message,
    load_swing_journal,
    log_swing,
    obv,
    pick_option,
    score_swing_journal,
    signal_rvol,
    simulate_swing,
    wilder_atr,
    wilder_rsi,
)


# ── Frame builders ───────────────────────────────────────────────────────────

def make_frame(closes, volumes=None, lows=None, highs=None, opens=None):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    idx = pd.bdate_range("2026-01-05", periods=n)
    return pd.DataFrame({
        "Open": opens if opens is not None else closes - 0.1,
        "High": highs if highs is not None else closes + 0.5,
        "Low": lows if lows is not None else closes - 0.5,
        "Close": closes,
        "Volume": volumes if volumes is not None else np.full(n, 1e6),
    }, index=idx)


def uptrend_frame(n=60, step=0.6):
    return make_frame(100 + step * np.arange(n))


def pullback_frame():
    """Uptrend, 3-bar pullback toward the 20 EMA, then a reversal bar that
    closes above the prior high on expansion volume."""
    n = 60
    closes = 100 + 0.6 * np.arange(n, dtype=float)
    lows = closes - 0.5
    highs = closes + 0.5
    vols = np.full(n, 1e6)
    for k, i in enumerate((56, 57, 58), start=1):
        closes[i] = 100 + 0.6 * i - 1.8 * k
        lows[i] = closes[i] - 1.0
        highs[i] = closes[i] + 0.5
    # reversal bar: close above bar-58's high, in the top of its range
    closes[59], highs[59], lows[59] = 131.5, 131.7, 129.2
    vols[59] = 1.6e6
    return make_frame(closes, volumes=vols, lows=lows, highs=highs)


# ── Indicators ───────────────────────────────────────────────────────────────

def test_rsi_bounds_and_direction():
    up = wilder_rsi(pd.Series(np.linspace(100, 120, 40)))
    down = wilder_rsi(pd.Series(np.linspace(120, 100, 40)))
    assert 50 < up.iloc[-1] <= 100
    assert 0 <= down.iloc[-1] < 50


def test_atr_positive():
    df = uptrend_frame()
    assert float(wilder_atr(df).iloc[-1]) > 0


def test_obv_rises_on_up_days():
    df = uptrend_frame()
    series = obv(df["Close"], df["Volume"])
    assert series.iloc[-1] > series.iloc[10]


def test_signal_rvol_excludes_own_bar():
    vols = np.full(30, 1e6)
    vols[-1] = 2e6
    df = make_frame(100 + np.arange(30, dtype=float), volumes=vols)
    assert signal_rvol(df["Volume"]) == pytest.approx(2.0)


def test_detect_trend():
    assert detect_trend(uptrend_frame()) == "up"
    assert detect_trend(make_frame(200 - 0.6 * np.arange(60))) == "down"
    flat = make_frame(100 + 0.3 * np.sin(np.arange(80)))
    assert detect_trend(flat) == "sideways"


# ── Setup detection ──────────────────────────────────────────────────────────

def test_pullback_long_fires():
    sig = evaluate_swing("SPY", pullback_frame())
    assert sig.trend == "up"
    assert sig.setup == "pullback long"
    assert sig.direction == "long"
    assert not sig.stand_aside
    assert sig.stop < sig.close < sig.target
    assert sig.rr == pytest.approx(TARGET_ATR / STOP_ATR)
    assert sig.stop_pct is not None and sig.target_pct is not None


def test_breakout_long_needs_volume():
    df = uptrend_frame()
    prev_close = float(df["Close"].iloc[-2])
    level = SimpleNamespace(kind="resistance",
                            price=(prev_close + float(df["Close"].iloc[-1])) / 2,
                            touches=3)
    quiet = evaluate_swing("SPY", df, sr_levels=[level])
    assert quiet.setup != "breakout long"  # RVOL 1.0 < 1.3
    vols = np.full(len(df), 1e6)
    vols[-1] = 1.5e6
    loud = evaluate_swing("SPY", make_frame(df["Close"].values, volumes=vols),
                          sr_levels=[level])
    assert loud.setup == "breakout long"
    assert loud.direction == "long"


def test_sideways_stands_aside():
    sig = evaluate_swing("SPY", make_frame(100 + 0.3 * np.sin(np.arange(80))))
    assert sig.direction == "none"


def test_sr_cap_turns_thin_rr_into_stand_aside():
    df = pullback_frame()
    close = float(df["Close"].iloc[-1])
    atr_val = float(wilder_atr(df).iloc[-1])
    wall = SimpleNamespace(kind="resistance", price=close + 1.0 * atr_val, touches=4)
    sig = evaluate_swing("SPY", df, sr_levels=[wall])
    assert sig.setup == "pullback long"
    assert sig.stand_aside
    assert sig.rr < MIN_RR


def test_short_history_is_safe():
    sig = evaluate_swing("SPY", uptrend_frame(n=20))
    assert sig.direction == "none"
    assert sig.warnings


# ── Option ticket ────────────────────────────────────────────────────────────

def synthetic_chain(spot=100.0, iv=0.2):
    strikes = np.arange(spot - 10, spot + 11, 1.0)
    return pd.DataFrame({
        "strike": strikes,
        "iv": iv,
        "call_bid": np.maximum(spot - strikes, 0.5) - 0.05,
        "call_ask": np.maximum(spot - strikes, 0.5) + 0.05,
        "call_last": np.maximum(spot - strikes, 0.5),
        "put_bid": np.maximum(strikes - spot, 0.5) - 0.05,
        "put_ask": np.maximum(strikes - spot, 0.5) + 0.05,
        "put_last": np.maximum(strikes - spot, 0.5),
    })


def test_bs_delta_atm_near_half():
    d = bs_delta(100, 100, 0.2, 30 / 365, "call")
    assert 0.5 < d < 0.56
    assert bs_delta(100, 100, 0.2, 30 / 365, "put") == pytest.approx(d - 1.0)


def test_pick_option_targets_delta():
    as_of = date(2026, 7, 10)
    chains = [(date(2026, 7, 20), synthetic_chain()),   # 10 DTE — outside window
              (date(2026, 8, 10), synthetic_chain())]   # 31 DTE
    opt = pick_option(chains, 100.0, "long", as_of)
    assert opt is not None
    assert opt["right"] == "call"
    assert opt["expiry"] == "2026-08-10"
    assert 96 <= opt["strike"] <= 100
    assert 0.55 <= opt["delta"] <= 0.68
    put = pick_option(chains, 100.0, "short", as_of)
    assert put["right"] == "put" and put["delta"] < 0


def test_pick_option_none_outside_window():
    chains = [(date(2026, 7, 15), synthetic_chain())]  # 5 DTE
    assert pick_option(chains, 100.0, "long", date(2026, 7, 10)) is None


# ── Simulation ───────────────────────────────────────────────────────────────

LONG = {"direction": "long", "stop_pct": 2.0, "target_pct": 4.0, "max_hold_days": 10}


def bars(rows):
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"],
                        index=pd.bdate_range("2026-07-13", periods=len(rows)))


def test_simulate_target_hit():
    sim = simulate_swing(LONG, bars([(100, 101, 99.5, 100.5),
                                     (100.5, 104.5, 100, 104)]))
    assert sim["outcome"] == "target"
    assert sim["r"] == pytest.approx(2.0)


def test_simulate_stop_hit_and_same_bar_tie():
    stop = simulate_swing(LONG, bars([(100, 101, 99.5, 100.5),
                                      (100, 100.5, 97.9, 98)]))
    assert stop["outcome"] == "stop" and stop["r"] == -1.0
    tie = simulate_swing(LONG, bars([(100, 101, 99.5, 100.5),
                                     (100, 104.5, 97.5, 100)]))
    assert tie["outcome"] == "stop" and tie["r"] == -1.0


def test_simulate_gap_exits_at_open():
    sim = simulate_swing(LONG, bars([(100, 101, 99.5, 100.5),
                                     (97, 98, 96.5, 97.5)]))
    assert sim["outcome"] == "stop"
    assert sim["r"] == pytest.approx(-1.5)  # gapped through the stop


def test_simulate_time_exit_and_pending():
    flat = [(100, 101.5, 99.5, 101)] * 10
    sim = simulate_swing(LONG, bars(flat))
    assert sim["outcome"] == "time"
    assert sim["r"] == pytest.approx(0.5)
    assert simulate_swing(LONG, bars(flat[:3]))["outcome"] == "pending"
    assert simulate_swing(LONG, None)["outcome"] == "pending"


def test_simulate_short_direction():
    short = {"direction": "short", "stop_pct": 2.0, "target_pct": 4.0,
             "max_hold_days": 10}
    sim = simulate_swing(short, bars([(100, 100.5, 99.5, 100),
                                      (99, 99.5, 95.9, 96)]))
    assert sim["outcome"] == "target"
    assert sim["r"] == pytest.approx(2.0)


# ── Journal + scoring ────────────────────────────────────────────────────────

def make_sig(**over) -> SwingSignal:
    base = dict(ticker="SPY", as_of=date(2026, 7, 10), direction="long",
                setup="pullback long", close=100.0, trend="up", atr=1.5,
                stop=97.0, target=106.0, stop_pct=3.0, target_pct=6.0, rr=2.0)
    base.update(over)
    return SwingSignal(**base)


def test_log_swing_dedups_per_day(tmp_path):
    path = tmp_path / "swing.jsonl"
    _, first = log_swing(make_sig(), path)
    _, second = log_swing(make_sig(), path)
    assert first is True and second is False
    assert len(load_swing_journal(path)) == 1


def test_score_swing_journal_buckets():
    win = bars([(100, 101, 99.5, 100.5), (100.5, 106.5, 100, 106)])
    lose = bars([(100, 101, 99.5, 100.5), (100, 100.5, 96.5, 97)])
    frames = {"2026-07-08": win, "2026-07-09": lose, "2026-07-10": win}
    entries = [
        {"ticker": "SPY", "as_of": "2026-07-08", "direction": "long",
         "setup": "pullback long", "stop_pct": 3.0, "target_pct": 6.0,
         "max_hold_days": 10, "stand_aside": False},
        {"ticker": "SPY", "as_of": "2026-07-08", "direction": "long",  # dup
         "setup": "pullback long", "stop_pct": 3.0, "target_pct": 6.0,
         "max_hold_days": 10, "stand_aside": False},
        {"ticker": "SPY", "as_of": "2026-07-09", "direction": "long",
         "setup": "breakout long", "stop_pct": 3.0, "target_pct": 6.0,
         "max_hold_days": 10, "stand_aside": False},
        {"ticker": "SPY", "as_of": "2026-07-10", "direction": "long",
         "setup": "pullback long", "stop_pct": 3.0, "target_pct": 6.0,
         "max_hold_days": 10, "stand_aside": True},
        {"ticker": "SPY", "as_of": "2026-07-11", "direction": "none"},
        {"ticker": "SPY", "as_of": "2026-07-12", "direction": "long",
         "setup": "pullback long", "stop_pct": 3.0, "target_pct": 6.0,
         "max_hold_days": 10, "stand_aside": False},  # no bars -> pending
    ]
    stats = score_swing_journal(entries, lambda t, d: frames.get(d))
    assert stats["overall"]["n"] == 2
    assert stats["overall"]["win_rate"] == 0.5
    assert stats["pending"] == 1
    assert stats["stand_aside"]["n"] == 1
    assert stats["stand_aside"]["win_rate"] == 1.0
    assert stats["by_setup"]["breakout long"]["stop_rate"] == 1.0
    assert stats["by_outcome"] == {"stop": 1, "target": 1, "time": 0}


# ── Formatting ───────────────────────────────────────────────────────────────

def test_format_message_long():
    msg = format_swing_message(make_sig(option={
        "right": "call", "expiry": "2026-08-10", "dte": 31, "strike": 99.0,
        "delta": 0.61, "mid": 3.4, "bid": 3.3, "ask": 3.5, "breakeven": 102.4}))
    assert "BUY CALLS" in msg
    assert "2026-08-10 99C" in msg
    assert "not financial advice" in msg


def test_format_message_stand_aside():
    msg = format_swing_message(make_sig(
        stand_aside=True, stand_aside_reason="reward:risk 0.5 < 1.5"))
    assert "✋" in msg and "BUY" not in msg


def test_format_message_none():
    sig = SwingSignal(ticker="SPY", as_of=date(2026, 7, 10), direction="none",
                      setup="", close=100.0, trend="sideways")
    assert "stand aside" in format_swing_message(sig).lower()


# ── Low-volume pump + flush short ────────────────────────────────────────────

def pump_base(n_base=54, base_step=0.25):
    """Rising base that stays below 750, ending ~745."""
    closes = 745 - base_step * np.arange(n_base)[::-1]
    return list(closes)


def pump_frame(streak_vol=0.7e6, flush=False, flush_vol=2.0e6):
    """~745 grind-up base, then a 6-day low-volume streak above 750
    (the May 26 - Jun 2 2026 SPY analog), optionally + the Jun 5 flush bar."""
    closes = pump_base()
    vols = [1e6] * len(closes)
    streak = [750.6, 750.5, 754.6, 756.5, 758.5, 759.6]
    closes += streak
    vols += [streak_vol] * len(streak)
    if flush:
        closes.append(737.5)
        vols.append(flush_vol)
    closes = np.asarray(closes)
    return make_frame(closes, volumes=np.asarray(vols),
                      lows=closes - 1.0, highs=closes + 1.0,
                      opens=closes + 0.5 if flush else closes - 0.1)


def test_detect_pump_on_may_analog():
    from quant_patterns.swing import detect_pump
    pump = detect_pump(pump_frame())
    assert pump is not None
    assert pump["level"] == 750.0
    assert pump["days"] == 6
    assert pump["mean_rvol"] < 0.9
    assert pump["gain_pct"] > 0


def test_no_pump_on_normal_volume():
    from quant_patterns.swing import detect_pump
    assert detect_pump(pump_frame(streak_vol=1.0e6)) is None


def test_pump_warns_but_no_flush_setup():
    sig = evaluate_swing("SPY", pump_frame())
    assert sig.pump is not None
    assert sig.setup != "pump flush short"
    assert any("low-volume pump" in w for w in sig.warnings)
    assert any("day 6" in w for w in sig.warnings)  # late-stage note


def test_flush_short_fires_counter_trend():
    sig = evaluate_swing("SPY", pump_frame(flush=True))
    # The flush bar itself breaks the EMA-stack "up" read — which is exactly
    # why this setup must be exempt from the trend gate.
    assert sig.trend != "down"
    assert sig.setup == "pump flush short"
    assert sig.direction == "short"
    assert not sig.stand_aside
    assert sig.stop > sig.close > sig.target
    assert sig.pump is not None and sig.pump["level"] == 750.0
    assert any("counter-trend" in e for e in sig.evidence)


def test_flush_requires_expansion_volume():
    sig = evaluate_swing("SPY", pump_frame(flush=True, flush_vol=0.9e6))
    assert sig.setup != "pump flush short"
