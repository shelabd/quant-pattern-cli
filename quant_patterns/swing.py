"""Swing signal engine: daily buy/sell guidance for 2-10 day SPY option swings.

Pure logic, no Rich/Click/network. One evaluation per completed daily bar,
built from three families of evidence:

* **Trend** — 20/50 EMA stack plus 50-EMA slope. Setups only fire in the
  trend's direction; a sideways tape stands aside.
* **Price structure** — pullback-to-trend reversals (the pullback tagged the
  20 EMA or dipped RSI, then a reversal bar closed above the prior high) and
  S/R breakouts (close through a multi-touch level detected by
  `analysis.find_support_resistance`).
* **Volumetrics** — signal-bar RVOL vs its 20-day baseline and the 10-day OBV
  slope. Breakouts *require* expansion volume; pullback reversals need at
  least one of RVOL/OBV confirming, and both are always reported.

Geometry is ATR-based: stop 1.5x ATR beyond entry, target 3x ATR (2R),
capped just inside the nearest opposing S/R level when that is closer —
a capped target that drops reward:risk below MIN_RR turns the signal into
an explicit stand-aside. Time exit after MAX_HOLD_DAYS sessions.

Forward-test honesty (lessons from the fly journal): the journal stores the
stop/target as *percentages* of the signal close, and `simulate_swing`
re-derives absolute levels from the realized **next-session open** on
unadjusted bars — so the scorecard measures a fill you could actually get,
and a later dividend adjustment cannot silently rewrite history.

Analysis only — never routes orders, not financial advice.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time
from statistics import NormalDist
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
# The signal reads completed daily bars only; before this ET time the cron
# stays silent and mid-session runs drop today's partial bar.
SESSION_CLOSE_ET = time(16, 0)

# ── Tunables ─────────────────────────────────────────────────────────────────

TREND_EMA_FAST = 20
TREND_EMA_SLOW = 50
SLOPE_LOOKBACK = 10          # bars the 50-EMA slope is measured over
RSI_PERIOD = 14
ATR_PERIOD = 14

# Pullback setup: within the last PULLBACK_BARS the tape must have pulled in —
# a low within PULLBACK_ATR_PROX ATRs of the 20 EMA, or RSI under PULLBACK_RSI.
PULLBACK_BARS = 5
PULLBACK_ATR_PROX = 0.5
PULLBACK_RSI = 45.0
# Reversal trigger bar must close in the top (bottom, for shorts) fraction of
# its range as well as beyond the prior bar's extreme.
TRIGGER_RANGE_POS = 0.60

# Volume confirmation. RVOL is the signal bar's volume over the mean of the
# prior RVOL_WINDOW bars; OBV slope is measured over OBV_LOOKBACK bars.
RVOL_WINDOW = 20
OBV_LOOKBACK = 10
BREAKOUT_RVOL = 1.3          # breakouts need real expansion volume
CONFIRM_RVOL = 1.0           # pullbacks need this OR a confirming OBV slope

# Breakout setup: level must have this many touches to count as structure.
BREAKOUT_MIN_TOUCHES = 2

# Trade geometry (ATR multiples) and horizon.
STOP_ATR = 1.5
TARGET_ATR = 3.0
MIN_RR = 1.5
MAX_HOLD_DAYS = 10
# Targets are capped just inside opposing S/R: this many ATRs inside it.
SR_CAP_BUFFER_ATR = 0.25

# Option ticket: expiry roughly 2x the max hold plus buffer, strike near
# TARGET_DELTA so the contract moves with the shares without pure-theta risk.
DTE_MIN = 21
DTE_MAX = 50
DTE_IDEAL = 30
TARGET_DELTA = 0.60


# ── Indicators (pandas, Wilder-style smoothing) ──────────────────────────────

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    # All-gain stretches (loss==0) are RSI 100 by convention.
    return rsi.fillna(100.0).where(delta.notna(), np.nan)


def wilder_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def signal_rvol(volume: pd.Series, window: int = RVOL_WINDOW) -> Optional[float]:
    """Last bar's volume vs the mean of the prior `window` bars (the bar's
    own volume excluded from its baseline)."""
    if len(volume) < window + 1:
        return None
    base = float(volume.iloc[-window - 1:-1].mean())
    if base <= 0:
        return None
    return float(volume.iloc[-1]) / base


def obv_rising(close: pd.Series, volume: pd.Series,
               lookback: int = OBV_LOOKBACK) -> Optional[bool]:
    series = obv(close, volume)
    if len(series) < lookback + 1:
        return None
    return bool(series.iloc[-1] > series.iloc[-1 - lookback])


def detect_trend(df: pd.DataFrame) -> str:
    """'up' | 'down' | 'sideways' from the EMA stack + slow-EMA slope."""
    if len(df) < TREND_EMA_SLOW + SLOPE_LOOKBACK:
        return "sideways"
    close = df["Close"]
    e_fast = ema(close, TREND_EMA_FAST)
    e_slow = ema(close, TREND_EMA_SLOW)
    slope = float(e_slow.iloc[-1] - e_slow.iloc[-1 - SLOPE_LOOKBACK])
    c, f, s = float(close.iloc[-1]), float(e_fast.iloc[-1]), float(e_slow.iloc[-1])
    if c > f > s and slope > 0:
        return "up"
    if c < f < s and slope < 0:
        return "down"
    return "sideways"


# ── Signal dataclass ─────────────────────────────────────────────────────────

@dataclass
class SwingSignal:
    """One end-of-day evaluation. `direction` is 'long'/'short'/'none';
    a directional signal with `stand_aside` set means the setup fired but
    the geometry doesn't pay — reported, journaled, not recommended."""
    ticker: str
    as_of: date                  # date of the completed signal bar
    direction: str
    setup: str                   # "pullback long", "breakout long", mirrors, or ""
    close: float
    trend: str
    atr: Optional[float] = None
    rsi: Optional[float] = None
    rvol: Optional[float] = None
    obv_rising: Optional[bool] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    stop_pct: Optional[float] = None     # % of close, positive
    target_pct: Optional[float] = None   # % of close, positive
    rr: Optional[float] = None
    max_hold_days: int = MAX_HOLD_DAYS
    stand_aside: bool = False
    stand_aside_reason: str = ""
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    option: Optional[dict] = None

    def to_dict(self) -> dict:
        def r2(v):
            return round(v, 2) if v is not None else None
        return {
            "ticker": self.ticker,
            "as_of": self.as_of.isoformat(),
            "direction": self.direction,
            "setup": self.setup,
            "close": r2(self.close),
            "trend": self.trend,
            "atr": r2(self.atr),
            "rsi": round(self.rsi, 1) if self.rsi is not None else None,
            "rvol": r2(self.rvol),
            "obv_rising": self.obv_rising,
            "stop": r2(self.stop),
            "target": r2(self.target),
            "stop_pct": round(self.stop_pct, 4) if self.stop_pct is not None else None,
            "target_pct": round(self.target_pct, 4) if self.target_pct is not None else None,
            "rr": round(self.rr, 2) if self.rr is not None else None,
            "max_hold_days": self.max_hold_days,
            "stand_aside": self.stand_aside,
            "stand_aside_reason": self.stand_aside_reason,
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "option": self.option,
            "disclaimer": "Analysis only — not financial advice.",
        }


# ── Setup detection ──────────────────────────────────────────────────────────

def _volume_context(df: pd.DataFrame) -> tuple[Optional[float], Optional[bool], list[str]]:
    rvol = signal_rvol(df["Volume"])
    rising = obv_rising(df["Close"], df["Volume"])
    lines = []
    if rvol is not None:
        lines.append(f"signal-bar volume {rvol:.1f}x its 20-day average")
    if rising is not None:
        lines.append("OBV rising over 10 sessions" if rising
                      else "OBV falling over 10 sessions")
    return rvol, rising, lines


def _pullback_ready(df: pd.DataFrame, atr_val: float, direction: str) -> bool:
    """Did the tape pull in against the trend within the last PULLBACK_BARS?"""
    close = df["Close"]
    e_fast = ema(close, TREND_EMA_FAST)
    rsi = wilder_rsi(close)
    recent_rsi = rsi.iloc[-PULLBACK_BARS:]
    if direction == "long":
        touched = (df["Low"].iloc[-PULLBACK_BARS:]
                   <= e_fast.iloc[-PULLBACK_BARS:] + PULLBACK_ATR_PROX * atr_val).any()
        dipped = (recent_rsi < PULLBACK_RSI).any()
    else:
        touched = (df["High"].iloc[-PULLBACK_BARS:]
                   >= e_fast.iloc[-PULLBACK_BARS:] - PULLBACK_ATR_PROX * atr_val).any()
        dipped = (recent_rsi > 100 - PULLBACK_RSI).any()
    return bool(touched or dipped)


def _reversal_trigger(df: pd.DataFrame, direction: str) -> bool:
    """Signal bar reclaims the prior bar's extreme and closes strong."""
    bar, prev = df.iloc[-1], df.iloc[-2]
    rng = bar["High"] - bar["Low"]
    if rng <= 0:
        return False
    pos = (bar["Close"] - bar["Low"]) / rng
    if direction == "long":
        return bool(bar["Close"] > prev["High"] and pos >= TRIGGER_RANGE_POS)
    return bool(bar["Close"] < prev["Low"] and pos <= 1 - TRIGGER_RANGE_POS)


def _breakout_level(df: pd.DataFrame, sr_levels: list, direction: str) -> Optional[object]:
    """The multi-touch S/R level the signal bar closed through, if any."""
    close, prev_close = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
    for lv in sr_levels or []:
        if getattr(lv, "touches", 0) < BREAKOUT_MIN_TOUCHES:
            continue
        if direction == "long" and lv.kind == "resistance" \
                and prev_close <= lv.price < close:
            return lv
        if direction == "short" and lv.kind == "support" \
                and prev_close >= lv.price > close:
            return lv
    return None


def _cap_target(close: float, target: float, sr_levels: list,
                atr_val: float, direction: str) -> tuple[float, Optional[str]]:
    """Pull the target just inside the nearest opposing S/R level when that
    level sits between entry and the raw ATR target."""
    buffer = SR_CAP_BUFFER_ATR * atr_val
    if direction == "long":
        blockers = [lv for lv in sr_levels or []
                    if lv.kind == "resistance" and close < lv.price - buffer < target]
        if blockers:
            lv = min(blockers, key=lambda b: b.price)
            return lv.price - buffer, f"target capped inside resistance {lv.price:.2f}"
    else:
        blockers = [lv for lv in sr_levels or []
                    if lv.kind == "support" and target < lv.price + buffer < close]
        if blockers:
            lv = max(blockers, key=lambda b: b.price)
            return lv.price + buffer, f"target capped inside support {lv.price:.2f}"
    return target, None


def evaluate_swing(ticker: str, df: pd.DataFrame,
                   sr_levels: Optional[list] = None,
                   warnings: Optional[list[str]] = None) -> SwingSignal:
    """Evaluate the last completed daily bar of `df` (OHLCV, date-indexed).

    Needs ~TREND_EMA_SLOW + SLOPE_LOOKBACK bars of history; fewer returns a
    stand-aside 'none' signal with an explanatory warning.
    """
    warns = list(warnings or [])
    as_of = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]
    close = float(df["Close"].iloc[-1])

    if len(df) < TREND_EMA_SLOW + SLOPE_LOOKBACK:
        warns.append(f"only {len(df)} daily bars — need "
                     f"{TREND_EMA_SLOW + SLOPE_LOOKBACK} for trend detection")
        return SwingSignal(ticker=ticker, as_of=as_of, direction="none",
                           setup="", close=close, trend="sideways", warnings=warns)

    trend = detect_trend(df)
    atr_val = float(wilder_atr(df).iloc[-1])
    rsi_val = float(wilder_rsi(df["Close"]).iloc[-1])
    rvol, rising, vol_lines = _volume_context(df)

    sig = SwingSignal(ticker=ticker, as_of=as_of, direction="none", setup="",
                      close=close, trend=trend, atr=atr_val, rsi=rsi_val,
                      rvol=rvol, obv_rising=rising, warnings=warns)

    direction = {"up": "long", "down": "short"}.get(trend)
    if direction is None:
        sig.evidence.append("EMA stack sideways — no trend to swing with")
        return sig

    # Breakout beats pullback when both fire on the same bar (rarer, cleaner).
    setup = ""
    evidence = [f"trend {trend} (close vs 20/50 EMA stack, 50-EMA slope)"]
    level = _breakout_level(df, sr_levels, direction)
    if level is not None:
        vol_ok = rvol is not None and rvol >= BREAKOUT_RVOL
        if vol_ok:
            setup = f"breakout {direction}"
            evidence.append(
                f"closed through {level.kind} {level.price:.2f} "
                f"({level.touches} touches) on {rvol:.1f}x volume")
            if rising is False:
                sig.warnings.append("OBV not confirming the breakout — "
                                    "watch for a failed break")
        else:
            sig.warnings.append(
                f"crossed {level.kind} {level.price:.2f} but volume "
                f"{'unknown' if rvol is None else f'{rvol:.1f}x'} < "
                f"{BREAKOUT_RVOL:g}x — unconfirmed break, standing aside")

    if not setup and _pullback_ready(df, atr_val, direction) \
            and _reversal_trigger(df, direction):
        vol_ok = (rvol is not None and rvol >= CONFIRM_RVOL) or rising is True
        if vol_ok:
            setup = f"pullback {direction}"
            evidence.append(
                ("pullback to the 20 EMA / RSI dip, then a reversal bar closed "
                 + ("above the prior high" if direction == "long"
                    else "below the prior low")))
        else:
            sig.warnings.append("pullback reversal fired but neither RVOL nor "
                                "OBV confirms — standing aside")

    if not setup:
        if not sig.warnings:
            sig.evidence.append(f"trend {trend}, no entry trigger — wait for a "
                                "pullback reversal or a confirmed breakout")
        return sig

    # ── Geometry ─────────────────────────────────────────────────────────
    stop_dist = STOP_ATR * atr_val
    sign = 1.0 if direction == "long" else -1.0
    stop = close - sign * stop_dist
    target = close + sign * TARGET_ATR * atr_val
    target, cap_note = _cap_target(close, target, sr_levels, atr_val, direction)
    if cap_note:
        evidence.append(cap_note)
    rr = abs(target - close) / stop_dist

    sig.direction = direction
    sig.setup = setup
    sig.evidence = evidence + vol_lines
    sig.stop, sig.target = stop, target
    sig.stop_pct = stop_dist / close * 100
    sig.target_pct = abs(target - close) / close * 100
    sig.rr = rr
    if rr < MIN_RR:
        sig.stand_aside = True
        sig.stand_aside_reason = (f"reward:risk {rr:.1f} < {MIN_RR:g} after the "
                                  "S/R cap — structure too close, stand aside")
    return sig


# ── Option ticket ────────────────────────────────────────────────────────────

def bs_delta(spot: float, strike: float, iv: float, t_years: float,
             right: str) -> Optional[float]:
    """Black-Scholes delta, zero rate/dividend. right: 'call' | 'put'."""
    if spot <= 0 or strike <= 0 or iv <= 0 or t_years <= 0:
        return None
    d1 = (math.log(spot / strike) + (iv * iv / 2) * t_years) / (iv * math.sqrt(t_years))
    nd = NormalDist().cdf(d1)
    return nd if right == "call" else nd - 1.0


def pick_option(chains: list[tuple[date, pd.DataFrame]], spot: float,
                direction: str, as_of: date) -> Optional[dict]:
    """Pick the contract for the swing: expiry nearest DTE_IDEAL within
    [DTE_MIN, DTE_MAX], strike with |delta| nearest TARGET_DELTA.

    `chains` are (expiry, CHAIN_COLUMNS frame) pairs. Returns None when no
    expiry fits or the chain is unusable — the swing signal stands alone.
    """
    right = "call" if direction == "long" else "put"
    window = [(exp, ch) for exp, ch in chains
              if DTE_MIN <= (exp - as_of).days <= DTE_MAX]
    if not window:
        return None
    expiry, chain = min(window, key=lambda t: abs((t[0] - as_of).days - DTE_IDEAL))
    dte = (expiry - as_of).days
    t_years = dte / 365.0

    best = None
    for _, row in chain.iterrows():
        iv = float(row.get("iv") or 0)
        delta = bs_delta(spot, float(row["strike"]), iv, t_years, right)
        if delta is None:
            continue
        bid = float(row.get(f"{right}_bid") or 0)
        ask = float(row.get(f"{right}_ask") or 0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 \
            else float(row.get(f"{right}_last") or 0)
        if mid <= 0:
            continue
        gap = abs(abs(delta) - TARGET_DELTA)
        if best is None or gap < best["gap"]:
            best = {"gap": gap, "strike": float(row["strike"]), "delta": delta,
                    "mid": mid, "bid": bid, "ask": ask}
    if best is None:
        return None
    breakeven = best["strike"] + best["mid"] if right == "call" \
        else best["strike"] - best["mid"]
    return {
        "right": right,
        "expiry": expiry.isoformat(),
        "dte": dte,
        "strike": best["strike"],
        "delta": round(best["delta"], 2),
        "mid": round(best["mid"], 2),
        "bid": round(best["bid"], 2),
        "ask": round(best["ask"], 2),
        "breakeven": round(breakeven, 2),
    }


def option_risk_estimate(option: dict, stop_dist: float) -> Optional[float]:
    """Rough $ loss per contract if the stop is hit: |delta| x stop distance
    x 100. Delta drift and theta make the real number worse — an estimate,
    not a bound (max loss is the full debit)."""
    if not option or option.get("delta") is None:
        return None
    return abs(option["delta"]) * stop_dist * 100


# ── Journal IO ───────────────────────────────────────────────────────────────

def load_swing_journal(path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"Skipping corrupt swing journal line: {line[:80]}")
    return entries


def log_swing(sig: SwingSignal, path) -> tuple[dict, bool]:
    """Append one signal per (ticker, as_of); duplicates (wake-coalesced cron
    runs, manual re-runs) are dropped. Returns (entry, appended)."""
    entry = {"logged_at": datetime.now().isoformat(timespec="seconds"),
             **sig.to_dict()}
    for existing in load_swing_journal(path):
        if existing.get("ticker") == entry["ticker"] \
                and existing.get("as_of") == entry["as_of"]:
            return existing, False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry, True


# ── Forward-test scoring (`qpat swing --score`) ─────────────────────────────

def simulate_swing(entry: dict, bars: pd.DataFrame) -> dict:
    """Replay one journaled signal against the daily bars AFTER its as_of.

    Fill at the first bar's open (the only entry you can actually place from
    an after-close signal); stop/target re-derived from the journaled
    percentages against that fill, so dividend adjustments can't shift them.
    Walk up to max_hold_days bars: stop wins same-bar ties, adverse gaps exit
    at the open (worse than the stop), favorable gaps exit at the open
    (better than the target), else mark to the last bar's close.

    Returns {outcome: stop|target|time|pending, r, fill, exit}.
    """
    if bars is None or bars.empty:
        return {"outcome": "pending", "r": None}
    direction = entry["direction"]
    sign = 1.0 if direction == "long" else -1.0
    fill = float(bars["Open"].iloc[0])
    stop = fill * (1 - sign * entry["stop_pct"] / 100)
    target = fill * (1 + sign * entry["target_pct"] / 100)
    risk = abs(fill - stop)
    if risk <= 0:
        return {"outcome": "pending", "r": None}

    hold = bars.iloc[:entry.get("max_hold_days", MAX_HOLD_DAYS)]
    for i, (_, bar) in enumerate(hold.iterrows()):
        o = float(bar["Open"])
        if i > 0 and (o <= stop if direction == "long" else o >= stop):
            return {"outcome": "stop", "r": (o - fill) / risk * sign,
                    "fill": fill, "exit": o}
        if i > 0 and (o >= target if direction == "long" else o <= target):
            return {"outcome": "target", "r": (o - fill) / risk * sign,
                    "fill": fill, "exit": o}
        stopped = bar["Low"] <= stop if direction == "long" else bar["High"] >= stop
        won = bar["High"] >= target if direction == "long" else bar["Low"] <= target
        if stopped:  # same-bar tie -> stop
            return {"outcome": "stop", "r": -1.0, "fill": fill, "exit": stop}
        if won:
            return {"outcome": "target", "r": abs(target - fill) / risk,
                    "fill": fill, "exit": target}
    if len(hold) < entry.get("max_hold_days", MAX_HOLD_DAYS):
        return {"outcome": "pending", "r": None}
    exit_px = float(hold["Close"].iloc[-1])
    return {"outcome": "time", "r": (exit_px - fill) / risk * sign,
            "fill": fill, "exit": exit_px}


def _score_bucket(results: list[dict]) -> dict:
    n = len(results)
    wins = [r for r in results if r["r"] > 0]
    stops = [r for r in results if r["outcome"] == "stop"]
    return {
        "n": n,
        "win_rate": round(len(wins) / n, 3) if n else None,
        "stop_rate": round(len(stops) / n, 3) if n else None,
        "avg_r": round(sum(r["r"] for r in results) / n, 2) if n else None,
        "total_r": round(sum(r["r"] for r in results), 2) if n else None,
    }


def score_swing_journal(entries: list[dict],
                        get_bars: Callable[[str, str], Optional[pd.DataFrame]]) -> dict:
    """Score journaled directional signals. `get_bars(ticker, as_of_iso)`
    returns completed daily bars strictly AFTER as_of (unadjusted), or None
    when unavailable. Dedup on (ticker, as_of)."""
    seen: set = set()
    scored: list[dict] = []
    pending = 0
    for e in entries:
        if e.get("direction") not in ("long", "short"):
            continue
        key = (e["ticker"], e["as_of"])
        if key in seen:
            continue
        seen.add(key)
        bars = get_bars(e["ticker"], e["as_of"])
        sim = simulate_swing(e, bars)
        if sim["outcome"] == "pending":
            pending += 1
            continue
        scored.append({**sim, "as_of": e["as_of"], "direction": e["direction"],
                       "setup": e.get("setup", ""),
                       "stand_aside": bool(e.get("stand_aside"))})

    traded = [r for r in scored if not r["stand_aside"]]
    setups = sorted({r["setup"] for r in traded})
    return {
        "n_signals": len(scored),
        "pending": pending,
        "overall": _score_bucket(traded),
        "by_direction": {d: _score_bucket([r for r in traded if r["direction"] == d])
                         for d in ("long", "short")},
        "by_setup": {s: _score_bucket([r for r in traded if r["setup"] == s])
                     for s in setups},
        "by_outcome": {o: sum(1 for r in traded if r["outcome"] == o)
                       for o in ("stop", "target", "time")},
        "stand_aside": _score_bucket([r for r in scored if r["stand_aside"]]),
    }


# ── Telegram formatting ──────────────────────────────────────────────────────

def format_swing_message(sig: SwingSignal) -> str:
    lines = [f"🎯 {sig.ticker} swing — {sig.as_of.isoformat()} close"]
    ctx = f"Close {sig.close:.2f} | trend {sig.trend.upper()}"
    if sig.rsi is not None:
        ctx += f" | RSI {sig.rsi:.0f}"
    if sig.rvol is not None:
        ctx += f" | RVOL {sig.rvol:.1f}x"
    if sig.obv_rising is not None:
        ctx += f" | OBV {'↑' if sig.obv_rising else '↓'}"
    lines.append(ctx)

    if sig.direction == "none":
        lines.append("No setup today — stand aside.")
        for e in sig.evidence:
            lines.append(f"• {e}")
    elif sig.stand_aside:
        lines.append(f"✋ {sig.setup.upper()} fired but: {sig.stand_aside_reason}")
    else:
        arrow = "🟢 BUY CALLS" if sig.direction == "long" else "🔴 BUY PUTS"
        lines.append(f"{arrow} — {sig.setup}")
        lines.append(f"Enter next open (~{sig.close:.2f} ref)")
        lines.append(f"Stop {sig.stop:.2f} (-{sig.stop_pct:.1f}%) | "
                     f"Target {sig.target:.2f} ({'+' if sig.direction == 'long' else '-'}"
                     f"{sig.target_pct:.1f}%) | R:R {sig.rr:.1f} | "
                     f"max hold {sig.max_hold_days} sessions")
        for e in sig.evidence:
            lines.append(f"• {e}")
        if sig.option:
            o = sig.option
            lines.append(f"Option: {sig.ticker} {o['expiry']} {o['strike']:g}"
                         f"{'C' if o['right'] == 'call' else 'P'} "
                         f"@ ~{o['mid']:.2f} mid (Δ{o['delta']:+.2f}, {o['dte']} DTE)")
            risk = option_risk_estimate(o, sig.close * sig.stop_pct / 100)
            if risk:
                lines.append(f"   est. loss at stop ≈ ${risk:.0f}/contract "
                             "(delta approx; max loss = debit)")
        else:
            lines.append("(no option ticket — chain unavailable, pick "
                         f"~{DTE_IDEAL} DTE Δ{TARGET_DELTA:.2f})")
    for w in sig.warnings:
        lines.append(f"⚠ {w}")
    lines.append("Exit early if the setup invalidates. Analysis only — "
                 "not financial advice.")
    return "\n".join(lines)
