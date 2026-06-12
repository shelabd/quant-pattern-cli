"""
Pin butterfly recommendation engine — the "3-Day Pin Fly" strategy.

A short-dated (2-5 DTE) butterfly whose body sits on the highest
open-interest "pin" strike near spot, targeting a structural risk:reward of
at least 1:12 (ideal 1:15). The engine detects price drift, scores strikes
inside a directional band by gamma-weighted open interest, selects the
expiry whose pin OI concentration is highest, and walks an adaptive wing
width (5 → 3 → 2) until the debit fits the ceiling `width / (min_rr + 1)`.

This module is pure logic: no Rich, no Click. Network access happens only
inside :func:`recommend_fly`. Everything else operates on plain DataFrames
and is unit-testable offline.

Data caveats
------------
yfinance ``openInterest`` is end-of-day stale (updated overnight) and
``impliedVolatility`` is a rough Black-Scholes inversion of possibly stale
quotes. Treat pin scores as indicative; verify OI on your broker before
entry. Output is analysis, not financial advice — this module recommends
and never places orders.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MIN_RR = 12.0
IDEAL_RR = 15.0
WIDTH_LADDER: tuple[float, ...] = (5.0, 3.0, 2.0)
MIN_WIDTH = 2.0
ROUND_NUMBER_BONUS = 1.10
DEFAULT_BAND_PCT = 1.5
DEFAULT_MIN_DTE = 2
DEFAULT_MAX_DTE = 5
DEFAULT_IV_FALLBACK = 0.25  # when the chain has no usable IV at a strike
BASE_SIZING_PCT = (0.5, 1.0)  # % of account per fly (low, high)


class UnpriceableStrikeError(ValueError):
    """A required leg has no bid/ask and no last price — never price as 0."""


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class Leg:
    """One leg of the butterfly order ticket."""
    action: str      # "BUY" or "SELL"
    quantity: int    # per fly: 1 / 2 / 1
    right: str       # "CALL" or "PUT"
    strike: float
    expiry: date
    mid: float       # per-share mid price used for the combo debit

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "quantity": self.quantity,
            "right": self.right,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "mid": round(self.mid, 4),
        }


@dataclass
class FlyRecommendation:
    """Complete pin-fly recommendation (or a NO TRADE with reasons)."""
    ticker: str
    spot: float
    drift: str                 # "bullish" | "bearish" | "neutral"
    right: str                 # "CALL" | "PUT"
    expiry: Optional[date]
    dte: Optional[int]
    body_strike: Optional[float]
    selected_width: Optional[float]
    width_was_adaptive: bool
    legs: list[Leg] = field(default_factory=list)
    debit: Optional[float] = None          # per-share combo mid
    max_profit: Optional[float] = None     # per-share
    risk_reward: Optional[float] = None
    breakeven_low: Optional[float] = None
    breakeven_high: Optional[float] = None
    limit_price: Optional[float] = None    # per-share, floored to $0.01
    max_debit_ceiling: Optional[float] = None
    body_oi: int = 0
    band_rank: int = 0                     # body's raw-OI rank inside the band (1 = highest)
    expiry_pin_oi: int = 0                 # pin OI concentration that won expiry selection
    verdict: str = "NO TRADE"              # "PASS" | "NO TRADE"
    no_trade_reason: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    width_attempts: list[dict] = field(default_factory=list)
    sizing_pct: tuple[float, float] = BASE_SIZING_PCT
    account_size: Optional[float] = None
    min_rr: float = DEFAULT_MIN_RR

    def to_dict(self) -> dict:
        per_fly = (lambda v: round(v * 100, 2) if v is not None else None)
        d = {
            "ticker": self.ticker,
            "spot": round(self.spot, 2),
            "drift": self.drift,
            "right": self.right,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "dte": self.dte,
            "body_strike": self.body_strike,
            "selected_width": self.selected_width,
            "width_was_adaptive": self.width_was_adaptive,
            "legs": [leg.to_dict() for leg in self.legs],
            "debit_per_share": round(self.debit, 4) if self.debit is not None else None,
            "debit_per_fly": per_fly(self.debit),
            "max_profit_per_fly": per_fly(self.max_profit),
            "risk_reward": round(self.risk_reward, 2) if self.risk_reward is not None else None,
            "breakeven_low": round(self.breakeven_low, 2) if self.breakeven_low is not None else None,
            "breakeven_high": round(self.breakeven_high, 2) if self.breakeven_high is not None else None,
            "limit_price_per_share": self.limit_price,
            "limit_price_per_fly": per_fly(self.limit_price),
            "max_debit_ceiling_per_share": round(self.max_debit_ceiling, 4) if self.max_debit_ceiling is not None else None,
            "body_oi": self.body_oi,
            "band_rank": self.band_rank,
            "expiry_pin_oi": self.expiry_pin_oi,
            "verdict": self.verdict,
            "no_trade_reason": self.no_trade_reason,
            "warnings": list(self.warnings),
            "width_attempts": list(self.width_attempts),
            "sizing_pct": {"low": self.sizing_pct[0], "high": self.sizing_pct[1]},
            "account_size": self.account_size,
            "min_rr": self.min_rr,
            "disclaimer": "Analysis only — not financial advice. OI as of last close; verify on your broker before entry.",
        }
        return d


# ── Drift detection ──────────────────────────────────────────────────────────

def detect_drift(close: pd.Series) -> str:
    """Classify short-term drift from 5/20 EMA alignment + 3-session momentum.

    Both signals must agree for a directional call; anything else is neutral.
    """
    if close is None or len(close) < 21:
        return "neutral"

    ema5 = close.ewm(span=5, adjust=False).mean().iloc[-1]
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    momentum = close.iloc[-1] - close.iloc[-4]  # 3 sessions back

    if ema5 > ema20 and momentum > 0:
        return "bullish"
    if ema5 < ema20 and momentum < 0:
        return "bearish"
    return "neutral"


# ── Black-Scholes gamma ──────────────────────────────────────────────────────

def bs_gamma(spot: float, strike: float, iv: float, t_years: float) -> float:
    """Black-Scholes gamma (same for calls and puts), zero rate/dividend.

    gamma = phi(d1) / (S * sigma * sqrt(T))
    """
    if spot <= 0 or strike <= 0 or iv <= 0 or t_years <= 0:
        return 0.0
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (iv * iv / 2) * t_years) / (iv * sqrt_t)
    phi = math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)
    return phi / (spot * iv * sqrt_t)


# ── Chain normalization ──────────────────────────────────────────────────────

CHAIN_COLUMNS = [
    "strike", "call_oi", "put_oi",
    "call_bid", "call_ask", "call_last",
    "put_bid", "put_ask", "put_last", "iv",
]


def normalize_chain(calls: pd.DataFrame, puts: pd.DataFrame) -> pd.DataFrame:
    """Merge yfinance-style call/put frames into one frame per strike.

    Expected input columns (per side): strike, openInterest, bid, ask,
    lastPrice, impliedVolatility. Output columns: CHAIN_COLUMNS, with iv as
    the mean of the usable per-side IVs (fallback DEFAULT_IV_FALLBACK).
    """
    def side(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        out = pd.DataFrame({
            "strike": df["strike"].astype(float),
            f"{prefix}_oi": df.get("openInterest", pd.Series(0, index=df.index)).fillna(0).astype(int),
            f"{prefix}_bid": df.get("bid", pd.Series(0.0, index=df.index)).fillna(0.0),
            f"{prefix}_ask": df.get("ask", pd.Series(0.0, index=df.index)).fillna(0.0),
            f"{prefix}_last": df.get("lastPrice", pd.Series(0.0, index=df.index)).fillna(0.0),
            f"{prefix}_iv": df.get("impliedVolatility", pd.Series(np.nan, index=df.index)),
        })
        return out

    merged = side(calls, "call").merge(side(puts, "put"), on="strike", how="outer")
    for col in merged.columns:
        if col.endswith("_oi"):
            merged[col] = merged[col].fillna(0).astype(int)
        elif col != "strike" and not col.endswith("_iv"):
            merged[col] = merged[col].fillna(0.0)

    ivs = merged[["call_iv", "put_iv"]].where(lambda x: x > 1e-4)
    merged["iv"] = ivs.mean(axis=1, skipna=True).fillna(DEFAULT_IV_FALLBACK)
    merged = merged.drop(columns=["call_iv", "put_iv"])
    return merged.sort_values("strike").reset_index(drop=True)[CHAIN_COLUMNS]


# ── Pin detection ────────────────────────────────────────────────────────────

def band_bounds(spot: float, band_pct: float, drift: str) -> tuple[float, float]:
    """Strike band in the drift direction: [0%, +band%] above spot when
    bullish, below when bearish, symmetric when neutral."""
    width = spot * band_pct / 100
    if drift == "bullish":
        return spot, spot + width
    if drift == "bearish":
        return spot - width, spot
    return spot - width, spot + width


def score_pins(
    chain: pd.DataFrame,
    spot: float,
    dte: int,
    band_pct: float = DEFAULT_BAND_PCT,
    drift: str = "neutral",
) -> pd.DataFrame:
    """Score every strike inside the drift band.

    Score = total OI (calls + puts) weighted by Black-Scholes gamma, with a
    +10% bonus for strikes divisible by 5. Returns the band subset with
    `total_oi`, `pin_score`, and `oi_rank` (1 = highest raw OI) columns,
    sorted by pin_score descending. Empty frame when no strikes in band.
    """
    lo, hi = band_bounds(spot, band_pct, drift)
    band = chain[(chain["strike"] >= lo) & (chain["strike"] <= hi)].copy()
    if band.empty:
        return band

    t_years = max(dte, 1) / 365.0
    band["total_oi"] = band["call_oi"] + band["put_oi"]
    band["gamma"] = band.apply(
        lambda r: bs_gamma(spot, r["strike"], r["iv"], t_years), axis=1)
    bonus = np.where(band["strike"] % 5 == 0, ROUND_NUMBER_BONUS, 1.0)
    band["pin_score"] = band["total_oi"] * band["gamma"] * bonus
    band["oi_rank"] = band["total_oi"].rank(ascending=False, method="min").astype(int)
    return band.sort_values("pin_score", ascending=False).reset_index(drop=True)


def select_pin(
    chain: pd.DataFrame,
    spot: float,
    dte: int,
    band_pct: float = DEFAULT_BAND_PCT,
    drift: str = "neutral",
) -> Optional[dict]:
    """Top-scoring pin strike inside the band, or None when the band is empty.

    Returns {strike, pin_score, total_oi, oi_rank}.
    """
    scored = score_pins(chain, spot, dte, band_pct, drift)
    if scored.empty:
        return None
    top = scored.iloc[0]
    return {
        "strike": float(top["strike"]),
        "pin_score": float(top["pin_score"]),
        "total_oi": int(top["total_oi"]),
        "oi_rank": int(top["oi_rank"]),
    }


# ── Pricing ──────────────────────────────────────────────────────────────────

def mid_price(bid: float, ask: float, last: float) -> float:
    """(bid+ask)/2 when both sides are live, else lastPrice, else raise.

    Never silently treats a missing quote as $0 — a fly priced off a phantom
    zero leg looks like free money and isn't.
    """
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    if last > 0:
        return last
    raise UnpriceableStrikeError("no bid/ask and no last price")


def price_fly(
    chain: pd.DataFrame,
    body: float,
    width: float,
    right: str,
    expiry: date,
) -> tuple[float, list[Leg]]:
    """Price a 1-2-1 butterfly from chain mids.

    Returns (debit per share, legs). Raises UnpriceableStrikeError when a
    wing strike is missing from the chain or a leg has no usable quote.
    """
    right = right.upper()
    prefix = "call" if right == "CALL" else "put"
    strikes_needed = [body - width, body, body + width]

    mids: dict[float, float] = {}
    for k in strikes_needed:
        row = chain[np.isclose(chain["strike"], k)]
        if row.empty:
            raise UnpriceableStrikeError(f"strike {k} not in chain")
        r = row.iloc[0]
        try:
            mids[k] = mid_price(r[f"{prefix}_bid"], r[f"{prefix}_ask"], r[f"{prefix}_last"])
        except UnpriceableStrikeError as e:
            raise UnpriceableStrikeError(f"strike {k}: {e}") from e

    debit = mids[body - width] - 2 * mids[body] + mids[body + width]
    legs = [
        Leg("BUY", 1, right, body - width, expiry, mids[body - width]),
        Leg("SELL", 2, right, body, expiry, mids[body]),
        Leg("BUY", 1, right, body + width, expiry, mids[body + width]),
    ]
    return debit, legs


# ── Ratio / ceiling arithmetic ───────────────────────────────────────────────

def max_debit_for(width: float, min_rr: float = DEFAULT_MIN_RR) -> float:
    """Debit ceiling for a structural risk:reward of at least 1:min_rr.

    max_profit = width - debit; requiring max_profit/debit >= min_rr gives
    debit <= width / (min_rr + 1).
    """
    return width / (min_rr + 1)


def evaluate_ratio(debit: float, width: float, body: float) -> dict:
    """Max profit, risk:reward, and breakevens for a fly at a given debit.

    Breakevens are body ∓ (width − debit).
    """
    if debit <= 0:
        raise ValueError("debit must be positive (credit flies are a data error here)")
    max_profit = width - debit
    return {
        "max_profit": max_profit,
        "risk_reward": max_profit / debit,
        "breakeven_low": body - (width - debit),
        "breakeven_high": body + (width - debit),
    }


def floor_to_cent(value: float) -> float:
    """Floor to $0.01 so a rounded limit price can never exceed the ceiling."""
    return math.floor(value * 100 + 1e-9) / 100


# ── Adaptive wing width ──────────────────────────────────────────────────────

def adaptive_width(
    chain: pd.DataFrame,
    body: float,
    right: str,
    expiry: date,
    min_rr: float = DEFAULT_MIN_RR,
    fixed_width: Optional[float] = None,
) -> dict:
    """Walk the width ladder (5 → 3 → 2) until the mid debit fits the ceiling.

    A fixed_width disables the ladder and tries only that width. Returns
    {selected_width, debit, legs, attempts, adaptive} with selected_width
    None when nothing passes; attempts records why each width failed.
    """
    widths = (fixed_width,) if fixed_width is not None else WIDTH_LADDER
    attempts: list[dict] = []

    for width in widths:
        if width < MIN_WIDTH and fixed_width is None:
            continue
        ceiling = max_debit_for(width, min_rr)
        try:
            debit, legs = price_fly(chain, body, width, right, expiry)
        except UnpriceableStrikeError as e:
            attempts.append({"width": width, "result": f"unpriceable: {e}"})
            continue
        if debit <= 0:
            attempts.append({"width": width, "result": f"non-positive mid debit {debit:.2f} (stale quotes)"})
            continue
        if debit <= ceiling:
            attempts.append({"width": width, "result": f"debit {debit:.2f} <= ceiling {ceiling:.2f} — selected"})
            return {
                "selected_width": width,
                "debit": debit,
                "legs": legs,
                "attempts": attempts,
                "adaptive": fixed_width is None,
            }
        attempts.append({"width": width, "result": f"debit {debit:.2f} > ceiling {ceiling:.2f}"})

    return {"selected_width": None, "debit": None, "legs": [],
            "attempts": attempts, "adaptive": fixed_width is None}


# ── OI-aware expiry selection ────────────────────────────────────────────────

def choose_expiry(
    candidates: list[tuple[date, pd.DataFrame]],
    spot: float,
    today: date,
    band_pct: float = DEFAULT_BAND_PCT,
    drift: str = "neutral",
) -> Optional[dict]:
    """Pick the expiry whose best pin strike carries the most open interest.

    The pin only exists where the OI lives: a 2-DTE chain with a
    22K-contract wall beats a 5-DTE chain with 2K. Ties break toward
    shorter DTE. Returns {expiry, dte, chain, pin} or None when no
    candidate has a strike in band.
    """
    best: Optional[dict] = None
    for expiry, chain in candidates:
        dte = (expiry - today).days
        pin = select_pin(chain, spot, dte, band_pct, drift)
        if pin is None:
            continue
        entry = {"expiry": expiry, "dte": dte, "chain": chain, "pin": pin}
        if best is None:
            best = entry
            continue
        if (pin["total_oi"], -dte) > (best["pin"]["total_oi"], -best["dte"]):
            best = entry
    return best


# ── Event skip rule ──────────────────────────────────────────────────────────

MACRO_EVENT_CATEGORIES = ("cpi", "ppi", "fomc", "nfp")


def event_warnings(
    events_in_window: list,
    calendar_coverage: Optional[date],
    today: date,
    expiry: date,
) -> tuple[list[str], bool]:
    """Event-skip warnings for the holding window [today, expiry].

    events_in_window: MarketEvent-like objects (need .name and .date) dated
    inside the window. calendar_coverage: the latest macro event date known
    to the calendar — when it doesn't reach the expiry we can't rule events
    out, so degrade with an "incomplete" notice instead of silence.

    Returns (warnings, half_size) where half_size is True when an actual
    event sits inside the window.
    """
    warnings: list[str] = []
    half_size = False

    for evt in sorted(events_in_window, key=lambda e: e.date):
        if today <= evt.date <= expiry:
            warnings.append(
                f"{evt.name} on {evt.date} falls inside the holding window — "
                "enter after the print or skip; half size at most."
            )
            half_size = True

    if calendar_coverage is None or calendar_coverage < expiry:
        warnings.append(
            "Event calendar incomplete for this window — verify CPI/PPI/FOMC/NFP "
            "dates manually (run: qpat events sync-calendar)."
        )

    return warnings, half_size


# ── Orchestrator (network lives here only) ───────────────────────────────────

def _no_trade(rec_kwargs: dict, reason: str) -> "FlyRecommendation":
    rec = FlyRecommendation(**rec_kwargs)
    rec.verdict = "NO TRADE"
    rec.no_trade_reason = reason
    return rec


def recommend_fly(
    ticker: str,
    min_rr: float = DEFAULT_MIN_RR,
    band_pct: float = DEFAULT_BAND_PCT,
    min_dte: int = DEFAULT_MIN_DTE,
    max_dte: int = DEFAULT_MAX_DTE,
    fixed_width: Optional[float] = None,
    account: Optional[float] = None,
    expiry_override: Optional[date] = None,
    today: Optional[date] = None,
) -> FlyRecommendation:
    """End-to-end pin-fly recommendation against live yfinance data.

    The only function in this module that touches the network. OHLCV (for
    drift and spot) goes through data.py's provider; option chains use
    yfinance directly.
    TODO: extend the DataProvider ABC with an options-chain method so IBKR
    can serve fresher OI than yfinance's end-of-day snapshot.
    """
    import yfinance as yf
    from datetime import timedelta
    from .data import get_provider
    from .events import EventCatalog, EventCategory

    ticker = ticker.upper()
    today = today or date.today()

    # ── Spot + drift from provider OHLCV ──────────────────────────────
    provider = get_provider("yfinance")
    history = provider.get_daily_ohlcv(ticker, today - timedelta(days=90), today)
    close = history["Close"]
    spot = float(close.iloc[-1])
    drift = detect_drift(close)

    base = dict(
        ticker=ticker, spot=spot, drift=drift, right="",
        expiry=None, dte=None, body_strike=None,
        selected_width=None, width_was_adaptive=fixed_width is None,
        account_size=account, min_rr=min_rr,
    )

    # ── Candidate expiries ────────────────────────────────────────────
    tk = yf.Ticker(ticker)
    all_expiries = [date.fromisoformat(e) for e in (tk.options or ())]
    if expiry_override is not None:
        candidates_dates = [e for e in all_expiries if e == expiry_override]
        if not candidates_dates:
            return _no_trade(base, f"expiry {expiry_override} not listed for {ticker}")
    else:
        candidates_dates = [e for e in all_expiries
                            if min_dte <= (e - today).days <= max_dte]
        if not candidates_dates:
            return _no_trade(base, f"no listed expiries within {min_dte}-{max_dte} DTE")

    candidates: list[tuple[date, pd.DataFrame]] = []
    for exp in candidates_dates:
        try:
            oc = tk.option_chain(exp.isoformat())
            candidates.append((exp, normalize_chain(oc.calls, oc.puts)))
        except Exception as e:
            logger.warning(f"Could not load chain for {exp}: {e}")
    if not candidates:
        return _no_trade(base, "no option chains could be loaded")

    # ── OI-aware expiry + pin selection ───────────────────────────────
    choice = choose_expiry(candidates, spot, today, band_pct, drift)
    if choice is None:
        return _no_trade(base, f"no strikes inside the {band_pct}% {drift} band on any candidate expiry")

    expiry, dte, chain, pin = choice["expiry"], choice["dte"], choice["chain"], choice["pin"]
    body = pin["strike"]

    # Drift decides the right: put fly when bearish, or neutral with the pin
    # at/below spot; call fly otherwise.
    if drift == "bearish" or (drift == "neutral" and body <= spot):
        right = "PUT"
    else:
        right = "CALL"

    base.update(right=right, expiry=expiry, dte=dte, body_strike=body)

    # ── Adaptive width / pricing ──────────────────────────────────────
    result = adaptive_width(chain, body, right, expiry, min_rr, fixed_width)
    if result["selected_width"] is None:
        rec = _no_trade(base, f"no wing width meets the 1:{min_rr:g} debit ceiling")
        rec.width_attempts = result["attempts"]
        rec.body_oi = pin["total_oi"]
        rec.band_rank = pin["oi_rank"]
        rec.expiry_pin_oi = pin["total_oi"]
        return rec

    width, debit, legs = result["selected_width"], result["debit"], result["legs"]
    ceiling = max_debit_for(width, min_rr)
    ratio = evaluate_ratio(debit, width, body)

    # ── Event skip rule ───────────────────────────────────────────────
    catalog = EventCatalog()
    macro_events = [e for e in catalog.events
                    if e.category.value in MACRO_EVENT_CATEGORIES]
    in_window = [e for e in macro_events if today <= e.date <= expiry]
    coverage = max((e.date for e in macro_events), default=None)
    warnings, half_size = event_warnings(in_window, coverage, today, expiry)

    sizing = (BASE_SIZING_PCT[0] / 2, BASE_SIZING_PCT[1] / 2) if half_size else BASE_SIZING_PCT

    rec = FlyRecommendation(
        **base,
        legs=legs,
        debit=debit,
        max_profit=ratio["max_profit"],
        risk_reward=ratio["risk_reward"],
        breakeven_low=ratio["breakeven_low"],
        breakeven_high=ratio["breakeven_high"],
        limit_price=floor_to_cent(min(debit, ceiling)),
        max_debit_ceiling=ceiling,
        body_oi=pin["total_oi"],
        band_rank=pin["oi_rank"],
        expiry_pin_oi=pin["total_oi"],
        verdict="PASS",
        warnings=warnings,
        width_attempts=result["attempts"],
        sizing_pct=sizing,
    )
    rec.selected_width = width
    rec.width_was_adaptive = result["adaptive"]
    return rec
