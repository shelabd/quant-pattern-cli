"""
Pin butterfly recommendation engine — the "3-Day Pin Fly" strategy.

A short-dated (2-5 DTE) butterfly whose body sits on the highest
open-interest "pin" strike near spot. The engine detects price drift, scores
strikes inside a directional band by gamma-weighted open interest, and selects
the expiry whose pin OI concentration is highest.

Wing-width selection has two modes. The default **payout mode** walks the
adaptive ladder (5 → 3 → 2) until the debit fits the R:R ceiling
`width / (min_rr + 1)`, maximizing headline risk:reward (targeting 1:5).
The optional **POP mode** instead searches every symmetric width listed in
the chain and picks the one that maximizes probability of profit among
*positive-EV* flies — high-POP flies are wide (their tent covers more of the
expected-move distribution), so this trades fat headline risk:reward for a
fly that actually tends to pin, and logs NO TRADE when the best fly is below
its POP target.

Both modes rest on an expected-move model: IV-implied diffusion ⊕ a
macro-event vol bump for FOMC/CPI/PPI/NFP landing in the holding window. It
yields the 1-sigma band, probability-of-profit, and expected value used to
rank (POP mode) or annotate (payout mode) the fly; a fly whose ±1σ band
exceeds its breakeven half-width is flagged as likely to finish at a loss.

This module is pure logic: no Rich, no Click. Network access happens only
inside :func:`recommend_fly`. Everything else operates on plain DataFrames
and is unit-testable offline.

Data caveats
------------
Chains come from the options_data provider layer: CBOE's free delayed feed
by default (reliable OI, 15-min delayed quotes, server-side greeks), or
Massive (OPRA-fed NBBO) when an API key is configured, degrading to
yfinance on failure — whose ``openInterest`` is often missing on fresh
weeklies and whose ``impliedVolatility`` is a rough Black-Scholes
inversion of possibly stale quotes. Open interest is end-of-day everywhere
(OCC publishes it overnight): verify OI on your broker before entry.
Output is analysis, not financial advice — this module recommends and
never places orders.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MIN_RR = 5.0
IDEAL_RR = 15.0
DEFAULT_TARGET_POP = 0.55      # POP-mode target; engine maximizes POP among +EV flies
POP_MIN_RR = 1.0               # POP-mode floor: keeps the fly balanced (debit ≤ width/2),
                               # blocking deep-ITM "boxes" that are +EV but torch capital
POP_MAX_WIDTH_FLOOR = 12.0     # POP-mode width search caps at max(this, 2·sigma)
WIDTH_LADDER: tuple[float, ...] = (5.0, 3.0, 2.0)
MIN_WIDTH = 2.0
ROUND_NUMBER_BONUS = 1.10
DEFAULT_BAND_PCT = 1.5
DEFAULT_MIN_DTE = 2
DEFAULT_MAX_DTE = 5
DEFAULT_IV_FALLBACK = 0.25  # when the chain has no usable IV at a strike
BASE_SIZING_PCT = (0.5, 1.0)  # % of account per fly (low, high)

# Forward 1-sigma move (as a fraction of spot) a scheduled macro print
# typically injects on its release day. Heuristic, drawn from the rough
# historical one-day SPX reaction to each release. Used as a FLOOR on the
# expected move, not an additive bump: a chain whose expiry spans the event
# already carries the event's variance in its ATM IV, so adding these on top
# double-counts. The floor only binds when IV understates a known print
# (missing/stale IV falling back to DEFAULT_IV_FALLBACK, zero-DTE edge cases).
# Only events landing strictly AFTER today count (an already-printed event is
# behind us for a forward trade). Tune freely — these are not market-implied.
EVENT_VOL_ADDON: dict[str, float] = {
    "fomc": 0.011,
    "cpi": 0.009,
    "nfp": 0.007,
    "ppi": 0.004,
}


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
    select_mode: str = "payout"            # "payout" | "pop" | "fixed"
    target_pop: float = DEFAULT_TARGET_POP
    data_source: str = "Yahoo Finance"
    # Expected-move / macro-uncertainty model (informational; never alters
    # which fly is selected — see recommend_fly).
    atm_iv: Optional[float] = None              # interpolated ATM implied vol
    expected_move_pct: Optional[float] = None   # 1-sigma move to expiry, % of spot
    expected_move_dollars: Optional[float] = None  # 1-sigma move in underlying points
    em_diffusion: Optional[float] = None        # IV-implied component (points)
    em_event_pct: float = 0.0                   # macro-event add-on (% of spot)
    event_addons: list[dict] = field(default_factory=list)  # per-event breakdown
    prob_profit: Optional[float] = None         # P(settle inside breakevens)
    expected_value: Optional[float] = None      # EV per fly, dollars (net of debit)
    body_sigma: Optional[float] = None          # |body - spot| in expected-move sigmas

    def to_dict(self) -> dict:
        def per_fly(v: Optional[float]) -> Optional[float]:
            return round(v * 100, 2) if v is not None else None

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
            "select_mode": self.select_mode,
            "target_pop": round(self.target_pop, 4),
            "data_source": self.data_source,
            "atm_iv": round(self.atm_iv, 4) if self.atm_iv is not None else None,
            "expected_move_pct": round(self.expected_move_pct, 4) if self.expected_move_pct is not None else None,
            "expected_move_dollars": round(self.expected_move_dollars, 2) if self.expected_move_dollars is not None else None,
            "em_diffusion_dollars": round(self.em_diffusion, 2) if self.em_diffusion is not None else None,
            "em_event_pct": round(self.em_event_pct, 4),
            "event_addons": list(self.event_addons),
            "prob_profit": round(self.prob_profit, 4) if self.prob_profit is not None else None,
            "expected_value_per_fly": round(self.expected_value, 2) if self.expected_value is not None else None,
            "body_sigma": round(self.body_sigma, 2) if self.body_sigma is not None else None,
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
    "strike", "call_oi", "put_oi", "call_vol", "put_vol",
    "call_bid", "call_ask", "call_last",
    "put_bid", "put_ask", "put_last", "iv",
]


def normalize_chain(calls: pd.DataFrame, puts: pd.DataFrame) -> pd.DataFrame:
    """Merge yfinance-style call/put frames into one frame per strike.

    Expected input columns (per side): strike, openInterest, volume, bid,
    ask, lastPrice, impliedVolatility. Output columns: CHAIN_COLUMNS, with
    iv as the mean of the usable per-side IVs (fallback DEFAULT_IV_FALLBACK).
    Volume is carried because freshly listed weekly chains often report OI=0
    until the overnight update — scoring falls back to volume then.
    """
    def side(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        out = pd.DataFrame({
            "strike": df["strike"].astype(float),
            f"{prefix}_oi": df.get("openInterest", pd.Series(0, index=df.index)).fillna(0).astype(int),
            f"{prefix}_vol": df.get("volume", pd.Series(0, index=df.index)).fillna(0).astype(int),
            f"{prefix}_bid": df.get("bid", pd.Series(0.0, index=df.index)).fillna(0.0),
            f"{prefix}_ask": df.get("ask", pd.Series(0.0, index=df.index)).fillna(0.0),
            f"{prefix}_last": df.get("lastPrice", pd.Series(0.0, index=df.index)).fillna(0.0),
            f"{prefix}_iv": df.get("impliedVolatility", pd.Series(np.nan, index=df.index)),
        })
        return out

    merged = side(calls, "call").merge(side(puts, "put"), on="strike", how="outer")
    for col in merged.columns:
        if col.endswith("_oi") or col.endswith("_vol"):
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

    Score = total OI (calls + puts) weighted by gamma, with a +10% bonus for
    strikes divisible by 5. Gamma is the chain's server-computed `gamma`
    column where present and positive (Massive provides one); strikes
    without it get a local Black-Scholes estimate. When the entire band
    reports zero OI (yfinance leaves freshly listed weeklies at 0 until the
    overnight update) the score falls back to total volume as the
    concentration proxy; `attrs["used_volume_fallback"]` is set on the
    result.

    Returns the band subset with `total_oi`, `total_vol`, `pin_score`, and
    `oi_rank` (1 = highest raw weight) columns, sorted by pin_score
    descending. Empty frame when no strikes in band.
    """
    lo, hi = band_bounds(spot, band_pct, drift)
    band = chain[(chain["strike"] >= lo) & (chain["strike"] <= hi)].copy()
    if band.empty:
        return band

    t_years = max(dte, 1) / 365.0
    band["total_oi"] = band["call_oi"] + band["put_oi"]
    band["total_vol"] = (band["call_vol"] + band["put_vol"]
                         if "call_vol" in band.columns else 0)

    used_volume = bool(band["total_oi"].sum() == 0
                       and "call_vol" in chain.columns
                       and band["total_vol"].sum() > 0)
    weight = band["total_vol"] if used_volume else band["total_oi"]

    bs = band.apply(
        lambda r: bs_gamma(spot, r["strike"], r["iv"], t_years), axis=1)
    if "gamma" in band.columns:
        provider_gamma = pd.to_numeric(band["gamma"], errors="coerce")
        band["gamma"] = provider_gamma.where(provider_gamma > 0).fillna(bs)
    else:
        band["gamma"] = bs
    bonus = np.where(band["strike"] % 5 == 0, ROUND_NUMBER_BONUS, 1.0)
    band["pin_score"] = weight * band["gamma"] * bonus
    band["oi_rank"] = weight.rank(ascending=False, method="min").astype(int)
    band = band.sort_values("pin_score", ascending=False).reset_index(drop=True)
    band.attrs["used_volume_fallback"] = used_volume
    return band


def select_pin(
    chain: pd.DataFrame,
    spot: float,
    dte: int,
    band_pct: float = DEFAULT_BAND_PCT,
    drift: str = "neutral",
) -> Optional[dict]:
    """Top-scoring pin strike inside the band, or None when the band is empty.

    Returns {strike, pin_score, total_oi, concentration, oi_rank,
    used_volume} where concentration is the weight that scored the pin (OI,
    or volume under the zero-OI fallback).
    """
    scored = score_pins(chain, spot, dte, band_pct, drift)
    if scored.empty:
        return None
    used_volume = bool(scored.attrs.get("used_volume_fallback", False))
    top = scored.iloc[0]
    return {
        "strike": float(top["strike"]),
        "pin_score": float(top["pin_score"]),
        "total_oi": int(top["total_oi"]),
        "concentration": int(top["total_vol"] if used_volume else top["total_oi"]),
        "oi_rank": int(top["oi_rank"]),
        "used_volume": used_volume,
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


# ── Expected-move / macro-uncertainty model ──────────────────────────────────

def atm_iv(chain: pd.DataFrame, spot: float) -> float:
    """At-the-money implied vol: linear-interpolate the chain's ``iv`` at spot.

    Uses the two strikes bracketing spot; falls back to the nearest usable
    strike when spot sits outside the listed range, then to
    ``DEFAULT_IV_FALLBACK`` when the chain carries no usable IV at all.
    """
    if chain is None or chain.empty or "iv" not in chain.columns or spot <= 0:
        return DEFAULT_IV_FALLBACK
    df = chain[["strike", "iv"]].dropna()
    df = df[df["iv"] > 1e-4]
    if df.empty:
        return DEFAULT_IV_FALLBACK
    strikes = df["strike"].to_numpy(dtype=float)
    ivs = df["iv"].to_numpy(dtype=float)
    below = strikes[strikes <= spot]
    above = strikes[strikes >= spot]
    if below.size and above.size:
        k_lo, k_hi = float(below.max()), float(above.min())
        iv_lo = float(ivs[strikes == k_lo][0])
        iv_hi = float(ivs[strikes == k_hi][0])
        if k_hi == k_lo:
            return iv_lo
        w = (spot - k_lo) / (k_hi - k_lo)
        return iv_lo + w * (iv_hi - iv_lo)
    idx = int(np.argmin(np.abs(strikes - spot)))
    return float(ivs[idx])


def event_vol_addon(events, today: date, expiry: date) -> tuple[float, list[dict]]:
    """Forward macro-event vol add-on (1-sigma, as a fraction of spot).

    Sums in quadrature the :data:`EVENT_VOL_ADDON` contribution of every macro
    event landing strictly after ``today`` and on/before ``expiry``. An event
    dated today is treated as already reflected in spot/IV (its move is behind
    a forward trade), so it adds no forward uncertainty. Unknown categories are
    ignored. The result is a *floor* on the expected move (see
    :func:`expected_move`), not an additive term — the chain's IV already
    prices scheduled events. Returns ``(addon_pct, breakdown)``.
    """
    breakdown: list[dict] = []
    var = 0.0
    for evt in sorted(events, key=lambda e: e.date):
        if not (today < evt.date <= expiry):
            continue
        pct = EVENT_VOL_ADDON.get(evt.category.value)
        if pct is None:
            continue
        var += pct * pct
        breakdown.append({"name": evt.name, "date": evt.date.isoformat(), "pct": pct})
    return math.sqrt(var), breakdown


def expected_move(spot: float, iv: float, dte: int, event_pct: float = 0.0) -> dict:
    """1-sigma expected move to expiry in underlying points.

    The IV-implied diffusion ``S·IV·sqrt(DTE/365)`` is the primary estimate:
    a chain spanning a scheduled macro event already prices that event in its
    ATM IV, so the macro component ``S·event_pct`` acts only as a *floor* —
    ``total = max(diffusion, event)`` — binding when IV understates a known
    print (no usable IV, stale fallback vol, zero-DTE). Adding the two in
    quadrature would count the event's variance twice. Returns ``{diffusion,
    event, total, pct}`` (pct = total / spot).
    """
    t = max(dte, 0) / 365.0
    diffusion = spot * max(iv, 0.0) * math.sqrt(t) if t > 0 else 0.0
    event = spot * max(event_pct, 0.0)
    total = max(diffusion, event)
    return {
        "diffusion": diffusion,
        "event": event,
        "total": total,
        "pct": (total / spot) if spot > 0 else 0.0,
    }


def prob_in_profit(
    breakeven_low: float, breakeven_high: float, sigma: float, center: float,
) -> float:
    """P(settle lands between the breakevens) under N(center, sigma).

    Normal approximation, no vol skew. Degrades to a point mass when sigma≈0.
    """
    if sigma <= 1e-9:
        return 1.0 if breakeven_low <= center <= breakeven_high else 0.0
    nd = NormalDist(center, sigma)
    return max(0.0, min(1.0, nd.cdf(breakeven_high) - nd.cdf(breakeven_low)))


def fly_expected_value(
    body: float, width: float, debit: float, sigma: float, center: float,
    steps: int = 400,
) -> float:
    """Expected net P&L per share of the 1-2-1 fly under N(center, sigma).

    Numeric midpoint integration over ±5 sigma of
    ``max(0, width - |S - body|) - debit``. Multiply by 100 for per-fly
    dollars. Falls back to the point payoff when sigma≈0.
    """
    if sigma <= 1e-9:
        return max(0.0, width - abs(center - body)) - debit
    nd = NormalDist(center, sigma)
    lo, hi = center - 5 * sigma, center + 5 * sigma
    step = (hi - lo) / steps
    ev = 0.0
    for i in range(steps):
        mid = lo + (i + 0.5) * step
        payoff = max(0.0, width - abs(mid - body)) - debit
        ev += payoff * nd.pdf(mid) * step
    return ev


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


def candidate_widths(
    chain: pd.DataFrame,
    body: float,
    min_width: float = MIN_WIDTH,
    max_width: Optional[float] = None,
) -> list[float]:
    """Symmetric wing widths actually listed on both sides of the body.

    A butterfly needs both ``body - w`` and ``body + w`` to exist in the
    chain. Returns the sorted list of such widths in ``[min_width,
    max_width]`` — the search space for POP-based selection.
    """
    strikes = np.sort(np.unique(chain["strike"].to_numpy(dtype=float)))
    widths: list[float] = []
    for k in strikes:
        w = round(float(k - body), 4)
        if w < min_width:
            continue
        if max_width is not None and w > max_width + 1e-9:
            break
        if (np.any(np.isclose(strikes, body - w))
                and np.any(np.isclose(strikes, body + w))):
            widths.append(w)
    return widths


def select_width_by_pop(
    chain: pd.DataFrame,
    body: float,
    right: str,
    expiry: date,
    sigma: float,
    center: float,
    target_pop: float = DEFAULT_TARGET_POP,
    min_rr: float = POP_MIN_RR,
    max_width: Optional[float] = None,
    widths: Optional[list[float]] = None,
) -> dict:
    """Pick the wing width that maximizes probability of profit.

    High-POP flies are *wide*: their tent covers more of the move
    distribution, which makes them expensive — the opposite of the
    R:R-maximizing ladder. Every symmetric width listed in the chain is
    priced, then scored for POP and EV per fly under N(center, sigma).
    Selection is the highest-POP fly (tie-break on EV) that clears two
    guardrails:

    * *positive expected value* — without it "max POP" just keeps buying
      wider, ever-more-expensive wings, and
    * a modest *risk:reward floor* (``min_rr``, debit ≤ width/(min_rr+1)) —
      without it the search drifts into deep-ITM "boxes" whose debit ≈ width:
      technically +EV off intrinsic value, but R:R ≈ 0 and capital-torching.

    Returns {selected, attempts, scored, eligible, had_positive,
    reached_target, best_pop_attempt}. ``selected`` is None when nothing
    clears both guardrails (a NO TRADE); ``best_pop_attempt`` then carries
    the highest-POP priced candidate for the explanation. ``scored`` is
    empty only when nothing could be priced.
    """
    width_list = (widths if widths is not None
                  else candidate_widths(chain, body, max_width=max_width))
    attempts: list[dict] = []
    scored: list[dict] = []
    for width in width_list:
        try:
            debit, legs = price_fly(chain, body, width, right, expiry)
        except UnpriceableStrikeError as e:
            attempts.append({"width": width, "result": f"unpriceable: {e}"})
            continue
        if debit <= 0:
            attempts.append({"width": width,
                             "result": f"non-positive mid debit {debit:.2f} (stale quotes)"})
            continue
        ratio = evaluate_ratio(debit, width, body)
        pop = prob_in_profit(ratio["breakeven_low"], ratio["breakeven_high"], sigma, center)
        ev = fly_expected_value(body, width, debit, sigma, center) * 100
        attempts.append({"width": width,
                         "result": f"POP {pop * 100:.0f}%  EV ${ev:+.0f}  RR 1:{ratio['risk_reward']:.1f}"})
        scored.append({"width": width, "debit": debit, "legs": legs,
                       "ratio": ratio, "pop": pop, "ev": ev})

    if not scored:
        return {"selected": None, "attempts": attempts, "scored": [],
                "eligible": [], "had_positive": False, "reached_target": False,
                "best_pop_attempt": None}

    best_pop_attempt = max(scored, key=lambda c: c["pop"])
    eligible = [c for c in scored
                if c["ev"] > 0 and c["ratio"]["risk_reward"] >= min_rr]
    if not eligible:
        return {"selected": None, "attempts": attempts, "scored": scored,
                "eligible": [], "had_positive": any(c["ev"] > 0 for c in scored),
                "reached_target": False, "best_pop_attempt": best_pop_attempt}
    best = max(eligible, key=lambda c: (c["pop"], c["ev"]))
    return {"selected": best, "attempts": attempts, "scored": scored,
            "eligible": eligible, "had_positive": True,
            "reached_target": best["pop"] >= target_pop,
            "best_pop_attempt": best_pop_attempt}


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
        if (pin["concentration"], -dte) > (best["pin"]["concentration"], -best["dte"]):
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
    chain_source: str = "auto",
    select: str = "payout",
    target_pop: float = DEFAULT_TARGET_POP,
) -> FlyRecommendation:
    """End-to-end pin-fly recommendation against live data.

    The only function in this module that touches the network. OHLCV (for
    drift and spot) goes through data.py's provider; option chains go
    through options_data's provider layer — Massive when an API key is
    configured, else CBOE's free delayed feed. A chain-source failure
    degrades to yfinance with a warning instead of erroring.
    """
    from datetime import timedelta
    from .data import get_provider
    from .events import EventCatalog
    from .options_data import fetch_chains_with_fallback, get_options_provider

    ticker = ticker.upper()
    today = today or date.today()

    # ── Spot + drift from provider OHLCV ──────────────────────────────
    provider = get_provider("yfinance")
    try:
        history = provider.get_daily_ohlcv(ticker, today - timedelta(days=90), today)
    except ValueError:
        # Index tickers (SPX, VIX, RUT) live under a caret on Yahoo.
        history = provider.get_daily_ohlcv(f"^{ticker}", today - timedelta(days=90), today)
    close = history["Close"]
    spot = float(close.iloc[-1])
    drift = detect_drift(close)

    # ── Candidate expiries + chains via the options provider ─────────
    if expiry_override is not None:
        window = (expiry_override, expiry_override)
    else:
        window = (today + timedelta(days=min_dte), today + timedelta(days=max_dte))

    oprov, candidates, source_warnings = fetch_chains_with_fallback(
        get_options_provider(chain_source), ticker, *window)

    base = dict(
        ticker=ticker, spot=spot, drift=drift, right="",
        expiry=None, dte=None, body_strike=None,
        selected_width=None, width_was_adaptive=fixed_width is None,
        account_size=account, min_rr=min_rr,
        select_mode="fixed" if fixed_width is not None else select,
        target_pop=target_pop, data_source=oprov.name(),
    )

    if not candidates:
        if expiry_override is not None:
            rec = _no_trade(base, f"expiry {expiry_override} not listed for {ticker}")
        else:
            rec = _no_trade(base, f"no listed expiries within {min_dte}-{max_dte} DTE")
        rec.warnings = source_warnings
        return rec

    # ── OI-aware expiry + pin selection ───────────────────────────────
    choice = choose_expiry(candidates, spot, today, band_pct, drift)
    if choice is None:
        rec = _no_trade(base, f"no strikes inside the {band_pct}% {drift} band on any candidate expiry")
        rec.warnings = source_warnings
        return rec

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
    oi_warnings: list[str] = list(source_warnings)
    if pin["used_volume"]:
        oi_warnings.append(
            "Chain reports zero open interest across the band (OI updates "
            "overnight) — pin scored by today's volume instead. "
            "Verify the OI wall on your broker before trusting this pin."
        )

    # ── Expected-move / macro-uncertainty model (informational only) ──
    # Computed up front so a NO-TRADE rec still carries the band that
    # explains why nothing fit. Never alters which fly is selected.
    catalog = EventCatalog()
    macro_events = [e for e in catalog.events
                    if e.category.value in MACRO_EVENT_CATEGORIES]
    in_window = [e for e in macro_events if today <= e.date <= expiry]
    coverage = max((e.date for e in macro_events), default=None)
    atm = atm_iv(chain, spot)
    event_pct, event_breakdown = event_vol_addon(in_window, today, expiry)
    em = expected_move(spot, atm, dte, event_pct)
    em_fields = dict(
        atm_iv=atm,
        expected_move_pct=em["pct"],
        expected_move_dollars=em["total"],
        em_diffusion=em["diffusion"],
        em_event_pct=event_pct,
        event_addons=event_breakdown,
        body_sigma=(abs(body - spot) / em["total"]) if em["total"] > 0 else None,
    )

    # ── Width selection ───────────────────────────────────────────────
    # POP mode (default) searches every listed width for the highest-POP
    # positive-EV fly; payout mode (and any explicit --width) walks the R:R
    # ladder against the debit ceiling. An explicit fixed_width always keeps
    # its R:R-ceiling semantics, so it routes through the ladder.
    sigma = em["total"]
    if fixed_width is None and select == "pop":
        sel = select_width_by_pop(
            chain, body, right, expiry, sigma, spot,
            target_pop=target_pop, min_rr=POP_MIN_RR,
            max_width=max(POP_MAX_WIDTH_FLOOR, 2 * sigma),
        )
        chosen = sel["selected"]
        if chosen is None:
            bp = sel["best_pop_attempt"]
            if bp is None:
                reason = "no priceable symmetric butterfly around the pin"
            elif sel["had_positive"]:
                reason = (
                    "no balanced butterfly clears the 1:"
                    f"{POP_MIN_RR:g} R:R floor — positive-EV widths here are "
                    "deep-ITM boxes (debit ≈ width), not pin flies"
                )
            else:
                reason = (
                    "no positive-EV butterfly at any listed wing width — the "
                    f"±{em['pct'] * 100:.1f}% expected move dominates every fly "
                    f"(widest profitable POP {bp['pop'] * 100:.0f}% was still −EV)"
                )
            rec = _no_trade(base, reason)
            rec.width_attempts = sel["attempts"]
            rec.body_oi = pin["total_oi"]
            rec.band_rank = pin["oi_rank"]
            rec.expiry_pin_oi = pin["concentration"]
            rec.warnings = oi_warnings
            for k, v in em_fields.items():
                setattr(rec, k, v)
            return rec
        if not sel["reached_target"]:
            # The best balanced +EV fly still falls short of the POP target —
            # in this regime the expected move is too large for a high-POP pin
            # fly, so log NO TRADE rather than recommend a sub-target one.
            best = chosen
            reason = (
                f"best balanced fly reaches only POP {best['pop'] * 100:.0f}% "
                f"(target {target_pop:.0%}) at width {best['width']:g} — the "
                f"±{em['pct'] * 100:.1f}% expected move is too large for a "
                f"high-POP fly at {dte} DTE"
            )
            rec = _no_trade(base, reason)
            rec.width_attempts = sel["attempts"]
            rec.body_oi = pin["total_oi"]
            rec.band_rank = pin["oi_rank"]
            rec.expiry_pin_oi = pin["concentration"]
            rec.warnings = oi_warnings
            for k, v in em_fields.items():
                setattr(rec, k, v)
            return rec
        width, debit, legs = chosen["width"], chosen["debit"], chosen["legs"]
        ratio = chosen["ratio"]
        attempts = sel["attempts"]
        adaptive = True
        ceiling = None
        pop, ev_per_fly = chosen["pop"], chosen["ev"]
    else:
        result = adaptive_width(chain, body, right, expiry, min_rr, fixed_width)
        if result["selected_width"] is None:
            rec = _no_trade(base, f"no wing width meets the 1:{min_rr:g} debit ceiling")
            rec.width_attempts = result["attempts"]
            rec.body_oi = pin["total_oi"]
            rec.band_rank = pin["oi_rank"]
            rec.expiry_pin_oi = pin["concentration"]
            rec.warnings = oi_warnings
            for k, v in em_fields.items():
                setattr(rec, k, v)
            return rec
        width, debit, legs = result["selected_width"], result["debit"], result["legs"]
        ratio = evaluate_ratio(debit, width, body)
        attempts = result["attempts"]
        adaptive = result["adaptive"]
        ceiling = max_debit_for(width, min_rr)
        pop = prob_in_profit(ratio["breakeven_low"], ratio["breakeven_high"], sigma, spot)
        ev_per_fly = fly_expected_value(body, width, debit, sigma, spot) * 100

    # ── Event skip rule ───────────────────────────────────────────────
    warnings, half_size = event_warnings(in_window, coverage, today, expiry)
    warnings = oi_warnings + warnings

    # ── Expected-move warning (POP mode already optimized against it) ──
    be_half_width = width - debit
    if sigma > be_half_width:
        warnings.append(
            f"±1σ expected move (±{em['pct'] * 100:.1f}% / ±${sigma:.2f}) "
            f"exceeds the fly's breakeven half-width (${be_half_width:.2f}) — a "
            f"typical move finishes this fly at a loss (POP ≈ {pop * 100:.0f}%)."
        )

    sizing = (BASE_SIZING_PCT[0] / 2, BASE_SIZING_PCT[1] / 2) if half_size else BASE_SIZING_PCT
    limit = floor_to_cent(debit if ceiling is None else min(debit, ceiling))

    rec = FlyRecommendation(
        **base,
        legs=legs,
        debit=debit,
        max_profit=ratio["max_profit"],
        risk_reward=ratio["risk_reward"],
        breakeven_low=ratio["breakeven_low"],
        breakeven_high=ratio["breakeven_high"],
        limit_price=limit,
        max_debit_ceiling=ceiling,
        body_oi=pin["total_oi"],
        band_rank=pin["oi_rank"],
        expiry_pin_oi=pin["concentration"],
        verdict="PASS",
        warnings=warnings,
        width_attempts=attempts,
        sizing_pct=sizing,
        prob_profit=pop,
        expected_value=ev_per_fly,
        **em_fields,
    )
    rec.selected_width = width
    rec.width_was_adaptive = adaptive
    return rec
