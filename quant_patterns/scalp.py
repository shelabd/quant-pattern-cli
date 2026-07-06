"""Intraday scalp levels: a floor and ceiling for the current session.

Pure logic, no Rich/Click/network. Levels come from three families of
evidence, clustered into a single actionable floor and ceiling:

* **OI walls** — the nearest-expiry option chain's largest put-OI strike
  below spot (floor candidate) and largest call-OI strike above spot
  (ceiling candidate). Dealer hedging around big walls tends to slow price;
  OI is as of last close (OCC publishes overnight), like everywhere else.
* **IV expected move** — the ATM-IV 1-sigma move over the *remaining*
  session minutes, banded around spot. Shrinks toward zero into the close.
* **Price structure** — session VWAP, opening range (first 30 minutes),
  today's high/low, prior session high/low/close.

Candidates on the same side of spot that sit within CLUSTER_PCT of each
other merge into one level; the strongest cluster wins. When a cluster
contains an OI wall the level snaps to the wall's strike — strikes are
where the actual hedging flow sits.

Analysis only — not financial advice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)
SESSION_MINUTES = 390
TRADING_MINUTES_PER_YEAR = 252 * SESSION_MINUTES

# Candidates within this % of each other merge into one level.
CLUSTER_PCT = 0.15
# How far from spot (in %) OI walls are searched. Wider than the fly's pin
# band — intraday ranges regularly stretch past 1.5%.
WALL_BAND_PCT = 2.5

# Trade-plan geometry (all % of spot). Stops sit beyond the level so a
# tag-and-reject doesn't shake the position out; entries are a zone, not a
# tick, because levels are cluster centers with CLUSTER_PCT resolution.
ENTRY_ZONE_PCT = 0.05
STOP_BUFFER_PCT = 0.10
# Below this reward:risk to the far target the range is too tight to scalp.
MIN_RR = 1.5

# Source weights: OI walls anchor the structure, the shrinking IV band and
# VWAP carry real-time information, static session levels confirm.
SOURCE_WEIGHTS = {
    "put wall": 3.0,
    "call wall": 3.0,
    "expected move": 2.0,
    "VWAP": 2.0,
    "opening range low": 1.5,
    "opening range high": 1.5,
    "session low": 1.5,
    "session high": 1.5,
    "prior low": 1.0,
    "prior high": 1.0,
    "prior close": 1.0,
}


@dataclass
class LevelCandidate:
    price: float
    source: str          # key into SOURCE_WEIGHTS
    detail: str = ""     # e.g. "18,240 OI @ 744"

    @property
    def weight(self) -> float:
        return SOURCE_WEIGHTS.get(self.source, 1.0)


@dataclass
class ScalpSetup:
    """A mechanical entry/exit plan hung off one of the levels.

    ``trigger`` is the level itself; the entry zone brackets it because
    levels are cluster centers, not ticks. R-multiples are measured from
    the trigger against the stop. A non-empty ``skip_reason`` means the
    geometry doesn't pay — stand aside rather than force it.
    """
    side: str                     # "long" | "short"
    trigger: float                # the level the plan hangs off
    trigger_label: str            # "floor bounce" / "ceiling fade"
    entry_lo: float
    entry_hi: float
    stop: float
    target1: float
    target1_label: str
    rr1: float
    target2: Optional[float] = None
    target2_label: str = ""
    rr2: Optional[float] = None
    with_trend: bool = True       # side agrees with spot-vs-VWAP
    notes: list[str] = field(default_factory=list)
    skip_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "trigger": round(self.trigger, 2),
            "trigger_label": self.trigger_label,
            "entry_zone": [round(self.entry_lo, 2), round(self.entry_hi, 2)],
            "stop": round(self.stop, 2),
            "target1": round(self.target1, 2),
            "target1_label": self.target1_label,
            "rr1": round(self.rr1, 1),
            "target2": round(self.target2, 2) if self.target2 is not None else None,
            "target2_label": self.target2_label,
            "rr2": round(self.rr2, 1) if self.rr2 is not None else None,
            "with_trend": self.with_trend,
            "notes": list(self.notes),
            "skip_reason": self.skip_reason,
        }


@dataclass
class ScalpLevels:
    """Floor/ceiling snapshot for one 30-minute check-in."""
    ticker: str
    asof: datetime               # tz-aware ET
    spot: float
    floor: Optional[float]
    floor_sources: list[str] = field(default_factory=list)
    ceiling: Optional[float] = None
    ceiling_sources: list[str] = field(default_factory=list)
    vwap: Optional[float] = None
    magnet: Optional[float] = None       # max gamma-weighted OI strike near spot
    magnet_detail: str = ""
    sigma_remaining: Optional[float] = None  # 1-sigma move left in the session ($)
    atm_iv: Optional[float] = None
    chain_expiry: Optional[str] = None
    minutes_left: int = 0
    warnings: list[str] = field(default_factory=list)
    setups: list[ScalpSetup] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "asof": self.asof.isoformat(),
            "spot": round(self.spot, 2),
            "floor": round(self.floor, 2) if self.floor is not None else None,
            "floor_sources": list(self.floor_sources),
            "ceiling": round(self.ceiling, 2) if self.ceiling is not None else None,
            "ceiling_sources": list(self.ceiling_sources),
            "vwap": round(self.vwap, 2) if self.vwap is not None else None,
            "magnet": self.magnet,
            "magnet_detail": self.magnet_detail,
            "sigma_remaining": round(self.sigma_remaining, 2) if self.sigma_remaining is not None else None,
            "atm_iv": round(self.atm_iv, 4) if self.atm_iv is not None else None,
            "chain_expiry": self.chain_expiry,
            "minutes_left": self.minutes_left,
            "warnings": list(self.warnings),
            "setups": [s.to_dict() for s in self.setups],
            "disclaimer": "Analysis only — not financial advice. OI as of last close.",
        }


# ── Session clock ─────────────────────────────────────────────────────────────

def is_market_open(now_et: datetime) -> bool:
    """Regular-session gate: Mon-Fri, 09:30-16:00 ET. Holidays are caught
    downstream (no intraday bars for today -> cron mode exits silently)."""
    if now_et.weekday() >= 5:
        return False
    return SESSION_OPEN <= now_et.time() < SESSION_CLOSE


def session_minutes_remaining(now_et: datetime) -> int:
    """Minutes of regular session left; a pre-open call gets the full 390
    (the whole session is still ahead), post-close gets 0."""
    close = now_et.replace(hour=SESSION_CLOSE.hour, minute=SESSION_CLOSE.minute,
                           second=0, microsecond=0)
    mins = int((close - now_et).total_seconds() // 60)
    return max(0, min(SESSION_MINUTES, mins))


def remaining_sigma(spot: float, iv: float, minutes_left: int) -> float:
    """1-sigma move (in $) over the remaining session minutes."""
    if spot <= 0 or iv <= 0 or minutes_left <= 0:
        return 0.0
    return spot * iv * math.sqrt(minutes_left / TRADING_MINUTES_PER_YEAR)


# ── Candidate builders ───────────────────────────────────────────────────────

def intraday_candidates(
    today: pd.DataFrame, prior: Optional[pd.DataFrame],
) -> tuple[list[LevelCandidate], Optional[float]]:
    """Price-structure candidates from 5-minute bars. Returns (candidates, vwap).

    ``today``/``prior`` are session bar frames with High/Low/Close/Volume.
    """
    cands: list[LevelCandidate] = []
    vwap = None
    if today is not None and not today.empty:
        typical = (today["High"] + today["Low"] + today["Close"]) / 3
        vol = today["Volume"].clip(lower=0)
        if float(vol.sum()) > 0:
            vwap = float((typical * vol).sum() / vol.sum())
            cands.append(LevelCandidate(vwap, "VWAP"))
        cands.append(LevelCandidate(float(today["Low"].min()), "session low"))
        cands.append(LevelCandidate(float(today["High"].max()), "session high"))
        opening = today.iloc[:6]  # first 30 minutes of 5m bars
        if len(today) > len(opening):  # only once the opening range is closed
            cands.append(LevelCandidate(float(opening["Low"].min()), "opening range low"))
            cands.append(LevelCandidate(float(opening["High"].max()), "opening range high"))
    if prior is not None and not prior.empty:
        cands.append(LevelCandidate(float(prior["Low"].min()), "prior low"))
        cands.append(LevelCandidate(float(prior["High"].max()), "prior high"))
        cands.append(LevelCandidate(float(prior["Close"].iloc[-1]), "prior close"))
    return cands, vwap


def oi_wall_candidates(
    chain: pd.DataFrame, spot: float, band_pct: float = WALL_BAND_PCT,
) -> tuple[list[LevelCandidate], Optional[float], str]:
    """Put/call OI walls inside the band, plus the gamma-OI magnet strike.

    Returns (candidates, magnet_strike, magnet_detail).
    """
    if chain is None or chain.empty:
        return [], None, ""
    lo, hi = spot * (1 - band_pct / 100), spot * (1 + band_pct / 100)
    band = chain[(chain["strike"] >= lo) & (chain["strike"] <= hi)].copy()
    if band.empty:
        return [], None, ""

    cands: list[LevelCandidate] = []
    puts = band[band["strike"] <= spot]
    if not puts.empty and puts["put_oi"].max() > 0:
        row = puts.loc[puts["put_oi"].idxmax()]
        cands.append(LevelCandidate(
            float(row["strike"]), "put wall",
            f"{int(row['put_oi']):,} put OI @ {row['strike']:g}"))
    calls = band[band["strike"] >= spot]
    if not calls.empty and calls["call_oi"].max() > 0:
        row = calls.loc[calls["call_oi"].idxmax()]
        cands.append(LevelCandidate(
            float(row["strike"]), "call wall",
            f"{int(row['call_oi']):,} call OI @ {row['strike']:g}"))

    # Magnet: strike with the most gamma-weighted total OI — where hedging
    # flow pins hardest. Falls back to raw OI when the chain has no gamma.
    total_oi = band["call_oi"].fillna(0) + band["put_oi"].fillna(0)
    if "gamma" in band.columns and band["gamma"].notna().any():
        score = total_oi * band["gamma"].fillna(0).abs()
        if float(score.max()) <= 0:
            score = total_oi
    else:
        score = total_oi
    magnet, magnet_detail = None, ""
    if float(score.max()) > 0:
        row = band.loc[score.idxmax()]
        magnet = float(row["strike"])
        magnet_detail = (f"{int(row['call_oi'] + row['put_oi']):,} total OI "
                         f"@ {row['strike']:g}")
    return cands, magnet, magnet_detail


# ── Clustering / selection ───────────────────────────────────────────────────

def _pick_level(
    cands: list[LevelCandidate], spot: float,
) -> tuple[Optional[float], list[str]]:
    """Cluster same-side candidates and return (level, contributing sources).

    Greedy clustering on price within CLUSTER_PCT of the cluster anchor;
    the cluster with the highest total weight wins (tie -> nearest spot).
    The level snaps to the cluster's OI wall strike when one is present,
    else the weight-weighted mean.
    """
    if not cands:
        return None, []
    ordered = sorted(cands, key=lambda c: c.price)
    clusters: list[list[LevelCandidate]] = []
    for c in ordered:
        if clusters and abs(c.price - clusters[-1][0].price) <= spot * CLUSTER_PCT / 100:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    def cluster_key(cl: list[LevelCandidate]) -> tuple:
        total = sum(c.weight for c in cl)
        center = sum(c.price * c.weight for c in cl) / total
        return (total, -abs(center - spot))

    best = max(clusters, key=cluster_key)
    total = sum(c.weight for c in best)
    walls = [c for c in best if c.source.endswith("wall")]
    level = walls[0].price if walls else sum(c.price * c.weight for c in best) / total
    sources = [c.detail or c.source for c in
               sorted(best, key=lambda c: c.weight, reverse=True)]
    return level, sources


# ── Entry/exit setups ────────────────────────────────────────────────────────

def _first_target(
    entry: float, stop: float, final: float, side: str,
    vwap: Optional[float], magnet: Optional[float],
) -> tuple[float, str]:
    """Nearest meaningful partial-profit spot between entry and the far
    target: VWAP or the OI magnet when one sits at least 1R inside the
    range, else the range midpoint."""
    risk = abs(entry - stop)
    lo, hi = (entry, final) if side == "long" else (final, entry)
    named = [(vwap, "VWAP"), (magnet, "magnet")]
    viable = [(p, lbl) for p, lbl in named
              if p is not None and lo < p < hi and abs(p - entry) >= risk]
    if viable:
        return min(viable, key=lambda t: abs(t[0] - entry))
    return (entry + final) / 2, "mid-range"


def _build_setup(
    side: str, trigger: float, far: float, spot: float,
    vwap: Optional[float], magnet: Optional[float],
    sigma: Optional[float],
) -> ScalpSetup:
    zone = spot * ENTRY_ZONE_PCT / 100
    buffer = spot * STOP_BUFFER_PCT / 100
    sign = 1.0 if side == "long" else -1.0
    stop = trigger - sign * buffer
    # Exit just inside the far level — the crowd's orders sit ON it.
    target2 = far - sign * zone
    risk = abs(trigger - stop)
    rr2 = abs(target2 - trigger) / risk
    target1, t1_label = _first_target(trigger, stop, target2, side, vwap, magnet)
    rr1 = abs(target1 - trigger) / risk

    label = "floor bounce" if side == "long" else "ceiling fade"
    with_trend = vwap is None or (spot >= vwap) == (side == "long")
    notes: list[str] = []
    if not with_trend:
        notes.append("counter-trend — wait for rejection (wick/stall) before entry")
    if sigma and abs(target2 - trigger) > sigma:
        notes.append("T2 beyond remaining 1σ — T1 is the realistic exit")

    skip = ""
    if rr2 < MIN_RR:
        skip = f"reward:risk {rr2:.1f} < {MIN_RR:g} — range too tight, stand aside"

    return ScalpSetup(
        side=side, trigger=trigger, trigger_label=label,
        entry_lo=trigger - zone, entry_hi=trigger + zone, stop=stop,
        target1=target1, target1_label=t1_label, rr1=rr1,
        target2=target2, target2_label="floor" if side == "short" else "ceiling",
        rr2=rr2, with_trend=with_trend, notes=notes, skip_reason=skip,
    )


def scalp_setups(lv: ScalpLevels) -> list[ScalpSetup]:
    """Derive mechanical long/short plans from a levels snapshot.

    Long buys the floor, short fades the ceiling — mean-reversion between
    the range extremes, which is the only trade the levels themselves
    support. Both sides need both levels (the opposite one is the target).
    Ordered nearest-trigger first, since that's the actionable one.
    """
    if lv.floor is None or lv.ceiling is None:
        return []
    setups = [
        _build_setup("long", lv.floor, lv.ceiling, lv.spot,
                     lv.vwap, lv.magnet, lv.sigma_remaining),
        _build_setup("short", lv.ceiling, lv.floor, lv.spot,
                      lv.vwap, lv.magnet, lv.sigma_remaining),
    ]
    setups.sort(key=lambda s: abs(s.trigger - lv.spot))
    return setups


def compute_scalp_levels(
    ticker: str,
    spot: float,
    now_et: datetime,
    today: pd.DataFrame,
    prior: Optional[pd.DataFrame],
    chain: Optional[pd.DataFrame],
    iv: Optional[float],
    chain_expiry: Optional[str] = None,
    warnings: Optional[list[str]] = None,
) -> ScalpLevels:
    """Assemble candidates from all families and pick floor & ceiling."""
    warns = list(warnings or [])
    cands, vwap = intraday_candidates(today, prior)
    wall_cands, magnet, magnet_detail = oi_wall_candidates(chain, spot) \
        if chain is not None else ([], None, "")
    cands += wall_cands

    minutes_left = session_minutes_remaining(now_et)
    sigma = remaining_sigma(spot, iv or 0.0, minutes_left)
    if sigma > 0:
        cands.append(LevelCandidate(spot - sigma, "expected move",
                                    f"-1σ ({minutes_left}m left)"))
        cands.append(LevelCandidate(spot + sigma, "expected move",
                                    f"+1σ ({minutes_left}m left)"))

    floor, floor_srcs = _pick_level([c for c in cands if c.price < spot], spot)
    ceiling, ceil_srcs = _pick_level([c for c in cands if c.price > spot], spot)
    if floor is None:
        warns.append("no candidates below spot — floor unavailable")
    if ceiling is None:
        warns.append("no candidates above spot — ceiling unavailable")

    levels = ScalpLevels(
        ticker=ticker, asof=now_et, spot=spot,
        floor=floor, floor_sources=floor_srcs,
        ceiling=ceiling, ceiling_sources=ceil_srcs,
        vwap=vwap, magnet=magnet, magnet_detail=magnet_detail,
        sigma_remaining=sigma if sigma > 0 else None,
        atm_iv=iv, chain_expiry=chain_expiry,
        minutes_left=minutes_left, warnings=warns,
    )
    levels.setups = scalp_setups(levels)
    return levels


def format_message(lv: ScalpLevels) -> str:
    """Plain-text summary for Telegram delivery."""
    lines = [f"⚡ {lv.ticker} scalp — {lv.asof.strftime('%H:%M')} ET",
             f"Spot {lv.spot:.2f}" + (f" | VWAP {lv.vwap:.2f}" if lv.vwap else "")]
    if lv.floor is not None:
        lines.append(f"Floor {lv.floor:.2f}  ({', '.join(lv.floor_sources[:3])})")
    if lv.ceiling is not None:
        lines.append(f"Ceiling {lv.ceiling:.2f}  ({', '.join(lv.ceiling_sources[:3])})")
    if lv.magnet is not None:
        lines.append(f"Magnet {lv.magnet:g}  ({lv.magnet_detail})")
    if lv.sigma_remaining:
        lines.append(f"1σ left: ±{lv.sigma_remaining:.2f} ({lv.minutes_left}m to close)")
    for s in lv.setups:
        lines.append("")
        lines.append(format_setup(s))
    for w in lv.warnings:
        lines.append(f"⚠ {w}")
    lines.append("Analysis only — not financial advice.")
    return "\n".join(lines)


def format_setup(s: ScalpSetup) -> str:
    """One setup as compact Telegram-friendly lines."""
    arrow = "🟢 LONG" if s.side == "long" else "🔴 SHORT"
    trend = "with trend" if s.with_trend else "counter-trend"
    head = (f"{arrow} {s.trigger_label} {s.entry_lo:.2f}–{s.entry_hi:.2f} "
            f"({trend})")
    if s.skip_reason:
        return f"{head}\n   ✋ {s.skip_reason}"
    body = (f"   stop {s.stop:.2f} | T1 {s.target1:.2f} "
            f"{s.target1_label} ({s.rr1:.1f}R) | T2 {s.target2:.2f} "
            f"{s.target2_label} ({s.rr2:.1f}R)")
    lines = [head, body]
    for n in s.notes:
        lines.append(f"   ⚠ {n}")
    return "\n".join(lines)
