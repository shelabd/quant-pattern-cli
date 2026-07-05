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

    return ScalpLevels(
        ticker=ticker, asof=now_et, spot=spot,
        floor=floor, floor_sources=floor_srcs,
        ceiling=ceiling, ceiling_sources=ceil_srcs,
        vwap=vwap, magnet=magnet, magnet_detail=magnet_detail,
        sigma_remaining=sigma if sigma > 0 else None,
        atm_iv=iv, chain_expiry=chain_expiry,
        minutes_left=minutes_left, warnings=warns,
    )


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
    for w in lv.warnings:
        lines.append(f"⚠ {w}")
    lines.append("Analysis only — not financial advice.")
    return "\n".join(lines)
