"""Rally-potential screener: cross-sectional factor scoring over a universe.

Pure logic, no Rich/Click/network. The screener ranks every ticker in the
universe on six factor families and blends them into a 0-100 composite per
profile, so the output always says *why* a name surfaced:

* **momentum** — 12-1 / 6m / 3m returns (classic cross-sectional momentum).
* **hi52** — proximity to and freshness of the 52-week high ("new high with
  room" beats "beaten down").
* **trend** — 20/50/200 EMA stack plus persistence above the 200-SMA
  (Stage-2 uptrend template).
* **squeeze** — Bollinger-width percentile of its own trailing history, ATR
  contraction, NR7: volatility compression, the coiled spring.
* **trigger** — distance to the 20d high, a fresh 50d-high breakout, and the
  close's position in the bar's range: is there an entry *now*.
* **volume** — RVOL, OBV slope, 20d up/down volume ratio: accumulation
  evidence.

Two profiles weight the same factor panel differently: ``swing`` (days-weeks:
squeeze + trigger + volume heavy) and ``position`` (weeks-months: momentum +
hi52 + trend heavy). Percentiles are cross-sectional — a score of 90 means
"top decile of the scanned universe today", not an absolute claim.

Factors are computed on ADJUSTED prices (splits/dividends folded in) — the
right basis for momentum and 52wH math. The journal therefore scores relative
forward returns from the same adjusted series, unlike swing's raw-level
stop/target replay (documented difference; there are no absolute price levels
to protect here).

Analysis only — never routes orders, not financial advice.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .swing import ema, obv, signal_rvol, wilder_atr

logger = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────────────

# Bars required before a ticker enters the panel at all; individual factors
# needing deeper history (12-1 momentum wants 252) go NaN and drop out of
# their family mean rather than excluding the ticker.
MIN_BARS = 120

SQUEEZE_HIST = 126          # trailing window the BB-width percentile is taken over
UPDOWN_VOL_WINDOW = 20
OBV_SLOPE_BARS = 21
REASON_FAMILIES = 3         # how many top families feed the reasons line

# Forward-return horizons (trading sessions) the journal is scored over.
FWD_HORIZONS = (5, 10, 21, 63)
SCORE_BANDS = ((80, 101), (60, 80), (0, 60))   # composite-score buckets

# Options-flow note thresholds (snapshot heuristic — the free CBOE feed has
# no volume history, so this is a same-day read, not a baseline comparison).
FLOW_NEAR_DTE = 45
FLOW_MIN_CONTRACTS = 2_000
FLOW_CP_RATIO = 2.0

# +1: higher raw value ranks higher. -1: lower ranks higher.
FACTOR_DIRECTIONS: dict[str, int] = {
    "ret_12_1": +1, "ret_6m": +1, "ret_3m": +1,
    "pct_off_52w_high": +1,        # -3% beats -40%: closer to the high
    "days_since_52w_high": -1,     # fresher high ranks higher
    "ema_stack": +1, "pct_above_200sma_3m": +1,
    "bb_width_pctile": -1, "atr_contraction": -1, "nr7": +1,
    "dist_to_20d_high": +1, "breakout_50d": +1, "close_range_pos": +1,
    "rvol": +1, "obv_slope": +1, "updown_vol": +1,
}

FAMILIES: dict[str, list[str]] = {
    "momentum": ["ret_12_1", "ret_6m", "ret_3m"],
    "hi52": ["pct_off_52w_high", "days_since_52w_high"],
    "trend": ["ema_stack", "pct_above_200sma_3m"],
    "squeeze": ["bb_width_pctile", "atr_contraction", "nr7"],
    "trigger": ["dist_to_20d_high", "breakout_50d", "close_range_pos"],
    "volume": ["rvol", "obv_slope", "updown_vol"],
}

PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    # days-weeks: the coiled spring with an entry, confirmed by volume
    "swing": {"squeeze": 0.25, "trigger": 0.25, "volume": 0.20,
              "trend": 0.15, "hi52": 0.10, "momentum": 0.05},
    # weeks-months: ride established momentum near highs
    "position": {"momentum": 0.30, "hi52": 0.25, "trend": 0.20,
                 "volume": 0.15, "squeeze": 0.05, "trigger": 0.05},
}


# ── Per-ticker factor computation ────────────────────────────────────────────

def _trailing_return(close: pd.Series, back: int, skip: int = 0) -> float:
    """% return from `back` bars ago to `skip` bars ago (0 = last bar)."""
    if len(close) <= back:
        return float("nan")
    ref = float(close.iloc[-1 - back])
    cur = float(close.iloc[-1 - skip])
    if ref <= 0:
        return float("nan")
    return (cur / ref - 1) * 100


def compute_factors(df: pd.DataFrame) -> Optional[dict[str, float]]:
    """Raw factor values for the last completed bar of one ticker's OHLCV.

    Returns None when the frame is unusable (too short, dead prices);
    individual factors may be NaN when their lookback exceeds the history.
    """
    if df is None or len(df) < MIN_BARS:
        return None
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    c = float(close.iloc[-1])
    if not math.isfinite(c) or c <= 0:
        return None

    f: dict[str, float] = {}

    # momentum (percent returns)
    f["ret_12_1"] = _trailing_return(close, 252, skip=21)
    f["ret_6m"] = _trailing_return(close, 126)
    f["ret_3m"] = _trailing_return(close, 63)

    # 52-week high (whatever history exists, up to 252 bars)
    hi_win = high.iloc[-252:]
    hi52 = float(hi_win.max())
    f["pct_off_52w_high"] = (c / hi52 - 1) * 100 if hi52 > 0 else float("nan")
    f["days_since_52w_high"] = float(len(hi_win) - 1 - int(np.argmax(hi_win.to_numpy())))

    # trend: graded EMA stack (0-4) + persistence above the 200-SMA
    e20 = float(ema(close, 20).iloc[-1])
    e50 = float(ema(close, 50).iloc[-1])
    e200_series = ema(close, 200)
    e200 = float(e200_series.iloc[-1])
    stack = float((c > e20) + (e20 > e50) + (e50 > e200))
    if len(e200_series) > 21:
        stack += float(e200 > float(e200_series.iloc[-22]))
    f["ema_stack"] = stack
    sma200 = close.rolling(200).mean()
    recent = (close.iloc[-63:] > sma200.iloc[-63:])
    f["pct_above_200sma_3m"] = float(recent.mean() * 100) \
        if not sma200.iloc[-63:].isna().all() else float("nan")

    # squeeze: BB width vs its own trailing distribution, ATR contraction, NR7
    m20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    bbw = (4 * sd20 / m20).replace([np.inf, -np.inf], np.nan)
    bbw_hist = bbw.iloc[-SQUEEZE_HIST:].dropna()
    f["bb_width_pctile"] = float((bbw_hist <= bbw_hist.iloc[-1]).mean() * 100) \
        if len(bbw_hist) >= 40 else float("nan")
    atr_pct = (wilder_atr(df) / close).replace([np.inf, -np.inf], np.nan)
    base = float(atr_pct.iloc[-SQUEEZE_HIST:].mean())
    f["atr_contraction"] = float(atr_pct.iloc[-1]) / base if base > 0 else float("nan")
    rng = (high - low).iloc[-7:]
    f["nr7"] = 1.0 if float(rng.iloc[-1]) <= float(rng.min()) else 0.0

    # trigger: distance to the 20d high, fresh 50d breakout, close-in-range
    prior_20d_high = float(high.iloc[-21:-1].max())
    f["dist_to_20d_high"] = (c / prior_20d_high - 1) * 100 if prior_20d_high > 0 else float("nan")
    prior_50d_high = float(high.iloc[-51:-1].max())
    f["breakout_50d"] = 1.0 if prior_50d_high > 0 and c > prior_50d_high else 0.0
    bar_rng = float(high.iloc[-1]) - float(low.iloc[-1])
    f["close_range_pos"] = (c - float(low.iloc[-1])) / bar_rng if bar_rng > 0 else 0.5

    # volume: RVOL, OBV slope in days-of-volume units, up/down volume ratio
    rvol = signal_rvol(vol)
    f["rvol"] = rvol if rvol is not None else float("nan")
    obv_series = obv(close, vol)
    vol_base = float(vol.iloc[-OBV_SLOPE_BARS:].mean())
    f["obv_slope"] = (float(obv_series.iloc[-1]) - float(obv_series.iloc[-1 - OBV_SLOPE_BARS])) \
        / vol_base if len(obv_series) > OBV_SLOPE_BARS and vol_base > 0 else float("nan")
    delta = close.diff().iloc[-UPDOWN_VOL_WINDOW:]
    v = vol.iloc[-UPDOWN_VOL_WINDOW:]
    up_v, down_v = float(v[delta > 0].sum()), float(v[delta < 0].sum())
    f["updown_vol"] = up_v / down_v if down_v > 0 else (2.0 if up_v > 0 else 1.0)

    return f


# ── Cross-sectional ranking & composites ─────────────────────────────────────

def build_factor_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """tickers x factors raw panel. Tickers with unusable frames drop out."""
    rows = {}
    for ticker, df in frames.items():
        factors = compute_factors(df)
        if factors is not None:
            rows[ticker] = factors
    return pd.DataFrame.from_dict(rows, orient="index")


def cross_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """Direction-aware 0-100 cross-sectional percentile of every factor.

    NaNs stay NaN (a missing factor never counts for or against a name).
    """
    ranked = {}
    for col in panel.columns:
        series = panel[col] * FACTOR_DIRECTIONS.get(col, +1)
        ranked[col] = series.rank(pct=True, na_option="keep") * 100
    return pd.DataFrame(ranked, index=panel.index)


def family_scores(pct_panel: pd.DataFrame) -> pd.DataFrame:
    """tickers x families: mean of each family's available factor percentiles."""
    out = {}
    for fam, cols in FAMILIES.items():
        present = [c for c in cols if c in pct_panel.columns]
        out[fam] = pct_panel[present].mean(axis=1, skipna=True) if present else np.nan
    return pd.DataFrame(out, index=pct_panel.index)


def composite_scores(fam_panel: pd.DataFrame, profile: str) -> pd.Series:
    """Weighted composite 0-100; weights renormalize over non-NaN families
    so a thin-history name isn't penalized for factors it can't have."""
    weights = PROFILE_WEIGHTS[profile]
    w = pd.Series(weights, dtype=float)
    aligned = fam_panel[list(weights)]
    mask = aligned.notna()
    denom = mask.mul(w, axis=1).sum(axis=1)
    num = aligned.fillna(0).mul(w, axis=1).sum(axis=1)
    return (num / denom.replace(0, np.nan)).round(1)


# ── Results & reasons ────────────────────────────────────────────────────────

@dataclass
class ScreenResult:
    """One ranked candidate for one profile on one as-of date."""
    ticker: str
    profile: str
    as_of: date
    score: float
    families: dict[str, float]
    reasons: list[str] = field(default_factory=list)
    close: Optional[float] = None
    pct_off_52w_high: Optional[float] = None
    rvol: Optional[float] = None
    sector: Optional[str] = None
    earnings_date: Optional[str] = None
    short_pct_float: Optional[float] = None
    options_note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "profile": self.profile,
            "as_of": self.as_of.isoformat(),
            "score": self.score,
            "families": {k: round(v, 1) for k, v in self.families.items()
                         if v == v},  # drop NaN
            "reasons": list(self.reasons),
            "close": round(self.close, 2) if self.close is not None else None,
            "pct_off_52w_high": round(self.pct_off_52w_high, 1)
                if self.pct_off_52w_high is not None else None,
            "rvol": round(self.rvol, 2) if self.rvol is not None else None,
            "sector": self.sector,
            "earnings_date": self.earnings_date,
            "short_pct_float": self.short_pct_float,
            "options_note": self.options_note,
            "disclaimer": "Analysis only — not financial advice.",
        }


def _fmt_reason(fam: str, raw: pd.Series, pct: pd.Series) -> Optional[str]:
    """One human string for a family, from its best-ranked raw factor."""
    def top(p):  # "top 4%" from a percentile
        return f"top {max(1, round(100 - p)):d}%"

    if fam == "momentum" and raw.get("ret_12_1") == raw.get("ret_12_1"):
        return f"12-1 momentum {raw['ret_12_1']:+.0f}% ({top(pct['ret_12_1'])})"
    if fam == "hi52" and raw.get("pct_off_52w_high") == raw.get("pct_off_52w_high"):
        return f"{abs(raw['pct_off_52w_high']):.1f}% off 52w high"
    if fam == "trend":
        return f"EMA stack {raw.get('ema_stack', 0):.0f}/4 aligned"
    if fam == "squeeze" and raw.get("bb_width_pctile") == raw.get("bb_width_pctile"):
        note = f"BB width in bottom {max(1, round(raw['bb_width_pctile'])):d}% of 6mo"
        if raw.get("nr7"):
            note += ", NR7"
        return note
    if fam == "trigger":
        if raw.get("breakout_50d"):
            return "fresh 50d-high breakout"
        if raw.get("dist_to_20d_high") == raw.get("dist_to_20d_high"):
            return f"{abs(raw['dist_to_20d_high']):.1f}% from 20d high"
    if fam == "volume" and raw.get("rvol") == raw.get("rvol"):
        return f"{raw['rvol']:.1f}x RVOL, up/down vol {raw.get('updown_vol', 1):.1f}"
    return None


def reasons_for(profile: str, fam_row: pd.Series, raw_row: pd.Series,
                pct_row: pd.Series) -> list[str]:
    """The REASON_FAMILIES families contributing most to this composite."""
    weights = PROFILE_WEIGHTS[profile]
    contrib = {fam: weights[fam] * fam_row[fam]
               for fam in weights if fam_row.get(fam) == fam_row.get(fam)}
    top_fams = sorted(contrib, key=contrib.get, reverse=True)[:REASON_FAMILIES]
    reasons = []
    for fam in top_fams:
        line = _fmt_reason(fam, raw_row, pct_row)
        if line:
            reasons.append(line)
    return reasons


def build_results(profile: str, as_of: date, raw_panel: pd.DataFrame,
                  pct_panel: pd.DataFrame, fam_panel: pd.DataFrame,
                  closes: dict[str, float], top: int) -> list[ScreenResult]:
    """Assemble the top `top` ScreenResults for one profile."""
    comp = composite_scores(fam_panel, profile).dropna().sort_values(ascending=False)
    results = []
    for ticker in comp.index[:top]:
        fam_row = fam_panel.loc[ticker]
        raw_row = raw_panel.loc[ticker]
        results.append(ScreenResult(
            ticker=ticker, profile=profile, as_of=as_of,
            score=float(comp[ticker]),
            families={k: float(v) for k, v in fam_row.items()},
            reasons=reasons_for(profile, fam_row, raw_row, pct_panel.loc[ticker]),
            close=closes.get(ticker),
            pct_off_52w_high=float(raw_row["pct_off_52w_high"])
                if raw_row["pct_off_52w_high"] == raw_row["pct_off_52w_high"] else None,
            rvol=float(raw_row["rvol"]) if raw_row["rvol"] == raw_row["rvol"] else None,
        ))
    return results


# ── Options-flow note (snapshot heuristic) ───────────────────────────────────

def options_flow_note(chains: list[tuple[date, pd.DataFrame]], spot: float,
                      as_of: date) -> Optional[str]:
    """Same-day options read from the free chain snapshot: total C/P volume
    ratio plus the share of call volume sitting in near-dated (<=45 DTE) OTM
    strikes. No history exists in the feed, so this is a snapshot heuristic,
    not a spike-vs-baseline claim. Returns None when volume is too thin to
    mean anything.
    """
    call_v = put_v = near_otm_call_v = 0.0
    for expiry, chain in chains or []:
        if chain is None or chain.empty or "call_vol" not in chain.columns:
            continue
        cv = chain["call_vol"].fillna(0)
        pv = chain["put_vol"].fillna(0) if "put_vol" in chain.columns else 0
        call_v += float(cv.sum())
        put_v += float(pv.sum()) if not isinstance(pv, int) else 0.0
        if 0 <= (expiry - as_of).days <= FLOW_NEAR_DTE:
            otm = chain["strike"] > spot
            near_otm_call_v += float(cv[otm].sum())
    total = call_v + put_v
    if total < FLOW_MIN_CONTRACTS:
        return None
    cp = call_v / put_v if put_v > 0 else float("inf")
    otm_share = near_otm_call_v / call_v * 100 if call_v > 0 else 0.0
    if cp < FLOW_CP_RATIO:
        return None
    cp_txt = f"{cp:.1f}" if math.isfinite(cp) else ">10"
    return (f"C/P vol {cp_txt}, {otm_share:.0f}% of call vol in near-dated OTM "
            f"({int(total):,} contracts today)")


# ── Journal IO & forward-test scoring ────────────────────────────────────────

def load_screen_journal(path) -> list[dict]:
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
            logger.warning(f"Skipping corrupt screen journal line: {line[:80]}")
    return entries


def log_screen(results: list[ScreenResult], path) -> int:
    """Append one line per (as_of, profile, ticker); duplicates from
    wake-coalesced cron re-fires are dropped. Returns lines appended."""
    existing = {(e.get("as_of"), e.get("profile"), e.get("ticker"))
                for e in load_screen_journal(path)}
    appended = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for r in results:
            key = (r.as_of.isoformat(), r.profile, r.ticker)
            if key in existing:
                continue
            entry = {"logged_at": datetime.now().isoformat(timespec="seconds"),
                     **r.to_dict()}
            fh.write(json.dumps(entry) + "\n")
            existing.add(key)
            appended += 1
    return appended


def _fwd_returns(bars: pd.DataFrame) -> dict[int, float]:
    """Forward % returns from the first bar's open (the placeable fill for an
    after-close signal) to each horizon's close, on the adjusted series."""
    out = {}
    if bars is None or bars.empty:
        return out
    fill = float(bars["Open"].iloc[0])
    if fill <= 0:
        return out
    for h in FWD_HORIZONS:
        if len(bars) >= h:
            out[h] = (float(bars["Close"].iloc[h - 1]) / fill - 1) * 100
    return out


def _return_bucket(rows: list[dict], h: int) -> dict:
    vals = [r["fwd"][h] for r in rows if h in r["fwd"]]
    rels = [r["rel"][h] for r in rows if h in r["rel"]]
    n = len(vals)
    return {
        "n": n,
        "avg_pct": round(float(np.mean(vals)), 2) if n else None,
        "median_pct": round(float(np.median(vals)), 2) if n else None,
        "win_rate": round(float(np.mean([v > 0 for v in vals])), 3) if n else None,
        "avg_vs_spy_pct": round(float(np.mean(rels)), 2) if rels else None,
    }


def score_screen_journal(entries: list[dict],
                         get_bars: Callable[[str, str], Optional[pd.DataFrame]]) -> dict:
    """Score journaled picks. `get_bars(ticker, as_of_iso)` returns completed
    ADJUSTED daily bars strictly after as_of, or None. Relative returns use
    SPY bars fetched through the same callable. Dedup (as_of, profile, ticker).
    """
    seen: set = set()
    rows: list[dict] = []
    pending = 0
    spy_cache: dict[str, dict[int, float]] = {}
    for e in entries:
        key = (e.get("as_of"), e.get("profile"), e.get("ticker"))
        if None in key or key in seen:
            continue
        seen.add(key)
        fwd = _fwd_returns(get_bars(e["ticker"], e["as_of"]))
        if not fwd:
            pending += 1
            continue
        if e["as_of"] not in spy_cache:
            spy_cache[e["as_of"]] = _fwd_returns(get_bars("SPY", e["as_of"]))
        spy_fwd = spy_cache[e["as_of"]]
        rel = {h: fwd[h] - spy_fwd[h] for h in fwd if h in spy_fwd}
        rows.append({"profile": e["profile"], "score": e.get("score") or 0.0,
                     "fwd": fwd, "rel": rel})

    def bucketize(sub: list[dict]) -> dict:
        return {f"+{h}d": _return_bucket(sub, h) for h in FWD_HORIZONS}

    by_profile = {}
    for profile in sorted({r["profile"] for r in rows}):
        sub = [r for r in rows if r["profile"] == profile]
        by_profile[profile] = {
            "overall": bucketize(sub),
            "by_score_band": {
                f"{lo}-{min(hi - 1, 100)}": bucketize(
                    [r for r in sub if lo <= r["score"] < hi])
                for lo, hi in SCORE_BANDS},
        }
    return {"n_picks": len(rows), "pending": pending, "by_profile": by_profile}


# ── Telegram formatting ──────────────────────────────────────────────────────

def format_screen_message(results_by_profile: dict[str, list[ScreenResult]],
                          as_of: date, regime: Optional[str] = None,
                          limit: int = 10) -> str:
    lines = [f"📡 qpat screen — {as_of.isoformat()} close"]
    if regime:
        lines.append(f"Regime: {regime}")
    for profile, results in results_by_profile.items():
        if not results:
            continue
        lines.append("")
        lines.append(f"— {profile.upper()} —")
        for i, r in enumerate(results[:limit], 1):
            fams = " · ".join(
                f"{k[:3]} {v:.0f}" for k, v in sorted(
                    r.families.items(),
                    key=lambda kv: PROFILE_WEIGHTS[profile].get(kv[0], 0),
                    reverse=True)[:3]
                if v == v)
            lines.append(f"{i}. {r.ticker} {r.score:.0f} — {fams}")
            if r.reasons:
                lines.append(f"   {r.reasons[0]}")
            if r.earnings_date:
                lines.append(f"   ⚠ earnings {r.earnings_date}")
    lines.append("")
    lines.append("Candidates to research, not entries. Analysis only — "
                 "not financial advice.")
    return "\n".join(lines)
