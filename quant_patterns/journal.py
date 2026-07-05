"""
Forward-test journal for pin-fly recommendations.

Historical chains with OI on 2-5 DTE expiries don't exist for free, so the
fly engine is validated forward instead: `qpat fly --log` appends each
recommendation (pin, OI, legs, debit — the full to_dict snapshot) to
~/.qpat/fly_journal.jsonl, and `qpat journal` scores expired entries
against the realized close on expiry day. Over time this accumulates the
two numbers no vendor sells: how often price actually settles at the
recommended pin, and the P&L of the recommended structure.

Pure logic except the jsonl read/append helpers; settle prices are
injected as a callable so scoring is offline-testable. Settlement uses the
expiry-day close — a proxy that is exact for PM-settled contracts
(SPY/QQQ/equities) but not for AM-settled index options (SPX/RUT).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from statistics import mean, median
from typing import Callable, Optional

from .macro_calendar import QPAT_DIR

logger = logging.getLogger(__name__)

JOURNAL_PATH = QPAT_DIR / "fly_journal.jsonl"


# ── IO ───────────────────────────────────────────────────────────────────────

def load_journal(path=None) -> list[dict]:
    """Read all journal entries, skipping corrupt lines."""
    path = path or JOURNAL_PATH
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
            logger.warning(f"Skipping corrupt journal line: {line[:80]}")
    return entries


def log_recommendation(rec, as_of: Optional[date] = None, path=None) -> tuple[Optional[dict], bool]:
    """Append a FlyRecommendation to the journal.

    Returns (entry, appended). entry is None when the rec has no pin to
    score (no expiry/body — nothing to forward-test). appended is False
    when an identical pin (same ticker/as-of/expiry/body) is already
    logged, so a second run on the same day doesn't double-count.
    """
    if rec.expiry is None or rec.body_strike is None:
        return None, False

    as_of = as_of or date.today()
    path = path or JOURNAL_PATH
    entry = {
        "as_of": as_of.isoformat(),
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        **rec.to_dict(),
    }

    for existing in load_journal(path):
        if (existing.get("ticker") == entry["ticker"]
                and existing.get("as_of") == entry["as_of"]
                and existing.get("expiry") == entry["expiry"]
                and existing.get("body_strike") == entry["body_strike"]):
            return existing, False

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry, True


# ── Scoring (pure) ───────────────────────────────────────────────────────────

def score_entry(entry: dict, settle: float) -> dict:
    """Score one journal entry against the realized settle price.

    Always scores pin accuracy (settle distance from the body). When the
    entry was a priced trade (width + debit present), also scores the fly:
    payoff = max(0, width - |settle - body|), pnl = payoff - debit.
    """
    body = entry["body_strike"]
    spot = entry.get("spot")
    out = dict(entry)
    out["settle"] = settle
    out["pin_dist"] = round(settle - body, 4)
    out["pin_dist_pct"] = round((settle - body) / spot * 100, 4) if spot else None

    width = entry.get("selected_width")
    debit = entry.get("debit_per_share")
    if width and debit:
        payoff = max(0.0, width - abs(settle - body))
        out["payoff_per_share"] = round(payoff, 4)
        out["pnl_per_share"] = round(payoff - debit, 4)
        out["pnl_per_fly"] = round((payoff - debit) * 100, 2)
        out["r_multiple"] = round((payoff - debit) / debit, 2)
        out["in_tent"] = abs(settle - body) < width
        out["win"] = payoff - debit > 0
    return out


def score_journal(
    entries: list[dict],
    get_close: Callable[[str, date], Optional[float]],
    today: Optional[date] = None,
) -> tuple[list[dict], list[dict]]:
    """Split entries into (scored, pending).

    Entries whose expiry is before today are scored against
    get_close(ticker, expiry); a missing close (data gap) leaves the entry
    pending rather than guessing.
    """
    today = today or date.today()
    scored, pending = [], []
    for entry in entries:
        expiry = date.fromisoformat(entry["expiry"])
        if expiry >= today:
            pending.append(entry)
            continue
        settle = get_close(entry["ticker"], expiry)
        if settle is None:
            logger.warning(f"No close for {entry['ticker']} on {expiry}; leaving pending")
            pending.append(entry)
            continue
        scored.append(score_entry(entry, settle))
    return scored, pending


def _pop_buckets(forecast: list[dict], edges=(0.2, 0.4, 0.6)) -> list[dict]:
    """Bucket forecasted trades by predicted POP and report realized win rate.

    Reliability table: a well-calibrated model's predicted POP should track
    the actual win rate inside each bucket. Empty buckets are dropped.
    """
    bounds = [0.0, *edges, 1.0001]
    buckets = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        grp = [e for e in forecast if lo <= e["prob_profit"] < hi]
        if grp:
            buckets.append({
                "range": f"{lo:.0%}–{min(hi, 1.0):.0%}",
                "n": len(grp),
                "mean_pred_pop": round(mean(e["prob_profit"] for e in grp), 3),
                "actual_win_rate": round(mean(1.0 if e["win"] else 0.0 for e in grp), 3),
            })
    return buckets


def summarize(scored: list[dict]) -> dict:
    """Aggregate forward-test stats across scored entries.

    Pin accuracy uses every scored entry (NO TRADE pins still test the
    pin hypothesis); trade stats use only priced PASS entries. When entries
    carry the expected-move forecast (prob_profit / expected_value_per_fly),
    a calibration block compares those ex-ante predictions to realized
    outcomes — the only honest test of whether the move model adds edge.
    """
    pin_pcts = [abs(e["pin_dist_pct"]) for e in scored if e.get("pin_dist_pct") is not None]
    trades = [e for e in scored if e.get("win") is not None]

    out = {
        "n_scored": len(scored),
        "n_trades": len(trades),
        "median_abs_pin_dist_pct": round(median(pin_pcts), 3) if pin_pcts else None,
        "pin_within_half_pct": round(mean(abs(p) <= 0.5 for p in pin_pcts), 3) if pin_pcts else None,
    }
    if trades:
        out.update({
            "hit_rate": round(mean(e["in_tent"] for e in trades), 3),
            "win_rate": round(mean(e["win"] for e in trades), 3),
            "total_pnl_per_fly": round(sum(e["pnl_per_fly"] for e in trades), 2),
            "avg_r_multiple": round(mean(e["r_multiple"] for e in trades), 2),
            "best_r": max(e["r_multiple"] for e in trades),
            "worst_r": min(e["r_multiple"] for e in trades),
        })

    # ── Realized centering coefficients (pin pull & drift) ───────────────
    # The POP/EV settle distribution centers at spot with POP_PIN_PULL and
    # POP_DRIFT_SHIFT both 0 (butterfly.py). These are the realized values
    # of the two coefficients — the evidence that would justify raising
    # them: pin_pull is the median of (settle−spot)/(pin−spot) (1 = settles
    # land on the pin, 0 = no pull, negative = repelled); drift_move is the
    # mean settle−spot move signed by the drift call (positive = the drift
    # direction was right on average).
    with_settle = [e for e in scored
                   if e.get("settle") is not None and e.get("spot")
                   and e.get("body_strike") is not None]
    pulls = [(e["settle"] - e["spot"]) / (e["body_strike"] - e["spot"])
             for e in with_settle if abs(e["body_strike"] - e["spot"]) > 1e-9]
    drift_moves = [(e["settle"] - e["spot"]) * (1 if e["drift"] == "bullish" else -1)
                   for e in with_settle if e.get("drift") in ("bullish", "bearish")]
    if pulls or drift_moves:
        centering: dict = {}
        if pulls:
            centering["n_pin"] = len(pulls)
            centering["median_pin_pull"] = round(median(pulls), 2)
        if drift_moves:
            centering["n_drift"] = len(drift_moves)
            centering["mean_drift_signed_move"] = round(mean(drift_moves), 2)
            centering["drift_hit_rate"] = round(mean(m > 0 for m in drift_moves), 3)
        out["centering"] = centering

    # ── Expected-move forecast calibration (POP & EV vs realized) ────────
    forecast = [e for e in trades if e.get("prob_profit") is not None]
    if forecast:
        wins = [1.0 if e["win"] else 0.0 for e in forecast]
        pops = [e["prob_profit"] for e in forecast]
        cal = {
            "n_forecast": len(forecast),
            "mean_pred_pop": round(mean(pops), 3),
            "actual_win_rate": round(mean(wins), 3),
            # Brier score: mean squared error of the probability forecast
            # (0 = perfect, 0.25 = a coin flip, lower is better).
            "pop_brier": round(mean((p - w) ** 2 for p, w in zip(pops, wins)), 4),
            "buckets": _pop_buckets(forecast),
        }
        ev_pairs = [(e["expected_value_per_fly"], e["pnl_per_fly"]) for e in forecast
                    if e.get("expected_value_per_fly") is not None]
        if ev_pairs:
            pred_ev = [p for p, _ in ev_pairs]
            actual = [a for _, a in ev_pairs]
            cal.update({
                "mean_pred_ev_per_fly": round(mean(pred_ev), 2),
                "mean_actual_pnl_per_fly": round(mean(actual), 2),
                # realized minus predicted: >0 means the model under-promised.
                "ev_bias_per_fly": round(mean(actual) - mean(pred_ev), 2),
            })
        out["calibration"] = cal
    return out
