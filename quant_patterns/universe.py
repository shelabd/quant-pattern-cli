"""Screener universe: which ~1,500 liquid US stocks `qpat screen` scans.

The live universe lives at ``~/.qpat/universe.json`` and is rebuilt on demand
(`qpat screen --refresh-universe`) from the free NASDAQ Trader symbol
directories, ranked by 20-day average dollar volume. Until the first refresh,
a bundled snapshot (``quant_patterns/data/universe_default.csv``) makes the
command work out of the box.

Network code uses stdlib urllib only, matching the rest of the repo.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import urllib.request
from datetime import date, datetime
from importlib import resources
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

QPAT_DIR = Path.home() / ".qpat"
UNIVERSE_PATH = QPAT_DIR / "universe.json"

SYMBOL_DIRECTORY_URLS = [
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
]

DEFAULT_SIZE = 1500
DEFAULT_MIN_DOLLAR_VOL = 20e6      # 20d average daily dollar volume floor
DOLLAR_VOL_WINDOW = 20

# Security names that are never common operating stock we want to screen.
_NAME_EXCLUDES = re.compile(
    r"warrant|right(s)?\b|\bunit(s)?\b|preferred|depositary|%|due \d{4}|"
    r"\bETN\b|\bnote(s)?\b", re.IGNORECASE)


def _parse_directory(text: str) -> list[dict]:
    """Parse one NASDAQ Trader pipe-delimited symbol file into candidate
    common stocks: {ticker, name}. Filters test issues, ETFs, and
    warrant/right/unit/preferred lines; the dollar-volume ranking in
    `refresh_universe` prunes whatever slips through."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or "|" not in lines[0]:
        return []
    header = [h.strip() for h in lines[0].split("|")]
    idx = {name: i for i, name in enumerate(header)}
    sym_col = "Symbol" if "Symbol" in idx else "ACT Symbol"
    out = []
    for ln in lines[1:]:
        if ln.startswith("File Creation Time"):
            continue
        parts = ln.split("|")
        if len(parts) < len(header):
            continue

        def col(name: str, default: str = "") -> str:
            return parts[idx[name]].strip() if name in idx else default

        sym, name = col(sym_col), col("Security Name")
        if not sym or not name:
            continue
        if col("Test Issue") == "Y" or col("ETF") == "Y":
            continue
        if col("NextShares") == "Y":
            continue
        # $ = preferred, . = class/when-issued suffixes on CQS symbology;
        # 5-letter NASDAQ symbols ending W/R/U are warrants/rights/units.
        if any(ch in sym for ch in "$.^~"):
            continue
        if len(sym) == 5 and sym[-1] in "WRU":
            continue
        if _NAME_EXCLUDES.search(name):
            continue
        out.append({"ticker": sym.replace("/", "-"), "name": name})
    return out


def fetch_symbol_directory(timeout: int = 30) -> list[dict]:
    """Download and merge both NASDAQ Trader symbol directories."""
    merged: dict[str, dict] = {}
    for url in SYMBOL_DIRECTORY_URLS:
        req = urllib.request.Request(url, headers={"User-Agent": "qpat/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        for row in _parse_directory(text):
            merged.setdefault(row["ticker"], row)
    if not merged:
        raise ValueError("Symbol directory download returned no candidates")
    return sorted(merged.values(), key=lambda r: r["ticker"])


def refresh_universe(
    fetch_ohlcv: Callable[[list[str]], dict],
    size: int = DEFAULT_SIZE,
    min_dollar_vol: float = DEFAULT_MIN_DOLLAR_VOL,
    path: Optional[Path] = None,
) -> list[dict]:
    """Rebuild the universe: symbol directories -> bulk recent OHLCV (via the
    injected `fetch_ohlcv(tickers) -> {ticker: DataFrame}`) -> rank by 20d
    average dollar volume -> keep the top `size` above `min_dollar_vol`."""
    candidates = fetch_symbol_directory()
    frames = fetch_ohlcv([c["ticker"] for c in candidates])
    names = {c["ticker"]: c["name"] for c in candidates}

    ranked = []
    for ticker, df in frames.items():
        if df is None or len(df) < DOLLAR_VOL_WINDOW:
            continue
        tail = df.iloc[-DOLLAR_VOL_WINDOW:]
        adv = float((tail["Close"] * tail["Volume"]).mean())
        if adv >= min_dollar_vol:
            ranked.append({"ticker": ticker, "name": names.get(ticker, ticker),
                           "avg_dollar_vol": round(adv)})
    ranked.sort(key=lambda r: r["avg_dollar_vol"], reverse=True)
    universe = ranked[:size]

    target = path or UNIVERSE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(
        {"as_of": datetime.now().isoformat(timespec="seconds"),
         "min_dollar_vol": min_dollar_vol, "tickers": universe}, indent=1))
    logger.info(f"Universe refreshed: {len(universe)} tickers -> {target}")
    return universe


def _bundled_universe() -> list[str]:
    """The committed snapshot fallback (see data/universe_default.csv)."""
    ref = resources.files("quant_patterns").joinpath("data/universe_default.csv")
    with ref.open("r") as fh:
        return [row["ticker"].strip().upper()
                for row in csv.DictReader(fh) if row.get("ticker", "").strip()]


def load_universe(path: Optional[Path] = None,
                  max_tickers: Optional[int] = None) -> tuple[list[str], str]:
    """Return (tickers, source_label). Prefers the refreshed universe file,
    falls back to the bundled snapshot."""
    target = path or UNIVERSE_PATH
    if target.exists():
        try:
            payload = json.loads(target.read_text())
            tickers = [row["ticker"] for row in payload.get("tickers", [])]
            if tickers:
                as_of = (payload.get("as_of") or "")[:10] or "unknown date"
                return tickers[:max_tickers], f"universe.json ({as_of})"
        except Exception as e:
            logger.warning(f"Unreadable {target}: {e}; using bundled universe")
    return _bundled_universe()[:max_tickers], "bundled snapshot"


def universe_age_days(path: Optional[Path] = None) -> Optional[int]:
    """Days since the live universe file was refreshed; None when absent."""
    target = path or UNIVERSE_PATH
    if not target.exists():
        return None
    try:
        as_of = json.loads(target.read_text()).get("as_of")
        return (date.today() - datetime.fromisoformat(as_of).date()).days
    except Exception:
        return None
