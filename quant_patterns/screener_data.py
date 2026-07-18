"""Bulk OHLCV fetching + disk cache for `qpat screen`.

The screener needs ~420 days of daily bars for ~1,500 tickers — far past what
per-ticker `YFinanceProvider.get_daily_ohlcv` calls can do politely. This
module batches tickers through chunked `yf.download` calls (Yahoo tolerates
batch downloads far better than symbol-by-symbol hammering) and persists the
panel to a versioned pickle at ``~/.qpat/screener_cache/ohlcv.pkl`` so the
steady-state nightly run only fetches the missing tail (usually one day).

Everything here is fail-soft: a ticker that returns nothing is dropped with a
warning count, a corrupt cache triggers a full refetch, never a crash — a
screener missing 2% of its universe is still a screener.

Prices are ADJUSTED (auto_adjust=True): momentum and 52-week-high factors need
split/dividend-consistent history. The screen journal scores relative forward
returns on the same adjusted series, so nothing here needs raw tape levels
(unlike swing/fly, which journal absolute prices).
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".qpat" / "screener_cache"
OHLCV_CACHE_PATH = CACHE_DIR / "ohlcv.pkl"
INFO_CACHE_PATH = CACHE_DIR / "info_cache.json"
CACHE_VERSION = 1

LOOKBACK_DAYS = 420
CHUNK_SIZE = 150
CHUNK_SLEEP_S = 1.0
INFO_TTL_DAYS = 7

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


# ── OHLCV panel cache ────────────────────────────────────────────────────────

def load_ohlcv_cache(path: Path = OHLCV_CACHE_PATH) -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("version") != CACHE_VERSION:
            logger.info("Screener cache version mismatch — full refetch")
            return {}
        return payload["frames"]
    except Exception as e:
        logger.warning(f"Screener cache unreadable ({e}) — full refetch")
        return {}


def save_ohlcv_cache(frames: dict[str, pd.DataFrame],
                     path: Path = OHLCV_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"version": CACHE_VERSION, "frames": frames}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _normalize_single(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """One ticker's slice of a yf.download result -> clean OHLCV or None."""
    if df is None or df.empty:
        return None
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        return None
    df = df[OHLCV_COLUMNS].dropna(how="all")
    df = df[df["Close"].notna()]
    if df.empty:
        return None
    df.index.name = "Date"
    return df


def _split_batch(batch: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Split a yf.download result into per-ticker frames. group_by='ticker'
    yields MultiIndex columns for multi-ticker calls; a single ticker can
    come back flat depending on the yfinance version — handle both."""
    out: dict[str, pd.DataFrame] = {}
    if batch is None or batch.empty:
        return out
    if not isinstance(batch.columns, pd.MultiIndex):
        if len(tickers) == 1:
            df = _normalize_single(batch)
            if df is not None:
                out[tickers[0]] = df
        return out
    have = set(batch.columns.get_level_values(0))
    for ticker in tickers:
        if ticker not in have:
            continue
        df = _normalize_single(batch[ticker])
        if df is not None:
            out[ticker] = df
    return out


def _download_chunk(tickers: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    """One chunked yf.download with a single retry. end is inclusive."""
    kwargs = dict(start=start.isoformat(),
                  end=(end + timedelta(days=1)).isoformat(),
                  auto_adjust=True, group_by="ticker", threads=True,
                  progress=False)
    for attempt in (1, 2):
        try:
            batch = yf.download(tickers=" ".join(tickers), **kwargs)
            return _split_batch(batch, tickers)
        except Exception as e:
            if attempt == 2:
                logger.warning(f"Chunk of {len(tickers)} failed twice: {e}")
                return {}
            time.sleep(5.0)
    return {}


def fetch_universe_ohlcv(
    tickers: list[str],
    lookback_days: int = LOOKBACK_DAYS,
    force: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
    chunk_size: int = CHUNK_SIZE,
    cache_path: Path = OHLCV_CACHE_PATH,
    today: Optional[date] = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Return ({ticker: OHLCV frame}, warnings) covering `lookback_days`.

    Warm path: tickers whose cached frame already reaches the last expected
    session are untouched; stale tickers fetch only the missing tail (grouped
    into one shared start date — the overlap rows dedup on append). Only
    tickers with no cached frame at all fetch the full window: a frame that
    starts later than the window (recent IPO, or a cache built with a shorter
    lookback) is kept as-is rather than re-fetched nightly — use
    `force=True` to deepen the history.
    """
    today = today or date.today()
    start_full = today - timedelta(days=lookback_days)
    cached = {} if force else load_ohlcv_cache(cache_path)
    warnings: list[str] = []

    # A cached frame is fresh when it reaches the most recent weekday; a
    # holiday makes every ticker look one day stale, and the resulting
    # one-day fetch is cheap and returns nothing — acceptable.
    last_expected = today
    while last_expected.weekday() >= 5:
        last_expected -= timedelta(days=1)

    fresh: dict[str, pd.DataFrame] = {}
    stale: list[str] = []
    missing: list[str] = []
    stale_start = last_expected
    for ticker in tickers:
        df = cached.get(ticker)
        if df is None or df.empty:
            missing.append(ticker)
            continue
        last_have = df.index[-1].date()
        if last_have >= last_expected:
            fresh[ticker] = df
        else:
            stale.append(ticker)
            stale_start = min(stale_start, last_have + timedelta(days=1))

    frames = dict(fresh)

    def run_chunks(symbols: list[str], start: date, label: str) -> None:
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            if progress_cb:
                progress_cb(f"{label}: {i + len(chunk)}/{len(symbols)}")
            got = _download_chunk(chunk, start, today)
            for ticker in chunk:
                new = got.get(ticker)
                if new is None:
                    continue
                prior = cached.get(ticker)
                if prior is not None and not prior.empty:
                    combined = pd.concat([prior, new])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    frames[ticker] = combined.sort_index()
                else:
                    frames[ticker] = new
            if i + chunk_size < len(symbols):
                time.sleep(CHUNK_SLEEP_S)

    if stale:
        run_chunks(stale, stale_start, "updating")
    if missing:
        run_chunks(missing, start_full, "fetching")

    # Prune to the lookback window and drop dead frames.
    cutoff = pd.Timestamp(start_full)
    frames = {t: df[df.index >= cutoff] for t, df in frames.items()}
    frames = {t: df for t, df in frames.items() if not df.empty}

    dropped = len(tickers) - len(frames)
    if dropped:
        warnings.append(f"{dropped}/{len(tickers)} tickers returned no data "
                        "and were dropped")
    if frames:
        save_ohlcv_cache(frames, cache_path)
    return frames, warnings


# ── Finalist enrichment (.info is ~1 slow request per ticker) ────────────────

def _load_info_cache() -> dict:
    if not INFO_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(INFO_CACHE_PATH.read_text())
    except Exception:
        return {}


def enrich_finalists(tickers: list[str],
                     ttl_days: int = INFO_TTL_DAYS) -> dict[str, dict]:
    """Earnings date / short interest / sector for the top-K finalists only.

    Cached to info_cache.json with a TTL so the nightly run typically hits
    Yahoo for just the handful of names newly entering the top-K. Every
    failure degrades to empty fields — enrichment never blocks the screen.
    """
    cache = _load_info_cache()
    now = datetime.now()
    out: dict[str, dict] = {}
    dirty = False
    for ticker in tickers:
        row = cache.get(ticker)
        if row:
            try:
                age = now - datetime.fromisoformat(row["fetched_at"])
                if age.days < ttl_days:
                    out[ticker] = row
                    continue
            except Exception:
                pass
        info_row = {"fetched_at": now.isoformat(timespec="seconds"),
                    "sector": None, "earnings_date": None,
                    "short_pct_float": None}
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            info_row["sector"] = info.get("sector")
            spf = info.get("shortPercentOfFloat")
            info_row["short_pct_float"] = round(spf * 100, 1) if spf else None
            try:
                cal = tk.calendar
                dates = (cal or {}).get("Earnings Date") if isinstance(cal, dict) else None
                if dates:
                    info_row["earnings_date"] = min(dates).isoformat()
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"info fetch failed for {ticker}: {e}")
        out[ticker] = info_row
        cache[ticker] = info_row
        dirty = True
    if dirty:
        try:
            INFO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            INFO_CACHE_PATH.write_text(json.dumps(cache, indent=1))
        except Exception as e:
            logger.warning(f"Could not write info cache: {e}")
    return out
