"""
Options chain providers for the pin butterfly engine.

Default: CBOE's public delayed-quotes feed — free, no auth, one GET returns
the whole chain with per-contract open interest, bid/ask, IV, and
server-side greeks (quotes 15-min delayed). Paid upgrade: Massive
(massive.com, the rebranded Polygon.io — OPRA-fed NBBO snapshots), used
when an API key is configured. Failure fallback: yfinance (free, but OI is
often missing on fresh weekly chains and quotes go 0/0 after hours).

Providers return per-expiry frames in the shape `butterfly.CHAIN_COLUMNS`
defines, optionally extended with a `gamma` column carrying server-computed
gamma — `score_pins` prefers it over the local Black-Scholes estimate.

Massive API key resolution: MASSIVE_API_KEY env var, then
~/.qpat/config.json (`qpat config set massive-api-key <KEY>`).

Open interest is end-of-day everywhere (OCC publishes it once, overnight) —
CBOE/Massive fix OI being *wrong or absent*, not OI being stale.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from .butterfly import CHAIN_COLUMNS, DEFAULT_IV_FALLBACK, normalize_chain
from .macro_calendar import load_config

logger = logging.getLogger(__name__)

MASSIVE_BASE_URL = "https://api.massive.com"
SNAPSHOT_PAGE_LIMIT = 250  # API maximum
MAX_PAGES = 40             # safety cap: 10K contracts is far beyond any DTE window

CBOE_BASE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options"

# OCC option symbol: root + yymmdd + C/P + strike*1000 zero-padded to 8
OCC_SYMBOL_RE = re.compile(r"^(?P<root>.+?)(?P<exp>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")


class MassiveAPIError(RuntimeError):
    """Massive REST call failed (auth, plan entitlement, or transport)."""


class CboeAPIError(RuntimeError):
    """CBOE delayed-quotes fetch failed (unknown symbol or transport)."""


def get_massive_api_key() -> Optional[str]:
    """Return Massive API key from env var or ~/.qpat/config.json."""
    key = os.environ.get("MASSIVE_API_KEY")
    if key:
        return key
    return load_config().get("massive_api_key")


# ── Provider ABC ─────────────────────────────────────────────────────────────

class OptionsChainProvider(ABC):
    """Base class for options chain providers."""

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def get_chains_window(
        self, ticker: str, start: date, end: date
    ) -> list[tuple[date, pd.DataFrame]]:
        """
        Normalized chains for every listed expiry in [start, end], sorted by
        expiry. Frames carry CHAIN_COLUMNS; providers with server-side greeks
        add a `gamma` column (NaN where unavailable).
        """
        ...


class YFinanceOptionsProvider(OptionsChainProvider):
    """Free yfinance chains. OI is end-of-day and often 0 on freshly listed
    weeklies; quotes go 0/0 after hours (mid_price falls back to lastPrice)."""

    def name(self) -> str:
        return "Yahoo Finance"

    def get_chains_window(
        self, ticker: str, start: date, end: date
    ) -> list[tuple[date, pd.DataFrame]]:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        expiries = [date.fromisoformat(e) for e in (tk.options or ())]
        out: list[tuple[date, pd.DataFrame]] = []
        for exp in expiries:
            if not (start <= exp <= end):
                continue
            try:
                oc = tk.option_chain(exp.isoformat())
                out.append((exp, normalize_chain(oc.calls, oc.puts)))
            except Exception as e:
                logger.warning(f"Could not load chain for {exp}: {e}")
        return out


class MassiveOptionsProvider(OptionsChainProvider):
    """Massive option-chain snapshots: one paginated sweep over the DTE
    window returns OI, NBBO bid/ask, IV, and greeks for every contract."""

    def __init__(self, api_key: str, base_url: str = MASSIVE_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def name(self) -> str:
        return "Massive (OPRA)"

    def _get_json(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "qpat/0.1",
        })
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise MassiveAPIError(
                    f"Massive rejected the API key (HTTP {e.code}) — check "
                    "`qpat config get massive-api-key` and that your plan "
                    "includes options snapshots"
                ) from e
            raise MassiveAPIError(f"Massive HTTP {e.code} for {url}") from e
        except urllib.error.URLError as e:
            raise MassiveAPIError(f"Massive unreachable: {e.reason}") from e

    def _fetch_contracts(self, ticker: str, params: dict) -> list[dict]:
        query = urllib.parse.urlencode({**params, "limit": SNAPSHOT_PAGE_LIMIT})
        url = f"{self.base_url}/v3/snapshot/options/{ticker}?{query}"
        contracts: list[dict] = []
        for _ in range(MAX_PAGES):
            payload = self._get_json(url)
            contracts.extend(payload.get("results") or [])
            url = payload.get("next_url")
            if not url:
                break
        else:
            logger.warning(f"Massive pagination hit the {MAX_PAGES}-page cap for {ticker}")
        return contracts

    def get_chains_window(
        self, ticker: str, start: date, end: date
    ) -> list[tuple[date, pd.DataFrame]]:
        contracts = self._fetch_contracts(ticker, {
            "expiration_date.gte": start.isoformat(),
            "expiration_date.lte": end.isoformat(),
        })
        by_expiry: dict[date, list[dict]] = {}
        for c in contracts:
            exp_str = (c.get("details") or {}).get("expiration_date")
            if not exp_str:
                continue
            by_expiry.setdefault(date.fromisoformat(exp_str), []).append(c)
        return [(exp, massive_chain_frame(by_expiry[exp]))
                for exp in sorted(by_expiry)]


class CboeOptionsProvider(OptionsChainProvider):
    """CBOE's public delayed-quotes feed: one unauthenticated GET returns
    the entire chain (all expiries) with OI, bid/ask, IV, and greeks.
    Quotes are 15-min delayed; OI is as of last close like everywhere."""

    def name(self) -> str:
        return "CBOE (delayed)"

    def _fetch_payload(self, ticker: str) -> dict:
        # Indexes (SPX, VIX, RUT, XSP) live under an underscore prefix.
        for symbol in (ticker, f"_{ticker}"):
            url = f"{CBOE_BASE_URL}/{symbol}.json"
            req = urllib.request.Request(url, headers={"User-Agent": "qpat/0.1"})
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code in (403, 404) and symbol == ticker:
                    continue
                raise CboeAPIError(f"CBOE HTTP {e.code} for {url}") from e
            except urllib.error.URLError as e:
                raise CboeAPIError(f"CBOE unreachable: {e.reason}") from e
        raise CboeAPIError(f"no CBOE option data for {ticker}")

    def get_chains_window(
        self, ticker: str, start: date, end: date
    ) -> list[tuple[date, pd.DataFrame]]:
        payload = self._fetch_payload(ticker)
        options = (payload.get("data") or {}).get("options") or []
        by_expiry = cboe_chain_rows(options)
        return [(exp, _sides_to_frame(by_expiry[exp]))
                for exp in sorted(by_expiry) if start <= exp <= end]


# ── Feed → chain frame mapping (pure, offline-testable) ─────────────────────

def _sides_to_frame(rows: dict[float, dict]) -> pd.DataFrame:
    """Merge per-strike call/put side dicts (keys like `call_oi`, `put_iv`,
    `call_gamma`) into the CHAIN_COLUMNS shape plus a `gamma` column.

    `iv` is the mean of usable per-side IVs (fallback DEFAULT_IV_FALLBACK);
    `gamma` the mean of usable per-side gammas (NaN when absent —
    score_pins then computes Black-Scholes locally). Missing sides are 0.
    """
    records = []
    for strike in sorted(rows):
        row = rows[strike]
        ivs = [v for side in ("call", "put")
               if (v := row.get(f"{side}_iv")) is not None and v > 1e-4]
        gammas = [v for side in ("call", "put")
                  if (v := row.get(f"{side}_gamma")) is not None and v > 0]
        rec = {"strike": strike,
               "iv": float(np.mean(ivs)) if ivs else DEFAULT_IV_FALLBACK,
               "gamma": float(np.mean(gammas)) if gammas else np.nan}
        for side in ("call", "put"):
            rec[f"{side}_oi"] = row.get(f"{side}_oi", 0)
            rec[f"{side}_vol"] = row.get(f"{side}_vol", 0)
            for f in ("bid", "ask", "last"):
                rec[f"{side}_{f}"] = row.get(f"{side}_{f}", 0.0)
        records.append(rec)

    return pd.DataFrame(records, columns=CHAIN_COLUMNS + ["gamma"])


def massive_chain_frame(contracts: list[dict]) -> pd.DataFrame:
    """Map Massive snapshot contracts (one expiry) to the CHAIN_COLUMNS shape
    plus a `gamma` column.

    Per strike: OI from `open_interest`, volume from `day.volume`, NBBO from
    `last_quote.bid/ask`, last from `last_trade.price` (else `day.close`).
    """
    rows: dict[float, dict] = {}
    for c in contracts:
        d = c.get("details") or {}
        ctype = (d.get("contract_type") or "").lower()
        if ctype not in ("call", "put") or d.get("strike_price") is None:
            continue
        strike = float(d["strike_price"])
        day = c.get("day") or {}
        quote = c.get("last_quote") or {}
        trade = c.get("last_trade") or {}
        row = rows.setdefault(strike, {})
        row[f"{ctype}_oi"] = int(c.get("open_interest") or 0)
        row[f"{ctype}_vol"] = int(day.get("volume") or 0)
        row[f"{ctype}_bid"] = float(quote.get("bid") or 0.0)
        row[f"{ctype}_ask"] = float(quote.get("ask") or 0.0)
        row[f"{ctype}_last"] = float(trade.get("price") or day.get("close") or 0.0)
        row[f"{ctype}_iv"] = c.get("implied_volatility")
        row[f"{ctype}_gamma"] = (c.get("greeks") or {}).get("gamma")
    return _sides_to_frame(rows)


def cboe_chain_rows(options: list[dict]) -> dict[date, dict[float, dict]]:
    """Group CBOE delayed-quotes contracts by expiry into per-strike side
    dicts for `_sides_to_frame`.

    Expiry, right, and strike are parsed from the OCC symbol (e.g.
    `SPY260615C00738000`); the root is ignored so weekly roots like SPXW
    fold into the same chain. Unparseable symbols are skipped.
    """
    by_expiry: dict[date, dict[float, dict]] = {}
    for o in options:
        m = OCC_SYMBOL_RE.match(o.get("option") or "")
        if not m:
            continue
        e = m["exp"]
        try:
            exp = date(2000 + int(e[:2]), int(e[2:4]), int(e[4:6]))
        except ValueError:
            continue
        strike = int(m["strike"]) / 1000
        ctype = "call" if m["right"] == "C" else "put"
        row = by_expiry.setdefault(exp, {}).setdefault(strike, {})
        row[f"{ctype}_oi"] = int(o.get("open_interest") or 0)
        row[f"{ctype}_vol"] = int(o.get("volume") or 0)
        row[f"{ctype}_bid"] = float(o.get("bid") or 0.0)
        row[f"{ctype}_ask"] = float(o.get("ask") or 0.0)
        row[f"{ctype}_last"] = float(o.get("last_trade_price") or 0.0)
        row[f"{ctype}_iv"] = o.get("iv")
        row[f"{ctype}_gamma"] = o.get("gamma")
    return by_expiry


# ── Factory + fallback ───────────────────────────────────────────────────────

def get_options_provider(source: str = "auto") -> OptionsChainProvider:
    """Resolve the chain source: explicit name, or `auto` = Massive when an
    API key is configured, else CBOE's free delayed feed."""
    if source == "yfinance":
        return YFinanceOptionsProvider()
    if source == "cboe":
        return CboeOptionsProvider()
    key = get_massive_api_key()
    if source == "massive":
        if not key:
            raise ValueError(
                "No Massive API key configured. Set one with: "
                "qpat config set massive-api-key <KEY> "
                "(or the MASSIVE_API_KEY env var)"
            )
        return MassiveOptionsProvider(key)
    if source != "auto":
        raise ValueError(f"Unknown chain source: {source}. Available: auto, cboe, massive, yfinance")
    return MassiveOptionsProvider(key) if key else CboeOptionsProvider()


class ChainSourceError(RuntimeError):
    """A chain provider failed and qpat refuses to silently degrade."""


def fetch_chains(
    provider: OptionsChainProvider, ticker: str, start: date, end: date
) -> tuple[OptionsChainProvider, list[tuple[date, pd.DataFrame]], list[str]]:
    """Fetch chains from the resolved provider — no silent fallback.

    A CBOE/Massive failure raises :class:`ChainSourceError` instead of
    degrading to yfinance: yfinance chains report zero/absent open interest
    on fresh weeklies and 0/0 quotes after hours, which corrupts OI-weighted
    pin scoring while the output still looks authoritative. Running with
    ``--chain-source yfinance`` opts into the degraded source explicitly —
    it then carries a data-quality warning on the recommendation.

    Returns (provider, chains, warnings).
    """
    warnings: list[str] = []
    if isinstance(provider, YFinanceOptionsProvider):
        warnings.append(
            "yfinance chain source: OI is often zero/absent on fresh weeklies "
            "and quotes go 0/0 after hours — pin scoring may be degraded. "
            "Verify the OI wall on your broker before trusting this pin."
        )
    try:
        return provider, provider.get_chains_window(ticker, start, end), warnings
    except Exception as e:
        if isinstance(provider, YFinanceOptionsProvider):
            raise
        logger.warning(f"{provider.name()} chain fetch failed ({e}); not falling back")
        raise ChainSourceError(
            f"{provider.name()} chain fetch failed: {e}. Not falling back to "
            "yfinance automatically — its chains report zero open interest on "
            "fresh weeklies, which silently corrupts pin scoring. Retry later, "
            "or rerun with --chain-source yfinance to accept degraded data "
            "explicitly."
        ) from e
