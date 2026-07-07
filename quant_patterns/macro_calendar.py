"""
Proximity-aware macro calendar for auto-detecting nearest market events.

Fetches upcoming macro release dates from FRED API (CPI, PPI, NFP, GDP,
Retail Sales), hardcodes FOMC dates, and uses yfinance for ticker-specific
earnings. Results are cached to ~/.qpat/macro_calendar.json (24h TTL).
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from .events import EventCategory

logger = logging.getLogger(__name__)

QPAT_DIR = Path.home() / ".qpat"
CONFIG_PATH = QPAT_DIR / "config.json"
MACRO_CACHE_PATH = QPAT_DIR / "macro_calendar.json"
CACHE_TTL_HOURS = 24

# FRED release IDs for macro categories
FRED_RELEASE_MAP: dict[EventCategory, int] = {
    EventCategory.CPI: 10,
    EventCategory.PPI: 46,
    EventCategory.NFP: 50,
    EventCategory.GDP: 53,
    EventCategory.RETAIL_SALES: 9,
}

# Impact ranking for tie-breaking (lower = higher impact)
IMPACT_RANK: dict[EventCategory, int] = {
    EventCategory.FOMC: 0,
    EventCategory.CPI: 1,
    EventCategory.NFP: 2,
    EventCategory.EARNINGS: 3,
    EventCategory.GDP: 4,
    EventCategory.PPI: 5,
    EventCategory.RETAIL_SALES: 6,
}

# FOMC meeting dates (announcement day) — FALLBACK ONLY, used when the live
# fetch from the Fed's calendar page fails. Snapshot of the published
# schedule; the live page is authoritative (it revealed this list's Dec 2026
# entry was stale the day the fetch was added).
FOMC_DATES_FALLBACK: list[date] = [
    # 2025
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 10, 29),
    date(2025, 12, 17),
    # 2026
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 16),
]


# ── Config management ─────────────────────────────────────────────────────────


def load_config() -> dict:
    """Read ~/.qpat/config.json, returning {} if missing or corrupt."""
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_config(cfg: dict):
    """Write config dict to ~/.qpat/config.json."""
    QPAT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def get_fred_api_key() -> Optional[str]:
    """Return FRED API key from env var or config file."""
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    return load_config().get("fred_api_key")


# ── FRED API ──────────────────────────────────────────────────────────────────


def fetch_fred_release_dates(
    release_id: int, api_key: str, limit: int = 20,
) -> list[date]:
    """Fetch upcoming release dates from FRED for a given release ID."""
    today = date.today().isoformat()
    url = (
        f"https://api.stlouisfed.org/fred/release/dates"
        f"?release_id={release_id}"
        f"&include_release_dates_with_no_data=true"
        f"&sort_order=asc"
        f"&realtime_start={today}"
        f"&limit={limit}"
        f"&api_key={api_key}"
        f"&file_type=json"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "qpat/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        return [
            datetime.strptime(rd["date"], "%Y-%m-%d").date()
            for rd in data.get("release_dates", [])
        ]
    except Exception as e:
        logger.warning("FRED fetch failed for release %d: %s", release_id, e)
        return []


# ── FOMC dates ────────────────────────────────────────────────────────────────

FOMC_CAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

# Document-order token stream: year headings ("2026 FOMC Meetings") precede
# their meetings; each meeting is a __month div followed by a __date div.
_FOMC_TOKEN_RE = re.compile(
    r"(\d{4})\s+FOMC\s+Meetings"
    r"|fomc-meeting__month[^>]*>\s*(?:<strong>)?\s*([^<]+)"
    r"|fomc-meeting__date[^>]*>\s*([^<]+)"
)


def _parse_fomc_meeting(year: int, month_name: str, date_text: str) -> Optional[date]:
    """One meeting row -> announcement date (the meeting's LAST day).

    Handles "27-28", "17-18*" (SEP asterisk), "22 (notation vote)", and
    cross-month rows like "October/November" "31-1" (announcement falls in
    the second month when the day range wraps).
    """
    days = [int(n) for n in re.findall(r"\d+", date_text.split("(")[0])]
    if not days:
        return None
    months = [p.strip() for p in month_name.split("/")]
    wraps = len(days) > 1 and days[-1] < days[0]
    name = months[-1] if (len(months) > 1 and wraps) else months[0]
    try:
        month = datetime.strptime(name[:3], "%b").month
        return date(year, month, days[-1])
    except ValueError:
        return None


def parse_fomc_calendar(html: str) -> list[date]:
    """Extract FOMC announcement dates from the Fed's calendar page HTML.

    Pure and offline-testable. Year panels appear in arbitrary order
    (current year first), so the year heading resets the running state.
    """
    year: Optional[int] = None
    month_name: Optional[str] = None
    found: set[date] = set()
    for y, m, d in _FOMC_TOKEN_RE.findall(html):
        if y:
            year, month_name = int(y), None
        elif m:
            month_name = m.strip()
        elif d and year and month_name:
            parsed = _parse_fomc_meeting(year, month_name, d)
            if parsed:
                found.add(parsed)
            month_name = None
    return sorted(found)


def fetch_fomc_dates_from_fed(timeout: int = 10) -> list[date]:
    """Live FOMC schedule from federalreserve.gov (~5y history + the next
    year's tentative schedule). Raises on fetch/parse failure."""
    req = urllib.request.Request(FOMC_CAL_URL, headers={"User-Agent": "qpat/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    dates = parse_fomc_calendar(html)
    if not dates:
        raise ValueError("no FOMC meetings parsed from the Fed calendar page")
    return dates


def get_fomc_dates() -> list[date]:
    """FOMC announcement dates: live from the Fed's published calendar,
    hardcoded fallback only when the fetch or parse fails. Results land in
    the 24h macro cache, so the Fed page is hit at most once a day."""
    try:
        return fetch_fomc_dates_from_fed()
    except Exception as e:
        logger.warning("FOMC calendar fetch failed (%s) — using hardcoded "
                       "fallback through %s", e, FOMC_DATES_FALLBACK[-1])
        return list(FOMC_DATES_FALLBACK)


# ── Earnings via yfinance ─────────────────────────────────────────────────────


def fetch_next_earnings_date(ticker: str) -> Optional[date]:
    """Return the next earnings date for *ticker* via yfinance, or None."""
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            return None
        # yfinance returns a dict with 'Earnings Date' as a list of Timestamps
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date")
            if dates and len(dates) > 0:
                return dates[0].date() if hasattr(dates[0], "date") else dates[0]
        return None
    except Exception as e:
        logger.warning("yfinance earnings lookup failed for %s: %s", ticker, e)
        return None


# ── Cache ─────────────────────────────────────────────────────────────────────


def load_macro_cache() -> Optional[dict]:
    """Load cached macro calendar, or None if missing."""
    if MACRO_CACHE_PATH.exists():
        try:
            return json.loads(MACRO_CACHE_PATH.read_text())
        except Exception:
            return None
    return None


def save_macro_cache(releases: dict[str, list[str]]):
    """Save releases dict to cache with timestamp."""
    QPAT_DIR.mkdir(parents=True, exist_ok=True)
    cache = {
        "fetched_at": datetime.utcnow().isoformat(),
        "releases": releases,
    }
    MACRO_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def is_cache_stale(cache: dict) -> bool:
    """Return True if cache is older than CACHE_TTL_HOURS."""
    try:
        fetched = datetime.fromisoformat(cache["fetched_at"])
        return datetime.utcnow() - fetched > timedelta(hours=CACHE_TTL_HOURS)
    except Exception:
        return True


# ── Sync orchestrator ─────────────────────────────────────────────────────────


def sync_macro_calendar(force: bool = False) -> dict[str, list[str]]:
    """Fetch macro calendar dates and cache them.

    Returns a dict mapping category value → list of ISO date strings.
    Graceful: no API key → only FOMC; FRED error → partial results; no
    network → stale cache.
    """
    cache = load_macro_cache()
    if not force and cache and not is_cache_stale(cache):
        return cache["releases"]

    releases: dict[str, list[str]] = {}

    # FOMC — always available (hardcoded)
    releases[EventCategory.FOMC.value] = [d.isoformat() for d in get_fomc_dates()]

    # FRED releases — need API key
    api_key = get_fred_api_key()
    if api_key:
        for cat, rid in FRED_RELEASE_MAP.items():
            dates = fetch_fred_release_dates(rid, api_key)
            if dates:
                releases[cat.value] = [d.isoformat() for d in dates]
            else:
                logger.warning("No dates fetched for %s (release %d)", cat.value, rid)
    else:
        logger.info("No FRED API key configured — only FOMC dates available. "
                     "Run: qpat config set fred-api-key <KEY>")

    # Only save if we got something beyond stale cache
    if releases:
        save_macro_cache(releases)

    # Fall back to stale cache if sync returned nothing useful
    if not releases and cache:
        logger.info("Using stale macro cache (sync produced no results)")
        return cache.get("releases", {})

    return releases


# ── Proximity lookup ──────────────────────────────────────────────────────────


@dataclass
class NearestEvent:
    category: EventCategory
    event_date: date
    distance_days: int  # negative = past, positive = future
    source: str  # "fred", "fomc", "earnings"


def _find_nearest_date(
    dates: list[date], reference: date,
) -> Optional[tuple[date, int]]:
    """Return (nearest_date, signed_distance) or None."""
    if not dates:
        return None
    best = min(dates, key=lambda d: abs((d - reference).days))
    return best, (best - reference).days


def find_nearest_macro_event(
    ticker: str,
    reference_date: Optional[date] = None,
    max_distance_days: int = 14,
) -> Optional[NearestEvent]:
    """Find the nearest upcoming/recent macro event to *reference_date*.

    Algorithm:
    1. Load/auto-refresh cache
    2. For each category's dates, find nearest to reference_date
    3. Check yfinance for ticker-specific earnings
    4. Rank by: abs(distance) asc → future over past → impact rank
    5. Return top candidate, or None if nothing within max_distance_days
    """
    if reference_date is None:
        reference_date = date.today()

    releases = sync_macro_calendar()
    candidates: list[NearestEvent] = []

    # Check each cached category
    for cat_val, date_strs in releases.items():
        try:
            cat = EventCategory(cat_val)
        except ValueError:
            continue
        dates = [datetime.strptime(ds, "%Y-%m-%d").date() for ds in date_strs]
        result = _find_nearest_date(dates, reference_date)
        if result:
            evt_date, dist = result
            if abs(dist) <= max_distance_days:
                source = "fomc" if cat == EventCategory.FOMC else "fred"
                candidates.append(NearestEvent(cat, evt_date, dist, source))

    # Check ticker-specific earnings
    earnings_date = fetch_next_earnings_date(ticker)
    if earnings_date:
        dist = (earnings_date - reference_date).days
        if abs(dist) <= max_distance_days:
            candidates.append(
                NearestEvent(EventCategory.EARNINGS, earnings_date, dist, "earnings")
            )

    if not candidates:
        return None

    # Sort: abs(distance) asc, future preferred over past, impact rank
    candidates.sort(key=lambda c: (
        abs(c.distance_days),
        0 if c.distance_days >= 0 else 1,  # future before past
        IMPACT_RANK.get(c.category, 99),
    ))

    return candidates[0]
