"""
POTUS calendar — fetch presidential actions from White House RSS feeds
and manage a local event cache.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .events import EventCategory, MarketEvent

FEED_URLS = {
    "actions": "https://www.whitehouse.gov/presidential-actions/feed/",
    "briefings": "https://www.whitehouse.gov/briefing-room/feed/",
}

MARKET_KEYWORDS = [
    "tariff",
    "trade",
    "executive order",
    "sanction",
    "embargo",
    "tax",
    "import",
    "export",
    "economy",
    "economic",
    "inflation",
    "federal reserve",
    "interest rate",
    "spending",
    "budget",
    "deficit",
    "debt ceiling",
    "regulation",
    "deregulation",
    "energy",
    "oil",
    "infrastructure",
    "chips",
    "semiconductor",
    "technology",
    "crypto",
    "digital asset",
    "banking",
    "financial",
    "stock",
    "market",
    "commerce",
    "industry",
    "manufacturing",
    "steel",
    "aluminum",
    "procurement",
    "defense",
    "doge",
    "government efficiency",
]


def fetch_potus_feed(feed_name: str = "actions") -> list[dict]:
    """Parse a White House RSS feed and return a list of item dicts.

    Each dict has: title, date, link, category, description.
    """
    import feedparser

    url = FEED_URLS.get(feed_name)
    if not url:
        raise ValueError(f"Unknown feed: {feed_name}. Choose from: {list(FEED_URLS)}")

    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries:
        pub_date = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            pub_date = datetime(*entry.published_parsed[:6]).date()
        elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
            pub_date = datetime(*entry.updated_parsed[:6]).date()

        categories = []
        if hasattr(entry, "tags"):
            categories = [t.term for t in entry.tags]

        description = ""
        if hasattr(entry, "summary"):
            description = entry.summary[:200]

        items.append({
            "title": entry.get("title", "Untitled"),
            "date": pub_date,
            "link": entry.get("link", ""),
            "category": ", ".join(categories) if categories else feed_name,
            "description": description,
        })

    return items


def is_market_relevant(item: dict) -> bool:
    """Check if an RSS item is market-relevant based on keywords."""
    text = f"{item.get('title', '')} {item.get('description', '')} {item.get('category', '')}".lower()
    return any(kw in text for kw in MARKET_KEYWORDS)


def load_potus_cache(cache_path: Optional[Path] = None) -> list[dict]:
    """Load cached POTUS events from disk."""
    path = cache_path or Path.home() / ".qpat" / "potus_events.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return []
    return []


def save_potus_cache(events: list[dict], cache_path: Optional[Path] = None):
    """Save POTUS events to disk."""
    path = cache_path or Path.home() / ".qpat" / "potus_events.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(events, indent=2))


def sync_potus_events(
    feed: str = "actions",
    market_only: bool = False,
) -> list[MarketEvent]:
    """Fetch RSS feed, convert to MarketEvents, merge with cache, and save.

    Returns the list of newly added events.
    """
    feeds_to_fetch = list(FEED_URLS) if feed == "all" else [feed]

    all_items = []
    for f in feeds_to_fetch:
        items = fetch_potus_feed(f)
        if market_only:
            items = [it for it in items if is_market_relevant(it)]
        all_items.extend(items)

    # Convert to MarketEvent dicts
    new_event_dicts = []
    for item in all_items:
        if item["date"] is None:
            continue
        evt = MarketEvent(
            name=item["title"][:80],
            date=item["date"],
            category=EventCategory.POTUS,
            description=item["description"][:200],
        )
        new_event_dicts.append(evt.to_dict())

    # Merge with existing cache (dedup by key)
    existing = load_potus_cache()
    existing_keys = {f"potus_{d['date']}" for d in existing}

    added = []
    for d in new_event_dicts:
        key = f"potus_{d['date']}"
        if key not in existing_keys:
            existing.append(d)
            existing_keys.add(key)
            added.append(MarketEvent.from_dict(d))

    save_potus_cache(existing)
    return added
