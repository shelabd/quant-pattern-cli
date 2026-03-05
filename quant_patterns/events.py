"""
Event catalog for key market-moving events.

Each event has a date, category, and description. Events are used as anchor points
for pattern matching across historical price data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class EventCategory(str, Enum):
    CPI = "cpi"
    PPI = "ppi"
    FOMC = "fomc"
    NFP = "nfp"  # Non-Farm Payrolls
    EARNINGS = "earnings"
    ELECTION = "election"
    GEOPOLITICAL = "geopolitical"
    GDP = "gdp"
    RETAIL_SALES = "retail_sales"
    OPEC = "opec"
    CRYPTO = "crypto"
    CUSTOM = "custom"

    def __str__(self) -> str:
        return self.value


EVENT_CATEGORY_LABELS = {
    EventCategory.CPI: "CPI Release",
    EventCategory.PPI: "PPI Release",
    EventCategory.FOMC: "FOMC Decision / Fed Speeches",
    EventCategory.NFP: "Non-Farm Payrolls",
    EventCategory.EARNINGS: "Earnings Report",
    EventCategory.ELECTION: "Election / Political Event",
    EventCategory.GEOPOLITICAL: "Geopolitical Event (War, Sanctions, etc.)",
    EventCategory.GDP: "GDP Release",
    EventCategory.RETAIL_SALES: "Retail Sales Data",
    EventCategory.OPEC: "OPEC Decision",
    EventCategory.CRYPTO: "Crypto Event (Hacks, Collapses, Regulation, etc.)",
    EventCategory.CUSTOM: "Custom Event",
}


@dataclass
class MarketEvent:
    name: str
    date: date
    category: EventCategory
    description: str = ""
    ticker_specific: Optional[str] = None  # None = broad market, else specific ticker

    def __post_init__(self):
        if isinstance(self.date, str):
            self.date = datetime.strptime(self.date, "%Y-%m-%d").date()
        if isinstance(self.category, str):
            self.category = EventCategory(self.category)

    @property
    def key(self) -> str:
        return f"{self.category.value}_{self.date.isoformat()}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["date"] = self.date.isoformat()
        d["category"] = self.category.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MarketEvent":
        return cls(**d)


# ── Built-in Event Database ────────────────────────────────────────────────────
# This is a curated set of major market events. Users can extend via custom events.

BUILTIN_EVENTS: list[MarketEvent] = [
    # ── CPI Releases (2022-2026 selection) ──────────────────────────────────
    MarketEvent("CPI Jan 2022", "2022-02-10", EventCategory.CPI, "YoY 7.5% - 40yr high"),
    MarketEvent("CPI Jun 2022", "2022-07-13", EventCategory.CPI, "YoY 9.1% - peak inflation"),
    MarketEvent("CPI Dec 2022", "2023-01-12", EventCategory.CPI, "YoY 6.5% - cooling trend"),
    MarketEvent("CPI Jun 2023", "2023-07-12", EventCategory.CPI, "YoY 3.0% - significant drop"),
    MarketEvent("CPI Sep 2023", "2023-10-12", EventCategory.CPI, "YoY 3.7%"),
    MarketEvent("CPI Dec 2023", "2024-01-11", EventCategory.CPI, "YoY 3.4%"),
    MarketEvent("CPI Mar 2024", "2024-04-10", EventCategory.CPI, "YoY 3.5% - sticky"),
    MarketEvent("CPI Jun 2024", "2024-07-11", EventCategory.CPI, "YoY 3.0%"),
    MarketEvent("CPI Sep 2024", "2024-10-10", EventCategory.CPI, "YoY 2.4%"),
    MarketEvent("CPI Dec 2024", "2025-01-15", EventCategory.CPI, "YoY 2.9%"),

    # ── FOMC Decisions ──────────────────────────────────────────────────────
    MarketEvent("FOMC Mar 2022 - First Hike", "2022-03-16", EventCategory.FOMC, "25bp hike, start of cycle"),
    MarketEvent("FOMC Jun 2022 - 75bp", "2022-06-15", EventCategory.FOMC, "75bp hike - aggressive"),
    MarketEvent("FOMC Jul 2022 - 75bp", "2022-07-27", EventCategory.FOMC, "75bp hike"),
    MarketEvent("FOMC Sep 2022 - 75bp", "2022-09-21", EventCategory.FOMC, "75bp hike"),
    MarketEvent("FOMC Nov 2022 - 75bp", "2022-11-02", EventCategory.FOMC, "75bp hike"),
    MarketEvent("FOMC Dec 2022 - 50bp", "2022-12-14", EventCategory.FOMC, "50bp hike - slowdown"),
    MarketEvent("FOMC Jul 2023 - Last Hike", "2023-07-26", EventCategory.FOMC, "25bp - final hike of cycle"),
    MarketEvent("FOMC Sep 2024 - First Cut", "2024-09-18", EventCategory.FOMC, "50bp cut - pivot"),
    MarketEvent("FOMC Nov 2024", "2024-11-07", EventCategory.FOMC, "25bp cut"),
    MarketEvent("FOMC Dec 2024", "2024-12-18", EventCategory.FOMC, "25bp cut, hawkish guidance"),

    # ── Non-Farm Payrolls ───────────────────────────────────────────────────
    MarketEvent("NFP Jan 2023", "2023-02-03", EventCategory.NFP, "517K jobs - blowout"),
    MarketEvent("NFP Jan 2024", "2024-02-02", EventCategory.NFP, "353K jobs - strong"),
    MarketEvent("NFP Jul 2024", "2024-08-02", EventCategory.NFP, "114K - recession fears"),
    MarketEvent("NFP Oct 2024", "2024-11-01", EventCategory.NFP, "12K - hurricane distorted"),

    # ── Geopolitical Events ─────────────────────────────────────────────────
    MarketEvent("Russia Invades Ukraine", "2022-02-24", EventCategory.GEOPOLITICAL, "Full-scale invasion begins"),
    MarketEvent("SVB Collapse", "2023-03-10", EventCategory.GEOPOLITICAL, "Silicon Valley Bank failure"),
    MarketEvent("Hamas Attack on Israel", "2023-10-07", EventCategory.GEOPOLITICAL, "Oct 7 attack"),
    MarketEvent("Iran-Israel Escalation", "2024-04-13", EventCategory.GEOPOLITICAL, "Iran drone/missile attack on Israel"),
    MarketEvent("Trump Tariffs Escalation", "2025-02-01", EventCategory.GEOPOLITICAL, "New tariff announcements"),
    MarketEvent("US-Israel Strike Iran", "2026-02-28", EventCategory.GEOPOLITICAL, "Operation Roaring Lion/Epic Fury, joint strikes on Iran"),

    # ── US Wars & Military Operations ──────────────────────────────────────
    MarketEvent("Gulf War Begins", "1991-01-17", EventCategory.GEOPOLITICAL, "Operation Desert Storm air campaign starts"),
    MarketEvent("Gulf War Ceasefire", "1991-02-28", EventCategory.GEOPOLITICAL, "Bush declares ceasefire after 100-hr ground war"),
    MarketEvent("US Strikes Iraq (Desert Fox)", "1998-12-16", EventCategory.GEOPOLITICAL, "4-day bombing campaign over WMD inspections"),
    MarketEvent("9/11 Attacks", "2001-09-11", EventCategory.GEOPOLITICAL, "World Trade Center & Pentagon attacks, markets closed 4 days"),
    MarketEvent("Afghanistan War Begins", "2001-10-07", EventCategory.GEOPOLITICAL, "Operation Enduring Freedom launches"),
    MarketEvent("Iraq War Begins", "2003-03-20", EventCategory.GEOPOLITICAL, "Operation Iraqi Freedom, shock and awe"),
    MarketEvent("Saddam Hussein Captured", "2003-12-14", EventCategory.GEOPOLITICAL, "Captured in Tikrit, announced Dec 14"),
    MarketEvent("Iraq Surge Announced", "2007-01-10", EventCategory.GEOPOLITICAL, "Bush announces 20K troop surge"),
    MarketEvent("Bin Laden Killed", "2011-05-02", EventCategory.GEOPOLITICAL, "US Navy SEAL raid in Abbottabad"),
    MarketEvent("US Strikes Syria (Chemical)", "2017-04-07", EventCategory.GEOPOLITICAL, "59 Tomahawk missiles at Shayrat airbase"),
    MarketEvent("Soleimani Assassination", "2020-01-03", EventCategory.GEOPOLITICAL, "US drone strike kills Iranian general"),
    MarketEvent("Afghanistan Withdrawal", "2021-08-15", EventCategory.GEOPOLITICAL, "Kabul falls to Taliban, US evacuation"),

    # ── Elections ────────────────────────────────────────────────────────────
    MarketEvent("US Midterms 2022", "2022-11-08", EventCategory.ELECTION, "US midterm elections"),
    MarketEvent("US Presidential 2024", "2024-11-05", EventCategory.ELECTION, "Trump wins 2024 election"),

    # ── GDP Releases ────────────────────────────────────────────────────────
    MarketEvent("GDP Q1 2022", "2022-04-28", EventCategory.GDP, "-1.6% - contraction"),
    MarketEvent("GDP Q2 2022", "2022-07-28", EventCategory.GDP, "-0.6% - technical recession"),
    MarketEvent("GDP Q3 2024", "2024-10-30", EventCategory.GDP, "2.8% advance estimate"),

    # ── Major Earnings (ticker-specific) ────────────────────────────────────
    MarketEvent("NVDA Q4 FY24 Earnings", "2024-02-21", EventCategory.EARNINGS, "Revenue $22.1B, beat massively", "NVDA"),
    MarketEvent("NVDA Q1 FY25 Earnings", "2024-05-22", EventCategory.EARNINGS, "Revenue $26B, 10:1 split announced", "NVDA"),
    MarketEvent("NVDA Q2 FY25 Earnings", "2024-08-28", EventCategory.EARNINGS, "Revenue $30B", "NVDA"),
    MarketEvent("NVDA Q3 FY25 Earnings", "2024-11-20", EventCategory.EARNINGS, "Revenue $35.1B", "NVDA"),
    MarketEvent("AAPL Q4 FY24 Earnings", "2024-10-31", EventCategory.EARNINGS, "Revenue $94.9B", "AAPL"),
    MarketEvent("TSLA Q3 2024 Earnings", "2024-10-23", EventCategory.EARNINGS, "Beat expectations, stock surged", "TSLA"),
    MarketEvent("META Q3 2024 Earnings", "2024-10-30", EventCategory.EARNINGS, "Revenue $40.6B", "META"),
    MarketEvent("MSFT Q1 FY25 Earnings", "2024-10-30", EventCategory.EARNINGS, "Revenue $65.6B", "MSFT"),
    MarketEvent("GOOGL Q3 2024 Earnings", "2024-10-29", EventCategory.EARNINGS, "Cloud growth strong", "GOOGL"),
    MarketEvent("AMZN Q3 2024 Earnings", "2024-10-31", EventCategory.EARNINGS, "AWS reacceleration", "AMZN"),

    # ── PPI ──────────────────────────────────────────────────────────────────
    MarketEvent("PPI Jun 2022", "2022-07-14", EventCategory.PPI, "YoY 11.3%"),
    MarketEvent("PPI Dec 2023", "2024-01-12", EventCategory.PPI, "YoY 1.0%"),
    MarketEvent("PPI Sep 2024", "2024-10-11", EventCategory.PPI, "YoY 1.8%"),

    # ── OPEC ─────────────────────────────────────────────────────────────────
    MarketEvent("OPEC+ Surprise Cut", "2023-04-03", EventCategory.OPEC, "1.16M bpd surprise cut"),
    MarketEvent("OPEC+ Production Cut", "2024-06-02", EventCategory.OPEC, "Extended cuts through 2025"),

    # ── Crypto Events ──────────────────────────────────────────────────────
    MarketEvent("Bitcoin Futures Launch (CME)", "2017-12-18", EventCategory.CRYPTO, "CME BTC futures go live, BTC near $20K ATH"),
    MarketEvent("Crypto Winter Begins", "2018-01-16", EventCategory.CRYPTO, "BTC crashes from $20K, broad crypto selloff"),
    MarketEvent("China Crypto Ban", "2021-05-21", EventCategory.CRYPTO, "China bans financial institutions from crypto services"),
    MarketEvent("China Mining Crackdown", "2021-06-21", EventCategory.CRYPTO, "Sichuan orders crypto miners to shut down, hashrate crashes"),
    MarketEvent("El Salvador BTC Legal Tender", "2021-09-07", EventCategory.CRYPTO, "El Salvador adopts Bitcoin as legal tender, BTC flash crashes"),
    MarketEvent("BTC All-Time High $69K", "2021-11-10", EventCategory.CRYPTO, "Bitcoin hits $68,789 ATH before reversal"),
    MarketEvent("Luna/Terra Collapse", "2022-05-09", EventCategory.CRYPTO, "UST depeg triggers death spiral, $40B wiped out"),
    MarketEvent("Three Arrows Capital Liquidation", "2022-06-15", EventCategory.CRYPTO, "3AC defaults, crypto contagion spreads"),
    MarketEvent("Celsius Network Bankruptcy", "2022-07-13", EventCategory.CRYPTO, "Celsius files Ch.11, $4.7B in liabilities"),
    MarketEvent("FTX Collapse", "2022-11-08", EventCategory.CRYPTO, "CoinDesk Alameda report triggers bank run, FTX halts withdrawals"),
    MarketEvent("FTX Bankruptcy Filed", "2022-11-11", EventCategory.CRYPTO, "FTX files Ch.11, SBF resigns, $8B hole"),
    MarketEvent("SEC Sues Binance", "2023-06-05", EventCategory.CRYPTO, "SEC files 13 charges against Binance and CZ"),
    MarketEvent("SEC Sues Coinbase", "2023-06-06", EventCategory.CRYPTO, "SEC sues Coinbase for operating unregistered exchange"),
    MarketEvent("Grayscale Wins SEC Lawsuit", "2023-08-29", EventCategory.CRYPTO, "Court rules SEC wrong to deny GBTC ETF conversion"),
    MarketEvent("Spot BTC ETF Approved", "2024-01-10", EventCategory.CRYPTO, "SEC approves 11 spot Bitcoin ETFs"),
    MarketEvent("Bitcoin Halving 2024", "2024-04-20", EventCategory.CRYPTO, "4th halving: block reward drops to 3.125 BTC"),
    MarketEvent("Spot ETH ETF Approved", "2024-05-23", EventCategory.CRYPTO, "SEC approves spot Ethereum ETFs"),
    MarketEvent("BTC New ATH $100K", "2024-12-05", EventCategory.CRYPTO, "Bitcoin crosses $100K for the first time"),
]


class EventCatalog:
    """Manages built-in + custom events with filtering and search."""

    def __init__(self, custom_events_path: Optional[Path] = None):
        self.events: list[MarketEvent] = list(BUILTIN_EVENTS)
        self.custom_path = custom_events_path or Path.home() / ".qpat" / "custom_events.json"
        self._load_custom()

    def _load_custom(self):
        if self.custom_path.exists():
            try:
                data = json.loads(self.custom_path.read_text())
                for d in data:
                    self.events.append(MarketEvent.from_dict(d))
            except Exception:
                pass

    def save_custom_event(self, event: MarketEvent):
        self.custom_path.parent.mkdir(parents=True, exist_ok=True)
        custom = []
        if self.custom_path.exists():
            try:
                custom = json.loads(self.custom_path.read_text())
            except Exception:
                pass
        custom.append(event.to_dict())
        self.custom_path.write_text(json.dumps(custom, indent=2))
        self.events.append(event)

    def filter_by_category(self, category: EventCategory) -> list[MarketEvent]:
        return sorted(
            [e for e in self.events if e.category == category],
            key=lambda e: e.date,
        )

    def filter_by_ticker(self, ticker: str) -> list[MarketEvent]:
        ticker = ticker.upper()
        return sorted(
            [e for e in self.events if e.ticker_specific is None or e.ticker_specific == ticker],
            key=lambda e: e.date,
        )

    def filter_by_date_range(self, start: date, end: date) -> list[MarketEvent]:
        return sorted(
            [e for e in self.events if start <= e.date <= end],
            key=lambda e: e.date,
        )

    def search(
        self,
        category: Optional[EventCategory] = None,
        ticker: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> list[MarketEvent]:
        results = self.events
        if category:
            results = [e for e in results if e.category == category]
        if ticker:
            t = ticker.upper()
            results = [e for e in results if e.ticker_specific is None or e.ticker_specific == t]
        if start:
            results = [e for e in results if e.date >= start]
        if end:
            results = [e for e in results if e.date <= end]
        return sorted(results, key=lambda e: e.date)

    @property
    def categories(self) -> list[EventCategory]:
        return sorted(set(e.category for e in self.events), key=lambda c: c.value)
