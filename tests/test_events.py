"""Tests for the event catalog and event model."""

import json
from datetime import date

import pytest

from quant_patterns.events import (
    BUILTIN_EVENTS,
    EVENT_CATEGORY_LABELS,
    EventCatalog,
    EventCategory,
    MarketEvent,
)


# ── MarketEvent ───────────────────────────────────────────────────────────────


class TestMarketEvent:
    def test_create_basic(self):
        e = MarketEvent("Test", date(2024, 1, 15), EventCategory.CPI, "desc")
        assert e.name == "Test"
        assert e.date == date(2024, 1, 15)
        assert e.category == EventCategory.CPI
        assert e.ticker_specific is None

    def test_create_with_string_date(self):
        e = MarketEvent("Test", "2024-01-15", EventCategory.CPI)
        assert e.date == date(2024, 1, 15)

    def test_create_with_string_category(self):
        e = MarketEvent("Test", date(2024, 1, 15), "cpi")
        assert e.category == EventCategory.CPI

    def test_ticker_specific(self):
        e = MarketEvent("NVDA ER", date(2024, 2, 21), EventCategory.EARNINGS, ticker_specific="NVDA")
        assert e.ticker_specific == "NVDA"

    def test_key(self):
        e = MarketEvent("Test", date(2024, 1, 15), EventCategory.CPI)
        assert e.key == "cpi_2024-01-15"

    def test_to_dict_roundtrip(self):
        original = MarketEvent("Test CPI", date(2024, 1, 15), EventCategory.CPI, "desc", "SPY")
        d = original.to_dict()
        restored = MarketEvent.from_dict(d)
        assert restored.name == original.name
        assert restored.date == original.date
        assert restored.category == original.category
        assert restored.description == original.description
        assert restored.ticker_specific == original.ticker_specific

    def test_to_dict_format(self):
        e = MarketEvent("Test", date(2024, 1, 15), EventCategory.FOMC, "desc")
        d = e.to_dict()
        assert d["date"] == "2024-01-15"
        assert d["category"] == "fomc"


# ── EventCategory ─────────────────────────────────────────────────────────────


class TestEventCategory:
    def test_all_categories_have_labels(self):
        for cat in EventCategory:
            assert cat in EVENT_CATEGORY_LABELS

    def test_str(self):
        assert str(EventCategory.CPI) == "cpi"
        assert str(EventCategory.FOMC) == "fomc"

    def test_from_value(self):
        assert EventCategory("cpi") == EventCategory.CPI
        assert EventCategory("geopolitical") == EventCategory.GEOPOLITICAL


# ── EventCatalog ──────────────────────────────────────────────────────────────


class TestEventCatalog:
    def test_builtin_events_loaded(self, catalog):
        assert len(catalog.events) == len(BUILTIN_EVENTS)

    def test_filter_by_category(self, catalog):
        fomc = catalog.filter_by_category(EventCategory.FOMC)
        assert len(fomc) > 0
        assert all(e.category == EventCategory.FOMC for e in fomc)

    def test_filter_by_category_sorted_by_date(self, catalog):
        cpi = catalog.filter_by_category(EventCategory.CPI)
        dates = [e.date for e in cpi]
        assert dates == sorted(dates)

    def test_filter_by_ticker_broad_market(self, catalog):
        """Broad market events (ticker_specific=None) should match any ticker."""
        spy_events = catalog.filter_by_ticker("SPY")
        assert any(e.ticker_specific is None for e in spy_events)

    def test_filter_by_ticker_specific(self, catalog):
        """NVDA-specific events should only appear for NVDA queries."""
        nvda_events = catalog.filter_by_ticker("NVDA")
        spy_events = catalog.filter_by_ticker("SPY")
        nvda_specific = [e for e in nvda_events if e.ticker_specific == "NVDA"]
        spy_specific = [e for e in spy_events if e.ticker_specific == "NVDA"]
        assert len(nvda_specific) > 0
        assert len(spy_specific) == 0

    def test_filter_by_ticker_case_insensitive(self, catalog):
        upper = catalog.filter_by_ticker("NVDA")
        lower = catalog.filter_by_ticker("nvda")
        assert len(upper) == len(lower)

    def test_filter_by_date_range(self, catalog):
        start = date(2024, 1, 1)
        end = date(2024, 6, 30)
        results = catalog.filter_by_date_range(start, end)
        assert all(start <= e.date <= end for e in results)

    def test_search_combined_filters(self, catalog):
        results = catalog.search(
            category=EventCategory.EARNINGS,
            ticker="NVDA",
        )
        assert len(results) > 0
        assert all(e.category == EventCategory.EARNINGS for e in results)
        assert all(
            e.ticker_specific is None or e.ticker_specific == "NVDA"
            for e in results
        )

    def test_search_no_results(self, catalog):
        results = catalog.search(category=EventCategory.EARNINGS, ticker="ZZZZZ")
        # Should still return broad market earnings if any exist
        # But there are no broad market earnings in the catalog
        assert all(e.ticker_specific is None for e in results)

    def test_search_returns_sorted(self, catalog):
        results = catalog.search(category=EventCategory.FOMC)
        dates = [e.date for e in results]
        assert dates == sorted(dates)

    def test_categories_property(self, catalog):
        cats = catalog.categories
        assert isinstance(cats, list)
        assert len(cats) > 0
        assert all(isinstance(c, EventCategory) for c in cats)


class TestCustomEvents:
    def test_save_and_load_custom_event(self, tmp_path):
        path = tmp_path / "custom.json"
        catalog1 = EventCatalog(custom_events_path=path)
        initial_count = len(catalog1.events)

        event = MarketEvent("Custom Test", date(2025, 6, 1), EventCategory.CUSTOM, "test")
        catalog1.save_custom_event(event)

        # Reload from disk
        catalog2 = EventCatalog(custom_events_path=path)
        assert len(catalog2.events) == initial_count + 1
        custom = [e for e in catalog2.events if e.name == "Custom Test"]
        assert len(custom) == 1
        assert custom[0].date == date(2025, 6, 1)

    def test_save_multiple_custom_events(self, tmp_path):
        path = tmp_path / "custom.json"
        catalog = EventCatalog(custom_events_path=path)
        initial_count = len(catalog.events)

        for i in range(3):
            event = MarketEvent(f"Custom {i}", date(2025, 1, i + 1), EventCategory.CUSTOM)
            catalog.save_custom_event(event)

        assert len(catalog.events) == initial_count + 3

        # Verify persisted
        data = json.loads(path.read_text())
        assert len(data) == 3

    def test_custom_events_dir_created(self, tmp_path):
        path = tmp_path / "subdir" / "custom.json"
        catalog = EventCatalog(custom_events_path=path)
        event = MarketEvent("Test", date(2025, 1, 1), EventCategory.CUSTOM)
        catalog.save_custom_event(event)
        assert path.exists()


# ── Builtin Data Integrity ────────────────────────────────────────────────────


class TestBuiltinEvents:
    def test_minimum_event_count(self):
        assert len(BUILTIN_EVENTS) >= 50

    def test_geopolitical_includes_us_wars(self):
        geo = [e for e in BUILTIN_EVENTS if e.category == EventCategory.GEOPOLITICAL]
        names = [e.name for e in geo]
        assert "Iraq War Begins" in names
        assert "9/11 Attacks" in names
        assert "Bin Laden Killed" in names
        assert "Afghanistan Withdrawal" in names

    def test_all_events_have_dates(self):
        for e in BUILTIN_EVENTS:
            assert isinstance(e.date, date)

    def test_all_events_have_names(self):
        for e in BUILTIN_EVENTS:
            assert len(e.name) > 0

    def test_earnings_are_ticker_specific(self):
        earnings = [e for e in BUILTIN_EVENTS if e.category == EventCategory.EARNINGS]
        assert all(e.ticker_specific is not None for e in earnings)
