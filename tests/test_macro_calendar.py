"""Offline tests for the FOMC calendar fetch/parse. No network access."""

from datetime import date

import pytest

from quant_patterns import macro_calendar as mc
from quant_patterns.macro_calendar import (
    FOMC_DATES_FALLBACK,
    get_fomc_dates,
    parse_fomc_calendar,
)


def panel(year: int, meetings: list[tuple[str, str]]) -> str:
    """Minimal HTML mirroring the Fed calendar page's real structure."""
    rows = "".join(
        f'<div class="fomc-meeting__month"><strong>{m}</strong></div>'
        f'<div class="fomc-meeting__date">{d}</div>'
        for m, d in meetings
    )
    return f"<h4>{year} FOMC Meetings</h4>{rows}"


class TestParseFomcCalendar:
    def test_announcement_is_last_meeting_day(self):
        html = panel(2026, [("January", "27-28"), ("March", "17-18*")])
        assert parse_fomc_calendar(html) == [date(2026, 1, 28),
                                             date(2026, 3, 18)]

    def test_year_panels_in_arbitrary_order(self):
        # The real page lists the current year first, then past, then next.
        html = (panel(2026, [("December", "8-9*")])
                + panel(2025, [("January", "28-29")])
                + panel(2027, [("January", "26-27")]))
        assert parse_fomc_calendar(html) == [
            date(2025, 1, 29), date(2026, 12, 9), date(2027, 1, 27)]

    def test_notation_vote_and_asterisk(self):
        html = panel(2025, [("August", "22 (notation vote)"),
                            ("September", "16-17*")])
        assert parse_fomc_calendar(html) == [date(2025, 8, 22),
                                             date(2025, 9, 17)]

    def test_cross_month_meeting_lands_in_second_month(self):
        html = panel(2023, [("October/November", "31-1")])
        assert parse_fomc_calendar(html) == [date(2023, 11, 1)]

    def test_garbage_rows_skipped(self):
        html = panel(2026, [("January", "27-28"), ("NotAMonth", "5-6"),
                            ("March", "TBD")])
        assert parse_fomc_calendar(html) == [date(2026, 1, 28)]

    def test_empty_page(self):
        assert parse_fomc_calendar("<html>maintenance</html>") == []


class TestGetFomcDates:
    def test_live_fetch_wins(self, monkeypatch):
        fetched = [date(2027, 1, 27)]
        monkeypatch.setattr(mc, "fetch_fomc_dates_from_fed", lambda: fetched)
        assert get_fomc_dates() == fetched

    def test_falls_back_when_fetch_fails(self, monkeypatch):
        def boom():
            raise OSError("offline")
        monkeypatch.setattr(mc, "fetch_fomc_dates_from_fed", boom)
        assert get_fomc_dates() == FOMC_DATES_FALLBACK

    def test_fallback_is_sorted_dates(self):
        assert FOMC_DATES_FALLBACK == sorted(FOMC_DATES_FALLBACK)
        assert all(isinstance(d, date) for d in FOMC_DATES_FALLBACK)
