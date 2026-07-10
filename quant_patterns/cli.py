"""
CLI entry point for Quant Pattern Analysis.

Commands:
  analyze   - Run full pattern analysis for a ticker around event type
  compare   - Compare a specific date range to historical events
  events    - List/search/add events
  sr        - Show support & resistance for a ticker
  export    - Export analysis as JSON for quant agent
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import click
import numpy as np
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from .analysis import (
    analyze_volume_price,
    build_pattern_profile,
    build_volume_profile,
    compare_windows,
    compute_baseline_stats,
    compute_signal_stats,
    export_for_agent,
    find_support_resistance,
    sliding_window_scan,
    _find_swing_point,
)
from .data import (
    DataProvider,
    fetch_event_window,
    fetch_ticker_info,
    get_provider,
    normalize_window,
)
from .display import (
    ascii_price_chart,
    console,
    display_agent_export,
    display_categories,
    display_comparison_chart,
    display_dashboard_signal,
    display_event_list,
    display_pattern_profile,
    display_potus_schedule,
    display_regime_chart,
    display_regime_conditional_winrates,
    display_regime_states,
    display_regime_summary,
    display_scan_forecast,
    display_similarity_results,
    display_support_resistance,
    display_ticker_info,
    display_volume_price_profile,
    display_volume_profile,
)
from .events import EventCatalog, EventCategory, MarketEvent

logger = logging.getLogger(__name__)


# ── Shared options ──────────────────────────────────────────────────────────────

def common_options(f):
    f = click.option("--provider", "-p", default="yfinance",
                     type=click.Choice(["yfinance", "ibkr"]),
                     help="Data provider")(f)
    f = click.option("--verbose", "-v", is_flag=True, help="Verbose logging")(f)
    return f


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s | %(name)s | %(message)s")


def get_data_provider(provider_name: str) -> DataProvider:
    try:
        return get_provider(provider_name)
    except ImportError as e:
        console.print(f"[red]Provider error: {e}[/red]")
        sys.exit(1)


# ── Forecast tuning constants ─────────────────────────────────────────────────
FORECAST_MIN_SCORE = 0.35  # null-calibrated "Moderate" floor (see analysis.py)
FORECAST_MIN_MATCHES = 2
FORECAST_FALLBACK_N = 3
FORECAST_CONF_DECAY = 0.05
FORECAST_CONF_FLOOR = 0.30


def _find_backtest_anchor(catalog: EventCatalog, category: EventCategory,
                          max_age_days: int = 60) -> Optional[date]:
    """Find the most recent past event in the category for forecast backtesting."""
    all_events = catalog.search(category=category)
    today = date.today()
    past_recent = [e for e in all_events
                   if e.date < today and (today - e.date).days <= max_age_days]
    if past_recent:
        return max(past_recent, key=lambda e: e.date).date
    return None


def _build_forecast_from_returns(
    forward_returns_by_day: dict[int, list[tuple[float, float]]],
    match_daily_returns: dict[int, list[float]],
    anchor_price: float,
    ticker: str,
    start_date: date | None,
    actuals: dict | None = None,
) -> None:
    """Build day-by-day forecast entries from pre-computed returns and display them."""
    match_cumulative: dict[int, list[float]] = {}
    for mi, rets in match_daily_returns.items():
        prices = []
        p = anchor_price
        for ret in rets:
            p = p * (1 + ret / 100)
            prices.append(p)
        match_cumulative[mi] = prices

    forecast = []
    projected = anchor_price
    for day in sorted(forward_returns_by_day.keys()):
        entries = forward_returns_by_day[day]
        total_weight = sum(w for _, w in entries)
        if total_weight == 0:
            break
        avg_ret = sum(ret * w for ret, w in entries) / total_weight
        projected = projected * (1 + avg_ret / 100)

        match_prices = []
        for mi in match_cumulative:
            if day - 1 < len(match_cumulative[mi]):
                match_prices.append(match_cumulative[mi][day - 1])

        entry_dict = {
            "day": day,
            "price": projected,
            "change_pct": avg_ret,
            "contributors": len(entries),
        }

        if len(match_prices) >= 2:
            sorted_prices = sorted(match_prices)
            entry_dict["low_25"] = float(np.percentile(sorted_prices, 25))
            entry_dict["high_75"] = float(np.percentile(sorted_prices, 75))
            entry_dict["low_min"] = float(sorted_prices[0])
            entry_dict["high_max"] = float(sorted_prices[-1])

            agree_count = sum(1 for ret, _ in entries if (ret >= 0) == (avg_ret >= 0))
            entry_dict["agree_pct"] = agree_count / len(entries) * 100
        else:
            entry_dict["low_25"] = projected
            entry_dict["high_75"] = projected
            entry_dict["low_min"] = projected
            entry_dict["high_max"] = projected
            entry_dict["agree_pct"] = 100.0

        entry_dict["confidence"] = max(FORECAST_CONF_FLOOR, 1.0 - FORECAST_CONF_DECAY * day)
        forecast.append(entry_dict)

    if forecast:
        display_scan_forecast(forecast, ticker, anchor_price,
                              start_date=start_date, actuals=actuals)


def _compare_events(
    dp: DataProvider,
    ticker: str,
    target_norm,
    events: list[MarketEvent],
    days_before: int,
    days_after: int,
) -> tuple[list, list]:
    """Fetch historical windows and compare each to the target. Returns (windows, similarity_results)."""
    windows = []
    similarity_results = []

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}")) as progress:
        task = progress.add_task("Comparing events...", total=len(events))
        for event in events:
            progress.update(task, description=f"Analyzing {event.name}...")
            try:
                hist_window = fetch_event_window(dp, ticker, event.date, days_before, days_after)
                hist_norm = normalize_window(hist_window)
                windows.append(hist_window)

                result = compare_windows(
                    target_norm, hist_norm,
                    event_name=event.name,
                    event_date=event.date,
                )
                result.window_data = hist_norm
                similarity_results.append(result)
            except Exception as e:
                logger.warning(f"Skipping {event.name}: {e}")
            progress.advance(task)

    return windows, similarity_results


def _compute_baseline(dp: DataProvider, ticker: str, t_date: date, horizon_days: int):
    """Fetch ~3y of history and compute the unconditional forward-return base
    rate for the signal's horizon. Returns None on any failure — the signal
    then falls back to a fair-coin null."""
    try:
        base_df = dp.get_daily_ohlcv(ticker, t_date - timedelta(days=1095), t_date)
        return compute_baseline_stats(base_df, horizon_days=horizon_days)
    except Exception as e:
        logger.warning(f"Baseline computation failed for {ticker}: {e}")
        return None


def _fetch_target_window(
    dp: DataProvider,
    ticker: str,
    t_date: date,
    days_before: int,
    days_after: int,
) -> Optional[tuple]:
    """Fetch and normalize the target window. Returns (target_window, target_norm) or None on error."""
    with Progress(SpinnerColumn(), TextColumn("[bold blue]Fetching target window...")) as progress:
        progress.add_task("fetch", total=None)
        try:
            target_window = fetch_event_window(dp, ticker, t_date, days_before, days_after)
            target_norm = normalize_window(target_window)
        except Exception as e:
            console.print(f"[red]Error fetching target data: {e}[/red]")
            return None

    console.print(f"\n  Target window: {target_window.index[0].date()} → {target_window.index[-1].date()}")
    console.print(f"  {len(target_window)} trading days, current close: ${target_window['Close'].iloc[-1]:.2f}\n")
    return target_window, target_norm


def _compute_sr_levels(
    dp: DataProvider,
    ticker: str,
    t_date: date,
    target_window,
) -> list:
    """Fetch 180-day context, compute S/R levels, display chart. Returns levels (empty on error)."""
    try:
        broad_start = t_date - timedelta(days=180)
        broad_df = dp.get_daily_ohlcv(ticker, broad_start, t_date)
        sr_levels = find_support_resistance(broad_df)
        current_price = target_window["Close"].iloc[-1]
        display_support_resistance(sr_levels, current_price=current_price)

        chart = ascii_price_chart(
            target_window,
            title=f"{ticker} Price (Target Window)",
            support_resistance=sr_levels,
        )
        console.print(f"\n{chart}\n")
        return sr_levels
    except Exception as e:
        logger.warning(f"S/R analysis error: {e}")
        return []


def _compute_volume_price(
    target_window,
    ticker: str,
    show_daily: bool = True,
):
    """Compute and display volume-price profile. Returns the profile or None."""
    vp_profile = analyze_volume_price(target_window)
    if vp_profile:
        vp_profile.ticker = ticker
        display_volume_price_profile(vp_profile, show_daily=show_daily)
    return vp_profile


def _run_event_forecast(
    similarity_results: list,
    target_window,
    ticker: str,
    catalog: EventCatalog,
    category: EventCategory,
    dp: DataProvider,
) -> None:
    """Compute current vol, find backtest anchor, and run _build_event_forecast."""
    current_price = float(target_window["Close"].iloc[-1])
    _tw_rets = target_window["Close"].pct_change().dropna().values
    _current_vol = float(np.std(_tw_rets)) if len(_tw_rets) > 0 else None
    anchor_date = _find_backtest_anchor(catalog, category)
    _build_event_forecast(similarity_results, current_price, ticker,
                          start_date=target_window.index[-1].date(),
                          event_date=anchor_date, dp=dp, current_vol=_current_vol)


def _filter_events_by_regime(
    dp: DataProvider,
    ticker: str,
    events: list[MarketEvent],
    regime_lookback: int,
) -> tuple[list[MarketEvent], Optional[object]]:
    """Run HMM regime detection, filter events to current regime. Returns (events, regime_result)."""
    from .regime import run_regime_detection, filter_events_by_regime

    regime_result = None
    console.print("  [bold]Detecting market regime...[/bold]")
    try:
        regime_result = run_regime_detection(dp, ticker, lookback_days=regime_lookback)
        current_regime = regime_result.current_regime
        console.print(f"  Current regime: [bold]{current_regime}[/bold]")

        filtered = filter_events_by_regime(events, regime_result, current_regime)
        if filtered:
            console.print(f"  Filtered {len(events)} → {len(filtered)} events matching {current_regime}\n")
            events = filtered
        else:
            console.print(f"  [yellow]No events in {current_regime} regime — using all events[/yellow]\n")
    except Exception as e:
        console.print(f"  [yellow]Regime detection failed ({e}), proceeding without filter[/yellow]\n")

    return events, regime_result


def _exclude_self_comparisons(
    events: list[MarketEvent],
    target_window,
    catalog: EventCatalog,
    category: EventCategory,
    search_ticker: str,
    regime_filter: bool,
) -> Optional[list[MarketEvent]]:
    """Filter out events inside target window date range. Returns filtered list or None if empty and no fallback."""
    tw_start = target_window.index[0].date()
    tw_end = target_window.index[-1].date()
    compare_events = [e for e in events if not (tw_start <= e.date <= tw_end)]
    skipped = len(events) - len(compare_events)
    if skipped:
        console.print(f"  Skipped {skipped} current event(s) (inside target window {tw_start}–{tw_end})")

    if not compare_events:
        if regime_filter:
            all_events = catalog.search(category=category, ticker=search_ticker)
            compare_events = [e for e in all_events if not (tw_start <= e.date <= tw_end)]
            console.print(f"  [yellow]No other events in current regime — comparing against all {len(compare_events)} historical events[/yellow]\n")
        else:
            console.print("[red]No historical events to compare against.[/red]")
            return None

    return compare_events


def _compute_regime_winrates(
    regime_result: object,
    catalog: EventCatalog,
    category: EventCategory,
    search_ticker: str,
    similarity_results: list,
    ticker: str,
    event_type: str,
) -> None:
    """Compute and display per-regime win rates from similarity results."""
    from .regime import get_regime_at_date

    console.print()
    all_category_events = catalog.search(category=category, ticker=search_ticker)
    regime_winrates: dict[str, dict] = {}
    for state in regime_result.states:
        label = state.label
        returns_in_regime = []
        for event in all_category_events:
            r_label = get_regime_at_date(regime_result, event.date)
            if r_label != label:
                continue
            for sr in similarity_results:
                if sr.event_date == event.date and sr.window_data is not None:
                    wd = sr.window_data
                    if "rel_day" in wd.columns:
                        post = wd[wd["rel_day"] > 0]["Close"]
                        ev = wd[wd["rel_day"] == 0]["Close"]
                        if not ev.empty and len(post) >= 1:
                            ret = (post.iloc[-1] / ev.values[0] - 1) * 100
                            returns_in_regime.append(ret)
                    break

        if returns_in_regime:
            win_count = sum(1 for r in returns_in_regime if r > 0)
            regime_winrates[label] = {
                "win_rate": win_count / len(returns_in_regime) * 100,
                "avg_return": float(np.mean(returns_in_regime)),
                "count": len(returns_in_regime),
            }
    if regime_winrates:
        display_regime_conditional_winrates(ticker, event_type, regime_winrates)


# ── Dashboard helpers ──────────────────────────────────────────────────────────


# Heuristic ticker→category mappings for broad-market / sector ETFs
_TICKER_CATEGORY_HEURISTICS: dict[str, EventCategory] = {
    "SPY": EventCategory.FOMC, "QQQ": EventCategory.FOMC, "IWM": EventCategory.FOMC,
    "DIA": EventCategory.FOMC, "VOO": EventCategory.FOMC, "VTI": EventCategory.FOMC,
    "XLE": EventCategory.OPEC, "USO": EventCategory.OPEC, "OIL": EventCategory.OPEC,
    "BTC-USD": EventCategory.CRYPTO, "ETH-USD": EventCategory.CRYPTO,
    "COIN": EventCategory.CRYPTO, "MSTR": EventCategory.CRYPTO,
}


def _auto_detect_event_category(
    ticker: str,
    catalog: EventCatalog,
    reference_date: Optional[date] = None,
) -> Optional[EventCategory]:
    """Pick the best event category for *ticker* using proximity awareness.

    Priority:
    1. Ticker-specific catalog events (e.g. NVDA → EARNINGS) — a ticker's own
       events dominate its price behavior, so they outrank macro proximity
    2. Nearest upcoming macro event via macro calendar (FRED / FOMC / earnings)
    3. Hard-coded heuristics (SPY → FOMC, XLE → OPEC, BTC-USD → CRYPTO)
    4. Most-common category in catalog that matches the ticker
    5. None (skip event analysis)
    """
    # 1. Ticker-specific events
    ticker_events = [e for e in catalog.events if e.ticker_specific == ticker]
    if ticker_events:
        from collections import Counter
        most_common = Counter(e.category for e in ticker_events).most_common(1)
        return most_common[0][0]

    # 2. Proximity-based detection via macro calendar
    try:
        from .macro_calendar import find_nearest_macro_event
        nearest = find_nearest_macro_event(ticker, reference_date)
        if nearest:
            logger.info(
                "Auto-detect: nearest macro event for %s is %s on %s (%+d days, source=%s)",
                ticker, nearest.category.value, nearest.event_date,
                nearest.distance_days, nearest.source,
            )
            return nearest.category
    except Exception as e:
        logger.debug("Macro calendar lookup failed: %s", e)

    # 3. Heuristic map
    if ticker in _TICKER_CATEGORY_HEURISTICS:
        return _TICKER_CATEGORY_HEURISTICS[ticker]

    # 4. Most-common broad-market category
    broad = catalog.search(ticker=ticker)
    if broad:
        from collections import Counter
        most_common = Counter(e.category for e in broad).most_common(1)
        return most_common[0][0]

    return None


def _build_anchor_dates(
    df,
    catalog: EventCatalog,
    ticker: str,
    event_type: Optional[EventCategory],
    end: date,
) -> list[tuple[str, date]]:
    """Build AVWAP anchor dates: YTD open, last event, last swing point."""
    anchor_dates: list[tuple[str, date]] = []
    data_start = df.index[0].date() if hasattr(df.index[0], "date") else df.index[0]

    # 1. YTD open
    ytd_start = date(end.year, 1, 1)
    if data_start <= ytd_start:
        anchor_dates.append(("YTD Open", ytd_start))
    else:
        anchor_dates.append(("Period Start", data_start))

    # 2. Last event
    if event_type:
        events = catalog.search(category=event_type, ticker=ticker, end=end)
        past = [e for e in events if data_start <= e.date <= end]
        if past:
            last_evt = past[-1]
            anchor_dates.append((f"Last {event_type.value.upper()}: {last_evt.name}", last_evt.date))
    else:
        for cat in [EventCategory.EARNINGS, EventCategory.FOMC]:
            events = catalog.search(category=cat, ticker=ticker, end=end)
            past = [e for e in events if data_start <= e.date <= end]
            if past:
                last_evt = past[-1]
                anchor_dates.append((f"Last {cat.value.upper()}: {last_evt.name}", last_evt.date))
                break

    # 3. Last swing point
    swing = _find_swing_point(df, window=10)
    if swing:
        swing_date, swing_type = swing
        anchor_dates.append((f"Last {swing_type.title()}", swing_date))

    return anchor_dates


def _compute_scan_forward_avg(
    df,
    results: list,
    window_size: int,
) -> dict:
    """Return avg forward returns {5: pct, 10: pct, 20: pct} and per-horizon direction counts."""
    import pandas as pd

    fwd_avgs: dict[int, float] = {}
    fwd_dirs: dict[int, list[float]] = {5: [], 10: [], 20: []}

    for r in results:
        if r.event_date is None:
            continue
        match_start_idx = df.index.searchsorted(pd.Timestamp(r.event_date))
        match_end_idx = match_start_idx + window_size

        for horizon in [5, 10, 20]:
            fwd_idx = match_end_idx + horizon - 1
            if match_end_idx < len(df) and fwd_idx < len(df):
                end_price = df["Close"].iloc[match_end_idx - 1]
                fwd_price = df["Close"].iloc[fwd_idx]
                ret = (fwd_price / end_price - 1) * 100
                fwd_dirs[horizon].append(ret)

    for h in [5, 10, 20]:
        if fwd_dirs[h]:
            fwd_avgs[h] = float(np.mean(fwd_dirs[h]))

    return fwd_avgs


def _build_dashboard_signal(
    event_profile,
    scan_results: list,
    scan_fwd: dict,
    vp,
    regime_result=None,
    event_stats=None,
    baseline=None,
) -> dict:
    """Synthesize overall signal from event, scan, and vprofile analyses.

    Returns dict with keys: event, scan, vprofile, overall_direction, overall_confidence.
    """
    signal: dict = {"event": None, "scan": None, "vprofile": None}
    votes: list[tuple[str, float, float]] = []  # (direction, confidence, component_weight)

    # Event signal — statistically grounded (Wilson-shrunk confidence)
    if event_profile is not None:
        stats = event_stats or compute_signal_stats(event_profile, baseline)
        signal["event"] = {
            "direction": stats.direction,
            "confidence": round(stats.confidence, 4),
            "edge": round(stats.edge_pct, 3),
            "n_events": stats.n,
            "p_value": round(stats.p_value, 4),
        }
        if stats.direction != "neutral":
            votes.append((stats.direction, stats.confidence, 0.40))

    # Scan signal
    if scan_fwd:
        # Use +10d as primary horizon
        horizon_ret = scan_fwd.get(10, scan_fwd.get(5, 0))
        direction = "bullish" if horizon_ret > 0 else "bearish"
        # Confidence: how many matches agree
        if scan_results:
            avg_score = float(np.mean([r.composite_score for r in scan_results]))
        else:
            avg_score = 0.5
        confidence = min(1.0, avg_score)
        signal["scan"] = {
            "direction": direction,
            "confidence": confidence,
            "edge": round(horizon_ret, 3),
        }
        votes.append((direction, confidence, 0.35))

    # Volume profile signal
    if vp is not None:
        # Determine direction from position + AVWAP
        if vp.position == "above_vah":
            direction = "bullish"
        elif vp.position == "below_val":
            direction = "bearish"
        else:
            # Inside value area the POC acts as a mean-reversion magnet
            # (consistent with the vprofile signal text): above POC implies
            # pull lower, below POC implies pull higher.
            direction = "bearish" if vp.poc_distance_pct > 0 else "bullish"

        # AVWAP consensus
        if vp.anchored_vwaps:
            above = sum(1 for v in vp.anchored_vwaps if v.distance_pct > 0)
            avwap_ratio = above / len(vp.anchored_vwaps)
        else:
            avwap_ratio = 0.5
        confidence = avwap_ratio if direction == "bullish" else (1 - avwap_ratio)
        confidence = max(0.3, min(1.0, confidence))

        signal["vprofile"] = {
            "direction": direction,
            "confidence": confidence,
            "edge": round(vp.poc_distance_pct, 3),
        }
        votes.append((direction, confidence, 0.25))

    # Weighted vote. Normalize by the total weight of components present, not
    # by the vote mass: a lone component then reports its own confidence
    # instead of an unconditional 1.0, and disagreement drags the score down.
    if votes:
        bull_weight = sum(c * w for d, c, w in votes if d == "bullish")
        bear_weight = sum(c * w for d, c, w in votes if d == "bearish")
        total_possible = sum(w for _, _, w in votes)
        if total_possible > 0 and (bull_weight or bear_weight):
            signal["overall_direction"] = "bullish" if bull_weight >= bear_weight else "bearish"
            signal["overall_confidence"] = max(bull_weight, bear_weight) / total_possible
        else:
            signal["overall_direction"] = "neutral"
            signal["overall_confidence"] = 0
    else:
        signal["overall_direction"] = "neutral"
        signal["overall_confidence"] = 0

    return signal


def _build_dashboard_export(
    ticker: str,
    signal: dict,
    event_profile,
    scan_results: list,
    scan_fwd: dict,
    vp,
    sr_levels: list,
    vp_profile,
    regime_result=None,
) -> dict:
    """Combine all dashboard results into a single JSON-exportable dict."""
    export: dict = {
        "ticker": ticker,
        "generated": date.today().isoformat(),
        "signal": signal,
    }

    if event_profile is not None:
        export["event_analysis"] = event_profile.to_dict()

    if scan_results:
        export["scan_matches"] = [
            {
                "period": r.event_name,
                "start_date": r.event_date.isoformat() if r.event_date else None,
                "composite_score": round(r.composite_score, 4),
                "correlation": round(r.correlation, 4),
                "direction_match": round(r.direction_match, 4),
                "label": r.score_label,
            }
            for r in scan_results
        ]

    if scan_fwd:
        export["scan_forward_returns"] = {f"+{k}d": round(v, 3) for k, v in scan_fwd.items()}

    if vp is not None:
        export["volume_profile"] = vp.to_dict()

    if sr_levels:
        export["support_resistance"] = [
            {"price": lvl.price, "type": lvl.kind, "touches": lvl.touches, "strength": lvl.strength}
            for lvl in sr_levels
        ]

    if vp_profile is not None:
        export["volume_price_authenticity"] = vp_profile.to_dict()

    if regime_result is not None:
        export["regime"] = regime_result.to_dict()

    return export


# ── CLI Group ───────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="qpat")
def cli():
    """
    ⚡ Quant Pattern CLI — Historical price pattern analysis around key market events.

    Analyze how tickers behave around CPI, FOMC, earnings, elections, and more.
    Find similar historical patterns and extract trading signals.
    """
    pass


# ── ANALYZE command ─────────────────────────────────────────────────────────────

@cli.command()
@click.argument("ticker")
@click.option("--event-type", "-e", type=click.Choice([c.value for c in EventCategory]),
              required=True, help="Event category to analyze")
@click.option("--days-before", "-b", default=10, help="Trading days before event")
@click.option("--days-after", "-a", default=10, help="Trading days after event")
@click.option("--target-date", "-t", default=None, help="Target date to compare (YYYY-MM-DD). Default: latest")
@click.option("--top-n", "-n", default=5, help="Number of top matches to show")
@click.option("--export-json", "-o", default=None, help="Export results to JSON file")
@click.option("--event-ticker", "-et", default=None, help="Use events for this ticker (e.g. analyze SPY around NVDA earnings)")
@click.option("--regime-filter", "-rf", is_flag=True, default=False, help="Filter events by current market regime")
@click.option("--regime-lookback", default=750, help="Lookback days for regime detection")
@common_options
def analyze(ticker, event_type, days_before, days_after, target_date, top_n,
            export_json, event_ticker, provider, verbose, regime_filter, regime_lookback):
    """
    Run full pattern analysis for TICKER around events of a given type.

    Examples:

      qpat analyze SPY -e cpi -b 10 -a 10

      qpat analyze NVDA -e earnings -b 5 -a 15

      qpat analyze SPY -e earnings --event-ticker NVDA -b 5 -a 15

      qpat analyze QQQ -e fomc -t 2025-01-29 -o analysis.json
    """
    setup_logging(verbose)
    ticker = ticker.upper()
    category = EventCategory(event_type)
    dp = get_data_provider(provider)
    catalog = EventCatalog()
    search_ticker = event_ticker.upper() if event_ticker else ticker

    evt_label = f"{search_ticker} {event_type.upper()}" if event_ticker else event_type.upper()
    console.print(f"\n[bold cyan]⚡ Analyzing {ticker} around {evt_label} events[/bold cyan]")
    console.print(f"   Provider: {dp.name()} | Window: -{days_before}/+{days_after} days\n")

    display_ticker_info(fetch_ticker_info(ticker))
    console.print()

    # Get relevant events. Future-dated events (macro-calendar sync) have no
    # price history yet and cannot be compared against.
    events = catalog.search(category=category, ticker=search_ticker, end=date.today())
    if not events:
        console.print(f"[red]No {event_type} events found for {ticker}[/red]")
        return

    # Regime filtering
    regime_result = None
    if regime_filter:
        events, regime_result = _filter_events_by_regime(dp, ticker, events, regime_lookback)

    console.print(f"  Found {len(events)} relevant events\n")
    display_event_list(events)

    # Fetch target window (most recent behavior or specified date)
    t_date = datetime.strptime(target_date, "%Y-%m-%d").date() if target_date else date.today()

    result = _fetch_target_window(dp, ticker, t_date, days_before, days_after)
    if result is None:
        return
    target_window, target_norm = result

    # Exclude self-comparisons
    compare_events = _exclude_self_comparisons(events, target_window, catalog, category,
                                                search_ticker, regime_filter)
    if compare_events is None:
        return

    windows, similarity_results = _compare_events(dp, ticker, target_norm,
                                                    compare_events, days_before, days_after)

    if not similarity_results:
        console.print("[red]No valid comparisons could be made.[/red]")
        return

    # Display similarity rankings
    console.print()
    display_similarity_results(similarity_results, top_n=top_n)

    # Display overlay chart
    top_matches = sorted(similarity_results, key=lambda s: s.composite_score, reverse=True)
    display_comparison_chart(target_norm, top_matches, max_overlays=3)

    # S/R levels on target window with broader context
    console.print()
    sr_levels = _compute_sr_levels(dp, ticker, t_date, target_window)

    # Volume-Price Authenticity
    console.print()
    vp_profile = _compute_volume_price(target_window, ticker, show_daily=True)

    # Build and display profile with statistically grounded signal
    profile = build_pattern_profile(ticker, event_type, windows, similarity_results)
    baseline = _compute_baseline(dp, ticker, t_date, days_after)
    signal_stats = compute_signal_stats(profile, baseline)
    display_pattern_profile(profile, signal_stats=signal_stats)

    # Regime-conditional win rates
    if regime_filter and regime_result is not None:
        _compute_regime_winrates(regime_result, catalog, category, search_ticker,
                                  similarity_results, ticker, event_type)

    # Day-by-day forecast
    console.print()
    _run_event_forecast(similarity_results, target_window, ticker, catalog, category, dp)

    # Export
    if export_json:
        regime_data = regime_result.to_dict() if regime_result else None
        export_data = export_for_agent(profile, sr_levels, target_window,
                                       volume_price=vp_profile, regime=regime_data,
                                       signal_stats=signal_stats)
        path = Path(export_json)
        path.write_text(json.dumps(export_data, indent=2, default=str))
        console.print(f"\n[green]✓ Exported to {path}[/green]")
    else:
        console.print(
            "\n[dim]Tip: Use --export-json analysis.json to export for your quant agent[/dim]"
        )


# ── COMPARE command ─────────────────────────────────────────────────────────────

@cli.command()
@click.argument("ticker")
@click.option("--current-start", "-cs", required=True, help="Start of current period (YYYY-MM-DD)")
@click.option("--current-end", "-ce", required=True, help="End of current period (YYYY-MM-DD)")
@click.option("--hist-start", "-hs", required=True, help="Start of historical period (YYYY-MM-DD)")
@click.option("--hist-end", "-he", required=True, help="End of historical period (YYYY-MM-DD)")
@common_options
def compare(ticker, current_start, current_end, hist_start, hist_end, provider, verbose):
    """
    Compare two specific date ranges for a TICKER.

    Example:

      qpat compare SPY --cs 2026-02-20 --ce 2026-02-28 --hs 2020-02-20 --he 2020-02-28
    """
    setup_logging(verbose)
    ticker = ticker.upper()
    dp = get_data_provider(provider)

    cs = datetime.strptime(current_start, "%Y-%m-%d").date()
    ce = datetime.strptime(current_end, "%Y-%m-%d").date()
    hs = datetime.strptime(hist_start, "%Y-%m-%d").date()
    he = datetime.strptime(hist_end, "%Y-%m-%d").date()

    console.print(f"\n[bold cyan]Comparing {ticker}[/bold cyan]")
    console.print(f"  Current: {cs} → {ce}")
    console.print(f"  Historical: {hs} → {he}\n")

    display_ticker_info(fetch_ticker_info(ticker))
    console.print()

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Fetching data...")) as progress:
        task = progress.add_task("fetch", total=None)
        try:
            df_current = dp.get_daily_ohlcv(ticker, cs, ce)
            df_hist = dp.get_daily_ohlcv(ticker, hs, he)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    # Normalize both from their first day
    df_current["rel_day"] = range(len(df_current))
    df_hist["rel_day"] = range(len(df_hist))

    curr_norm = normalize_window(df_current)
    hist_norm = normalize_window(df_hist)

    result = compare_windows(curr_norm, hist_norm, event_name=f"Historical {hs}", event_date=hs)
    result.window_data = hist_norm

    display_similarity_results([result], top_n=1)
    display_comparison_chart(curr_norm, [result], target_label=f"{cs}→{ce}", max_overlays=1)

    # Side-by-side charts
    console.print(ascii_price_chart(df_current, title=f"Current: {cs} → {ce}"))
    console.print()
    console.print(ascii_price_chart(df_hist, title=f"Historical: {hs} → {he}"))

    # Volume-Price Authenticity for both periods
    console.print()
    try:
        # Fetch extra 40 calendar days before each period for RVOL baseline
        df_curr_broad = dp.get_daily_ohlcv(ticker, cs - timedelta(days=40), ce)
        vp_current = analyze_volume_price(df_curr_broad, report_last_n=len(df_current))
        if vp_current:
            vp_current.ticker = ticker
            console.print(f"  [bold]Current Period ({cs} → {ce}):[/bold]")
            display_volume_price_profile(vp_current, show_daily=True)

        df_hist_broad = dp.get_daily_ohlcv(ticker, hs - timedelta(days=40), he)
        vp_hist = analyze_volume_price(df_hist_broad, report_last_n=len(df_hist))
        if vp_hist:
            vp_hist.ticker = ticker
            console.print(f"\n  [bold]Historical Period ({hs} → {he}):[/bold]")
            display_volume_price_profile(vp_hist, show_daily=True)
    except Exception as e:
        logger.warning(f"Volume-price analysis error: {e}")


# ── SR command ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("ticker")
@click.option("--lookback", "-l", default=180, help="Lookback period in days")
@click.option("--levels", "-n", default=5, help="Number of S/R levels per type")
@click.option("--window", "-w", default=5, help="Local extrema detection window")
@common_options
def sr(ticker, lookback, levels, window, provider, verbose):
    """
    Show support & resistance levels for TICKER.

    Example:

      qpat sr NVDA --lookback 365 --levels 8
    """
    setup_logging(verbose)
    ticker = ticker.upper()
    dp = get_data_provider(provider)

    end = date.today()
    start = end - timedelta(days=lookback)

    console.print(f"\n[bold cyan]S/R Analysis: {ticker}[/bold cyan]")
    console.print(f"  Period: {start} → {end} ({lookback} days)\n")

    display_ticker_info(fetch_ticker_info(ticker))
    console.print()

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Fetching data...")) as progress:
        task = progress.add_task("fetch", total=None)
        try:
            df = dp.get_daily_ohlcv(ticker, start, end)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    current_price = df["Close"].iloc[-1]
    sr_levels = find_support_resistance(df, window=window, num_levels=levels)

    display_support_resistance(sr_levels, current_price=current_price)
    console.print(f"\n  Current price: [bold]${current_price:.2f}[/bold]\n")

    chart = ascii_price_chart(
        df.tail(60),  # Last 60 trading days
        title=f"{ticker} ({start} → {end})",
        support_resistance=sr_levels,
    )
    console.print(chart)


# ── VPROFILE command ──────────────────────────────────────────────────────────

@cli.command()
@click.argument("ticker")
@click.option("--lookback", "-l", default=60, help="Lookback period in trading days")
@click.option("--bins", "-b", default=100, help="Number of price bins for volume profile")
@click.option("--event", "-e", default=None, type=click.Choice(
    [c.value for c in EventCategory], case_sensitive=False),
    help="Event category for anchored VWAP (e.g. earnings, fomc)")
@common_options
def vprofile(ticker, lookback, bins, event, provider, verbose):
    """
    Volume Profile & Anchored VWAP analysis for TICKER.

    Shows where volume actually traded across price levels, identifies the
    Point of Control (POC), Value Area (70% zone), and computes anchored
    VWAPs from key dates (YTD open, last earnings/event, last swing point).

    Examples:

      qpat vprofile SPY

      qpat vprofile NVDA --lookback 120 --bins 150

      qpat vprofile AAPL -e earnings
    """
    setup_logging(verbose)
    ticker = ticker.upper()
    dp = get_data_provider(provider)

    end = date.today()
    # Use calendar days with buffer for lookback trading days
    start = end - timedelta(days=int(lookback * 1.5) + 15)

    console.print(f"\n[bold cyan]Volume Profile: {ticker}[/bold cyan]")
    console.print(f"  Lookback: ~{lookback} trading days  |  Bins: {bins}\n")

    display_ticker_info(fetch_ticker_info(ticker))
    console.print()

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Fetching data...")) as progress:
        task = progress.add_task("fetch", total=None)
        try:
            df = dp.get_daily_ohlcv(ticker, start, end)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    # Trim to requested lookback trading days
    if len(df) > lookback:
        df = df.iloc[-lookback:]

    # Build anchor dates for AVWAP
    anchor_dates: list[tuple[str, date]] = []

    # 1. YTD open
    ytd_start = date(end.year, 1, 1)
    if df.index[0].date() <= ytd_start if hasattr(df.index[0], "date") else df.index[0] <= ytd_start:
        anchor_dates.append(("YTD Open", ytd_start))
    else:
        # Use start of data as fallback
        first_dt = df.index[0].date() if hasattr(df.index[0], "date") else df.index[0]
        anchor_dates.append(("Period Start", first_dt))

    # 2. Last event from catalog (earnings, FOMC, etc.)
    catalog = EventCatalog()
    if event:
        cat = EventCategory(event)
        events = catalog.search(category=cat, ticker=ticker, end=end)
        # Find the most recent one within our data range
        data_start = df.index[0].date() if hasattr(df.index[0], "date") else df.index[0]
        past_events = [e for e in events if data_start <= e.date <= end]
        if past_events:
            last_evt = past_events[-1]
            anchor_dates.append((f"Last {cat.value.upper()}: {last_evt.name}", last_evt.date))
    else:
        # Auto-detect: try earnings first, then FOMC
        for cat in [EventCategory.EARNINGS, EventCategory.FOMC]:
            events = catalog.search(category=cat, ticker=ticker, end=end)
            data_start = df.index[0].date() if hasattr(df.index[0], "date") else df.index[0]
            past_events = [e for e in events if data_start <= e.date <= end]
            if past_events:
                last_evt = past_events[-1]
                anchor_dates.append((f"Last {cat.value.upper()}: {last_evt.name}", last_evt.date))
                break

    # 3. Last major swing point
    swing = _find_swing_point(df, window=10)
    if swing:
        swing_date, swing_type = swing
        anchor_dates.append((f"Last {swing_type.title()}", swing_date))

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Computing volume profile...")) as progress:
        task = progress.add_task("compute", total=None)
        vp = build_volume_profile(df, ticker, num_bins=bins, anchor_dates=anchor_dates)

    if vp is None:
        console.print("[red]Could not compute volume profile — insufficient data[/red]")
        return

    display_volume_profile(vp)

    # Also show S/R levels for comparison
    sr_levels = find_support_resistance(df, window=5, num_levels=5)
    if sr_levels:
        console.print()
        display_support_resistance(sr_levels, current_price=vp.current_price)

    # Price chart with S/R overlay
    console.print()
    chart = ascii_price_chart(
        df.tail(60),
        title=f"{ticker} Price ({df.index[-60].strftime('%Y-%m-%d') if len(df) >= 60 else df.index[0].strftime('%Y-%m-%d')} → {end})",
        support_resistance=sr_levels,
    )
    console.print(chart)
    console.print()


# ── SCAN command ───────────────────────────────────────────────────────────────

@cli.command()
@click.argument("ticker")
@click.option("--days", "-d", default=10, help="Window size in trading days")
@click.option("--lookback", "-l", default=1000, help="How many calendar days of history to scan")
@click.option("--step", "-s", default=1, help="Slide step in trading days (1=daily, 5=weekly)")
@click.option("--top-n", "-n", default=5, help="Number of top matches to show")
@click.option("--export-json", "-o", default=None, help="Export results to JSON file")
@common_options
def scan(ticker, days, lookback, step, top_n, export_json, provider, verbose):
    """
    Scan history for periods most similar to recent price action. No events needed.

    Slides a window across all historical data and ranks by similarity to the
    most recent N trading days.

    Examples:

      qpat scan SPY --days 10 --lookback 1000

      qpat scan NVDA -d 20 -l 2000 -s 5 -n 10

      qpat scan QQQ -d 15 -l 1500 -o scan_results.json
    """
    setup_logging(verbose)
    ticker = ticker.upper()
    dp = get_data_provider(provider)

    end = date.today()
    start = end - timedelta(days=lookback)

    console.print(f"\n[bold cyan]⚡ Scanning {ticker} for similar price patterns[/bold cyan]")
    console.print(f"   Window: {days} trading days | Lookback: {lookback} calendar days | Step: {step}")
    console.print(f"   Provider: {dp.name()}\n")

    display_ticker_info(fetch_ticker_info(ticker))
    console.print()

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Fetching historical data...")) as progress:
        task = progress.add_task("fetch", total=None)
        try:
            df = dp.get_daily_ohlcv(ticker, start, end)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    console.print(f"  Loaded {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})")

    target_start = df.index[-days] if len(df) >= days else df.index[0]
    console.print(f"  Target window: {target_start.date()} → {df.index[-1].date()}")
    console.print(f"  Current close: ${df['Close'].iloc[-1]:.2f}\n")

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Scanning {task.description}")) as progress:
        task = progress.add_task(
            f"{len(df) - days} windows...",
            total=None,
        )
        results = sliding_window_scan(df, window_size=days, step=step, top_n=top_n)

    if not results:
        console.print("[red]No similar patterns found.[/red]")
        return

    # Display results
    display_similarity_results(results, top_n=top_n)

    # Overlay chart: target vs top matches
    target_df = df.iloc[-days:].copy()
    target_df["rel_day"] = range(days)
    target_ref = target_df["Close"].iloc[0]
    target_df["Close_norm"] = ((target_df["Close"] / target_ref) - 1) * 100
    display_comparison_chart(target_df, results, target_label="Current", max_overlays=3)

    # S/R on recent data
    console.print()
    try:
        sr_levels = find_support_resistance(df.tail(180))
        current_price = df["Close"].iloc[-1]
        display_support_resistance(sr_levels, current_price=current_price)

        chart = ascii_price_chart(
            df.tail(60),
            title=f"{ticker} (Last 60 Trading Days)",
            support_resistance=sr_levels,
        )
        console.print(f"\n{chart}\n")
    except Exception as e:
        logger.warning(f"S/R error: {e}")

    # What happened after each historical match
    console.print()
    _display_scan_forward_returns(df, results, days)

    # Volume-Price Authenticity (summary only for scan)
    console.print()
    vp_profile = analyze_volume_price(df, report_last_n=days)
    if vp_profile:
        vp_profile.ticker = ticker
        display_volume_price_profile(vp_profile, show_daily=False)

    # Day-by-day forecast based on top matches
    console.print()
    _display_forecast(df, results, days, ticker)

    # Export
    if export_json:
        export_data = {
            "ticker": ticker,
            "scan_window_days": days,
            "lookback_days": lookback,
            "target": {
                "start": target_df.index[0].isoformat(),
                "end": target_df.index[-1].isoformat(),
                "close_start": round(float(target_df["Close"].iloc[0]), 2),
                "close_end": round(float(target_df["Close"].iloc[-1]), 2),
            },
            "matches": [
                {
                    "period": r.event_name,
                    "start_date": r.event_date.isoformat() if r.event_date else None,
                    "composite_score": round(r.composite_score, 4),
                    "correlation": round(r.correlation, 4),
                    "direction_match": round(r.direction_match, 4),
                    "dtw_distance": round(r.dtw_distance, 4),
                    "label": r.score_label,
                }
                for r in results
            ],
        }
        path = Path(export_json)
        path.write_text(json.dumps(export_data, indent=2, default=str))
        console.print(f"[green]✓ Exported to {path}[/green]")


def _display_scan_forward_returns(df, results, window_size: int):
    """Show what happened after each matched historical period."""
    import pandas as pd
    from rich.table import Table
    from rich import box

    table = Table(
        title="What Happened After Each Match",
        box=box.ROUNDED,
        header_style="bold green",
    )
    table.add_column("Period", width=28)
    table.add_column("Score", justify="center", width=8)
    table.add_column("+5d", justify="right", width=8)
    table.add_column("+10d", justify="right", width=8)
    table.add_column("+20d", justify="right", width=8)

    for r in results:
        if r.event_date is None:
            continue
        # Find the end of the matched window in the dataframe
        match_start_idx = df.index.searchsorted(pd.Timestamp(r.event_date))
        match_end_idx = match_start_idx + window_size

        fwd = {}
        for horizon in [5, 10, 20]:
            fwd_idx = match_end_idx + horizon - 1
            if match_end_idx < len(df) and fwd_idx < len(df):
                end_price = df["Close"].iloc[match_end_idx - 1]
                fwd_price = df["Close"].iloc[fwd_idx]
                fwd[horizon] = (fwd_price / end_price - 1) * 100
            else:
                fwd[horizon] = None

        def _fmt(val):
            if val is None:
                return "[dim]—[/dim]"
            color = "green" if val > 0 else "red"
            return f"[{color}]{val:+.2f}%[/{color}]"

        table.add_row(
            r.event_name[:28],
            f"{r.composite_score:.3f}",
            _fmt(fwd[5]),
            _fmt(fwd[10]),
            _fmt(fwd[20]),
        )

    console.print(table)


def _build_event_forecast(similarity_results, current_price, ticker, start_date=None,
                          event_date=None, dp=None, current_vol=None):
    """Build a day-by-day forecast from event-based matches' post-event returns.

    Uses quality-gated filtering, exponential score weighting, volatility scaling,
    and per-match tracking for confidence bands.

    When event_date and dp are provided, anchors forecast to event-day close
    and includes actual prices for days that have already passed.
    """
    import pandas as pd

    sorted_results = sorted(similarity_results, key=lambda s: s.composite_score, reverse=True)

    # Quality gate: only include matches above score threshold
    quality = [r for r in sorted_results if r.composite_score >= FORECAST_MIN_SCORE]
    if len(quality) < FORECAST_MIN_MATCHES:
        quality = sorted_results[:FORECAST_FALLBACK_N]
    top = quality

    forward_returns_by_day: dict[int, list[tuple[float, float]]] = {}
    match_daily_returns: dict[int, list[float]] = {}  # match_idx -> [day1_ret, ...]

    for mi, r in enumerate(top):
        wd = r.window_data
        if wd is None or "rel_day" not in wd.columns or "Close" not in wd.columns:
            continue

        post = wd[wd["rel_day"] >= 0].sort_values("rel_day")
        if len(post) < 2:
            continue

        weight = r.composite_score ** 2  # Exponential weighting

        # Historical volatility for this match
        hist_vol = None
        if len(wd) > 1:
            closes_all = wd["Close"].values
            rets = np.diff(closes_all) / closes_all[:-1]
            if len(rets) > 0:
                hist_vol = float(np.std(rets))

        vol_ratio = 1.0
        if current_vol is not None and hist_vol is not None and hist_vol > 0:
            vol_ratio = max(0.5, min(2.0, current_vol / hist_vol))

        closes = post["Close"].values
        daily_rets = []
        for i in range(1, len(closes)):
            daily_ret = (closes[i] / closes[i - 1] - 1) * 100
            scaled_ret = daily_ret * vol_ratio
            forward_returns_by_day.setdefault(i, []).append((scaled_ret, weight))
            daily_rets.append(scaled_ret)

        match_daily_returns[mi] = daily_rets

    if not forward_returns_by_day:
        return

    # When event_date provided, anchor forecast to event-day close and get actuals
    forecast_price = current_price
    forecast_start = start_date
    actuals = None

    if event_date and dp:
        try:
            actual_df = dp.get_daily_ohlcv(ticker, event_date - timedelta(days=5), date.today())
            if not actual_df.empty:
                event_ts = pd.Timestamp(event_date)
                idx = actual_df.index.searchsorted(event_ts)
                if idx < len(actual_df):
                    forecast_price = float(actual_df["Close"].iloc[idx])
                    forecast_start = actual_df.index[idx].date()

                    actuals = {}
                    for j in range(idx + 1, len(actual_df)):
                        d = actual_df.index[j].date()
                        actuals[d] = float(actual_df["Close"].iloc[j])
        except Exception as e:
            logger.warning(f"Could not fetch actuals for backtest: {e}")

    _build_forecast_from_returns(forward_returns_by_day, match_daily_returns,
                                  forecast_price, ticker, forecast_start, actuals)


def _display_forecast(df, results, window_size: int, ticker: str):
    """Build a weighted day-by-day price forecast from top matches' forward returns.

    Uses quality-gated filtering, exponential score weighting, volatility scaling,
    and per-match tracking for confidence bands.
    """
    import pandas as pd

    current_price = float(df["Close"].iloc[-1])

    # Compute current volatility from recent window
    recent = df.iloc[-window_size:]
    current_vol = None
    if len(recent) > 1:
        recent_rets = recent["Close"].pct_change().dropna().values
        if len(recent_rets) > 0:
            current_vol = float(np.std(recent_rets))

    # Quality gate
    sorted_results = sorted(results, key=lambda s: s.composite_score, reverse=True)
    quality = [r for r in sorted_results if r.composite_score >= FORECAST_MIN_SCORE]
    if len(quality) < FORECAST_MIN_MATCHES:
        quality = sorted_results[:FORECAST_FALLBACK_N]

    forward_returns_by_day: dict[int, list[tuple[float, float]]] = {}
    match_daily_returns: dict[int, list[float]] = {}

    for mi, r in enumerate(quality):
        if r.event_date is None:
            continue
        match_start_idx = df.index.searchsorted(pd.Timestamp(r.event_date))
        match_end_idx = match_start_idx + window_size

        if match_end_idx >= len(df):
            continue

        end_price = float(df["Close"].iloc[match_end_idx - 1])
        weight = r.composite_score ** 2  # Exponential weighting

        # Historical volatility from match window
        match_window = df.iloc[match_start_idx:match_end_idx]
        hist_vol = None
        if len(match_window) > 1:
            match_rets = match_window["Close"].pct_change().dropna().values
            if len(match_rets) > 0:
                hist_vol = float(np.std(match_rets))

        vol_ratio = 1.0
        if current_vol is not None and hist_vol is not None and hist_vol > 0:
            vol_ratio = max(0.5, min(2.0, current_vol / hist_vol))

        prev_price = end_price
        daily_rets = []
        for d in range(1, window_size + 1):
            fwd_idx = match_end_idx + d - 1
            if fwd_idx >= len(df):
                break
            fwd_price = float(df["Close"].iloc[fwd_idx])
            daily_ret = (fwd_price / prev_price - 1) * 100
            scaled_ret = daily_ret * vol_ratio
            forward_returns_by_day.setdefault(d, []).append((scaled_ret, weight))
            daily_rets.append(scaled_ret)
            prev_price = fwd_price

        match_daily_returns[mi] = daily_rets

    if not forward_returns_by_day:
        return

    last_date = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]
    _build_forecast_from_returns(forward_returns_by_day, match_daily_returns,
                                  current_price, ticker, last_date)


# ── EVENTS command ──────────────────────────────────────────────────────────────

@cli.group()
def events():
    """List, search, and manage market events."""
    pass


@events.command("list")
@click.option("--category", "-c", type=click.Choice([c.value for c in EventCategory]),
              default=None, help="Filter by category")
@click.option("--ticker", "-t", default=None, help="Filter by ticker")
def events_list(category, ticker):
    """List events from the catalog."""
    catalog = EventCatalog()
    cat = EventCategory(category) if category else None
    results = catalog.search(category=cat, ticker=ticker)

    if not results:
        console.print("[yellow]No events found matching criteria[/yellow]")
        return

    display_event_list(results)
    console.print(f"\n  Total: {len(results)} events")


@events.command("categories")
def events_categories():
    """Show available event categories."""
    display_categories()


@events.command("add")
@click.option("--name", "-n", required=True, help="Event name")
@click.option("--date", "-d", "event_date", required=True, help="Event date (YYYY-MM-DD)")
@click.option("--category", "-c", type=click.Choice([c.value for c in EventCategory]),
              required=True, help="Event category")
@click.option("--description", "-desc", default="", help="Description")
@click.option("--ticker", "-t", default=None, help="Ticker-specific event (None = broad market)")
def events_add(name, event_date, category, description, ticker):
    """Add a custom event to the catalog."""
    catalog = EventCatalog()
    event = MarketEvent(
        name=name,
        date=datetime.strptime(event_date, "%Y-%m-%d").date(),
        category=EventCategory(category),
        description=description,
        ticker_specific=ticker.upper() if ticker else None,
    )
    catalog.save_custom_event(event)
    console.print(f"[green]✓ Added custom event: {event.name} ({event.date})[/green]")


@events.command("sync-calendar")
@click.option("--force", "-f", is_flag=True, default=False, help="Force refresh (ignore cache TTL)")
def events_sync_calendar(force):
    """Sync macro calendar from FRED API + the Fed's FOMC schedule.

    Fetches upcoming release dates for CPI, PPI, NFP, GDP, Retail Sales
    from FRED, plus FOMC announcement dates live from federalreserve.gov
    (hardcoded fallback if the fetch fails). Results are cached for 24 hours.

    Requires a FRED API key — set via: qpat config set fred-api-key <KEY>
    """
    from .macro_calendar import sync_macro_calendar, get_fred_api_key

    api_key = get_fred_api_key()
    if not api_key:
        console.print("[yellow]⚠ No FRED API key configured.[/yellow]")
        console.print("  FOMC dates still sync (fetched from federalreserve.gov).")
        console.print("  For CPI/PPI/NFP/GDP/Retail Sales, run:")
        console.print("  [cyan]qpat config set fred-api-key <YOUR_KEY>[/cyan]\n")

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Syncing macro calendar...")) as progress:
        progress.add_task("sync", total=None)
        releases = sync_macro_calendar(force=force)

    if not releases:
        console.print("[red]No release dates fetched.[/red]")
        return

    console.print("[green]✓ Macro calendar synced[/green]\n")
    for cat_val, dates in sorted(releases.items()):
        label = cat_val.upper().replace("_", " ")
        upcoming = [d for d in dates if d >= date.today().isoformat()]
        console.print(f"  [bold]{label}[/bold]: {len(upcoming)} upcoming dates")
        for d in upcoming[:3]:
            console.print(f"    {d}")
        if len(upcoming) > 3:
            console.print(f"    ... and {len(upcoming) - 3} more")

    # A shrinking FOMC horizon means the live fetch has been failing and the
    # fallback list is running out — say so before it goes silent.
    fomc = releases.get("fomc", [])
    horizon = max(fomc) if fomc else None
    if horizon is None or horizon < (date.today() + timedelta(days=90)).isoformat():
        console.print(f"[yellow]⚠ FOMC schedule only extends to {horizon or 'nothing'} — "
                      "the federalreserve.gov fetch is likely failing; event warnings "
                      "will degrade past that date.[/yellow]")
    console.print()


# ── CONFIG command ─────────────────────────────────────────────────────────────

@cli.group()
def config():
    """Manage qpat configuration (API keys, preferences)."""
    pass


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Set a configuration value.

    Example: qpat config set fred-api-key YOUR_API_KEY
    """
    from .macro_calendar import load_config, save_config

    cfg = load_config()
    # Normalize key: fred-api-key → fred_api_key
    storage_key = key.replace("-", "_")
    cfg[storage_key] = value
    save_config(cfg)
    console.print(f"[green]✓ Set {key}[/green]")


@config.command("get")
@click.argument("key")
def config_get(key):
    """Get a configuration value (sensitive values are masked).

    Example: qpat config get fred-api-key
    """
    from .macro_calendar import load_config

    cfg = load_config()
    storage_key = key.replace("-", "_")
    val = cfg.get(storage_key)
    if val is None:
        console.print(f"[yellow]{key} is not set[/yellow]")
    elif "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
        masked = val[:4] + "..." + val[-4:] if len(val) > 8 else "****"
        console.print(f"  {key} = {masked}")
    else:
        console.print(f"  {key} = {val}")


@config.command("show")
def config_show():
    """Show all configuration values."""
    from .macro_calendar import load_config

    cfg = load_config()
    if not cfg:
        console.print("[yellow]No configuration set.[/yellow]")
        console.print("  Run: [cyan]qpat config set fred-api-key <KEY>[/cyan]")
        return

    for k, v in cfg.items():
        display_key = k.replace("_", "-")
        if "key" in k or "secret" in k or "token" in k:
            masked = v[:4] + "..." + v[-4:] if len(str(v)) > 8 else "****"
            console.print(f"  {display_key} = {masked}")
        else:
            console.print(f"  {display_key} = {v}")


# ── EXPORT command ──────────────────────────────────────────────────────────────

@cli.command()
@click.argument("ticker")
@click.option("--event-type", "-e", type=click.Choice([c.value for c in EventCategory]),
              required=True, help="Event category")
@click.option("--output", "-o", default="qpat_export.json", help="Output file path")
@click.option("--days-before", "-b", default=10, help="Trading days before event")
@click.option("--days-after", "-a", default=10, help="Trading days after event")
@click.option("--event-ticker", "-et", default=None, help="Use events for this ticker (e.g. export SPY around NVDA earnings)")
@common_options
def export(ticker, event_type, output, days_before, days_after, event_ticker, provider, verbose):
    """
    Export full analysis as JSON for quant agent consumption.

    Example:

      qpat export SPY -e fomc -o spy_fomc.json

      qpat export SPY -e earnings --event-ticker NVDA -o spy_nvda_earnings.json
    """
    setup_logging(verbose)
    ticker = ticker.upper()
    category = EventCategory(event_type)
    dp = get_data_provider(provider)
    catalog = EventCatalog()
    search_ticker = event_ticker.upper() if event_ticker else ticker

    events_list = catalog.search(category=category, ticker=search_ticker, end=date.today())
    if not events_list:
        console.print(f"[red]No events found[/red]")
        return

    console.print(f"[bold cyan]Exporting {ticker} × {event_type} analysis...[/bold cyan]\n")

    display_ticker_info(fetch_ticker_info(ticker))
    console.print()

    t_date = date.today()
    target_window = fetch_event_window(dp, ticker, t_date, days_before, days_after)
    target_norm = normalize_window(target_window)

    windows, similarity_results = _compare_events(dp, ticker, target_norm,
                                                    events_list, days_before, days_after)

    # S/R (no try/except — let errors propagate in export)
    broad_df = dp.get_daily_ohlcv(ticker, t_date - timedelta(days=180), t_date)
    sr_levels = find_support_resistance(broad_df)

    profile = build_pattern_profile(ticker, event_type, windows, similarity_results)
    baseline = _compute_baseline(dp, ticker, t_date, days_after)
    signal_stats = compute_signal_stats(profile, baseline)

    # Volume-Price Authenticity
    vp_profile = _compute_volume_price(target_window, ticker, show_daily=False)

    # Day-by-day forecast
    if similarity_results:
        console.print()
        _run_event_forecast(similarity_results, target_window, ticker, catalog, category, dp)

    export_data = export_for_agent(profile, sr_levels, target_window, volume_price=vp_profile,
                                   signal_stats=signal_stats)

    path = Path(output)
    path.write_text(json.dumps(export_data, indent=2, default=str))
    console.print(f"[green]✓ Exported to {path}[/green]")
    display_agent_export(export_data)


# ── INTERACTIVE command ─────────────────────────────────────────────────────────

@cli.command()
@common_options
def interactive(provider, verbose):
    """
    Interactive mode - guided analysis with prompts.
    """
    setup_logging(verbose)
    dp = get_data_provider(provider)
    catalog = EventCatalog()

    console.print("\n[bold cyan]⚡ Quant Pattern CLI — Interactive Mode[/bold cyan]\n")

    # Step 1: Ticker
    ticker = Prompt.ask("[bold]Enter ticker symbol", default="SPY").upper()

    # Step 2: Event category
    console.print()
    display_categories()
    console.print()
    cat_choice = Prompt.ask(
        "[bold]Select event category",
        choices=[c.value for c in EventCategory],
        default="fomc",
    )
    category = EventCategory(cat_choice)

    # Step 3: Window
    days_before = IntPrompt.ask("[bold]Days before event", default=10)
    days_after = IntPrompt.ask("[bold]Days after event", default=10)

    # Step 4: Target date
    target_str = Prompt.ask("[bold]Target date (YYYY-MM-DD or 'today')", default="today")
    t_date = date.today() if target_str == "today" else datetime.strptime(target_str, "%Y-%m-%d").date()

    # Step 5: Specific event or all?
    all_events = catalog.search(category=category, ticker=ticker, end=date.today())
    if not all_events:
        console.print(f"[red]No events found for {category.value} + {ticker}[/red]")
        return

    console.print()
    for i, e in enumerate(all_events, 1):
        console.print(f"  [cyan]{i:>3}[/cyan]) {e.name} ({e.date}) [dim]{e.description[:40]}[/dim]")
    console.print()

    event_choice = Prompt.ask(
        "[bold]Enter event # to compare against one, or 'all'",
        default="all",
    )

    if event_choice.strip().lower() == "all":
        events_list = all_events
    else:
        try:
            idx = int(event_choice)
            if 1 <= idx <= len(all_events):
                events_list = [all_events[idx - 1]]
                console.print(f"\n  Selected: [bold]{events_list[0].name}[/bold] ({events_list[0].date})")
            else:
                console.print(f"[yellow]Invalid choice, using all events[/yellow]")
                events_list = all_events
        except ValueError:
            console.print(f"[yellow]Invalid input, using all events[/yellow]")
            events_list = all_events

    # Run analysis
    console.print(f"\n[bold cyan]Running analysis: {ticker} × {category.value.upper()}...[/bold cyan]\n")

    try:
        target_window = fetch_event_window(dp, ticker, t_date, days_before, days_after)
        target_norm = normalize_window(target_window)
    except Exception as e:
        console.print(f"[red]Error fetching target: {e}[/red]")
        return

    windows, similarity_results = _compare_events(dp, ticker, target_norm,
                                                    events_list, days_before, days_after)

    if similarity_results:
        display_similarity_results(similarity_results, top_n=len(similarity_results))
        top = sorted(similarity_results, key=lambda s: s.composite_score, reverse=True)
        display_comparison_chart(target_norm, top, max_overlays=3)

    # S/R
    sr_levels = _compute_sr_levels(dp, ticker, t_date, target_window)

    # Volume-Price Authenticity
    console.print()
    vp_profile = _compute_volume_price(target_window, ticker, show_daily=True)

    if windows and similarity_results:
        profile = build_pattern_profile(ticker, cat_choice, windows, similarity_results)
        signal_stats = compute_signal_stats(profile, _compute_baseline(dp, ticker, t_date, days_after))
        display_pattern_profile(profile, signal_stats=signal_stats)

        # Day-by-day forecast
        console.print()
        _run_event_forecast(similarity_results, target_window, ticker, catalog, category, dp)

    # Export option
    if Prompt.ask("\n[bold]Export to JSON?", choices=["y", "n"], default="n") == "y":
        out = Prompt.ask("[bold]Output file", default=f"{ticker.lower()}_{cat_choice}_analysis.json")
        if windows and similarity_results:
            export_data = export_for_agent(profile, sr_levels, target_window, volume_price=vp_profile,
                                           signal_stats=signal_stats)
            Path(out).write_text(json.dumps(export_data, indent=2, default=str))
            console.print(f"[green]✓ Exported to {out}[/green]")


# ── REGIME command ─────────────────────────────────────────────────────────────

@cli.command()
@click.argument("ticker")
@click.option("--lookback", "-l", default=750, help="Lookback period in calendar days")
@click.option("--states", "-s", default=4, help="Number of HMM states (2-6)")
@click.option("--export-json", "-o", default=None, help="Export regime result to JSON file")
@common_options
def regime(ticker, lookback, states, export_json, provider, verbose):
    """
    Detect the current market regime for TICKER using HMM.

    Identifies Bull-Trend, Bear-Trend, Low-Vol-Range, and High-Vol-Stress
    regimes from price, volatility, and macro features.

    Examples:

      qpat regime SPY

      qpat regime QQQ --lookback 1500

      qpat regime NVDA -o regime.json
    """
    setup_logging(verbose)
    ticker = ticker.upper()
    dp = get_data_provider(provider)

    console.print(f"\n[bold cyan]⚡ Regime Detection: {ticker}[/bold cyan]")
    console.print(f"   Lookback: {lookback} days | States: {states} | Provider: {dp.name()}\n")

    display_ticker_info(fetch_ticker_info(ticker))
    console.print()

    # Lazy import to avoid loading hmmlearn at startup
    from .regime import run_regime_detection

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Detecting regimes...")) as progress:
        task = progress.add_task("detect", total=None)
        try:
            result = run_regime_detection(dp, ticker, lookback_days=lookback, n_states=states)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    console.print()
    display_regime_summary(result)
    console.print()
    display_regime_states(result)
    console.print()
    display_regime_chart(result)

    if export_json:
        path = Path(export_json)
        path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        console.print(f"\n[green]✓ Exported regime data to {path}[/green]")


# ── DASHBOARD command ──────────────────────────────────────────────────────────

@cli.command()
@click.argument("ticker")
@click.option("--event-type", "-e", type=click.Choice([c.value for c in EventCategory]),
              default=None, help="Event category (auto-detected if omitted)")
@click.option("--days", "-d", default=15, help="Window size in trading days")
@click.option("--lookback", "-l", default=1000, help="Calendar days of history to scan")
@click.option("--top-n", "-n", default=5, help="Number of top matches to show per section")
@click.option("--bins", "-b", default=100, help="Volume profile price bins")
@click.option("--regime", "-r", is_flag=True, default=False, help="Include regime detection")
@click.option("--target-date", "-t", default=None, help="Target date (YYYY-MM-DD). Default: today")
@click.option("--export-json", "-o", default=None, help="Export combined results to JSON")
@common_options
def dashboard(ticker, event_type, days, lookback, top_n, bins, regime, target_date,
              export_json, provider, verbose):
    """
    Unified research dashboard combining event analysis, pattern scan, and volume profile.

    Runs all three analyses with shared data fetching and presents results in
    labeled sections ending with a combined signal summary.

    Examples:

      qpat dashboard SPY

      qpat dashboard NVDA -e earnings --regime

      qpat dashboard QQQ -d 20 -o dash.json
    """
    setup_logging(verbose)
    ticker = ticker.upper()
    dp = get_data_provider(provider)
    catalog = EventCatalog()

    end = datetime.strptime(target_date, "%Y-%m-%d").date() if target_date else date.today()
    start = end - timedelta(days=lookback)

    # Auto-detect event category (proximity-aware)
    category = EventCategory(event_type) if event_type else _auto_detect_event_category(ticker, catalog, reference_date=end)
    cat_label = category.value.upper() if category else "NONE"

    console.print(f"\n[bold cyan]⚡ Dashboard: {ticker}[/bold cyan]")
    console.print(f"   Events: {cat_label} | Window: {days}d | Lookback: {lookback}d | Bins: {bins}")
    console.print(f"   Provider: {dp.name()}\n")

    display_ticker_info(fetch_ticker_info(ticker))
    console.print()

    # ── Single data fetch ──────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("[bold blue]Fetching historical data...")) as progress:
        progress.add_task("fetch", total=None)
        try:
            df = dp.get_daily_ohlcv(ticker, start, end)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    current_price = float(df["Close"].iloc[-1])
    console.print(f"  Loaded {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})")
    console.print(f"  Current close: ${current_price:.2f}\n")

    # ── Regime detection (optional) ────────────────────────────────────────
    regime_result = None
    if regime:
        from .regime import run_regime_detection
        console.print("[bold]Detecting market regime...[/bold]")
        try:
            regime_result = run_regime_detection(dp, ticker, lookback_days=lookback)
            display_regime_summary(regime_result)
            console.print()
        except Exception as e:
            console.print(f"  [yellow]Regime detection failed ({e}), continuing without[/yellow]\n")

    # ══════════════════════════════════════════════════════════════════════
    #  Section 1: Market Context — S/R, Price Chart, Volume Profile, Vol-Price
    # ══════════════════════════════════════════════════════════════════════
    from rich.rule import Rule
    console.print(Rule("[bold magenta]Market Context"))

    # S/R levels
    sr_levels = find_support_resistance(df.tail(180))
    if sr_levels:
        display_support_resistance(sr_levels, current_price=current_price)

    # Price chart with S/R overlay
    chart_df = df.tail(60)
    chart = ascii_price_chart(
        chart_df,
        title=f"{ticker} (Last 60 Trading Days)",
        support_resistance=sr_levels,
    )
    console.print(f"\n{chart}\n")

    # Volume Profile + AVWAP
    anchor_dates = _build_anchor_dates(df, catalog, ticker, category, end)
    vp_df = df.tail(max(60, days * 3))
    vp = build_volume_profile(vp_df, ticker, num_bins=bins, anchor_dates=anchor_dates)
    if vp:
        display_volume_profile(vp)

    # Volume-Price Authenticity
    console.print()
    vp_profile = analyze_volume_price(df, report_last_n=days)
    if vp_profile:
        vp_profile.ticker = ticker
        display_volume_price_profile(vp_profile, show_daily=False)

    # ══════════════════════════════════════════════════════════════════════
    #  Section 2: Event Analysis (if category found)
    # ══════════════════════════════════════════════════════════════════════
    event_profile = None
    event_similarity_results = []
    event_baseline = None
    event_signal_stats = None

    if category:
        events = catalog.search(category=category, ticker=ticker, end=date.today())
        if events:
            console.print()
            console.print(Rule(f"[bold magenta]Event Analysis — {cat_label}"))

            # Target window for event comparison
            t_date = end
            result = _fetch_target_window(dp, ticker, t_date, days, days)
            if result is not None:
                target_window, target_norm = result

                # Exclude self-comparisons
                tw_start = target_window.index[0].date()
                tw_end = target_window.index[-1].date()
                compare_events = [e for e in events if not (tw_start <= e.date <= tw_end)]

                if compare_events:
                    windows, event_similarity_results = _compare_events(
                        dp, ticker, target_norm, compare_events, days, days,
                    )

                    if event_similarity_results:
                        display_similarity_results(event_similarity_results, top_n=top_n)

                        top_matches = sorted(event_similarity_results,
                                             key=lambda s: s.composite_score, reverse=True)
                        display_comparison_chart(target_norm, top_matches, max_overlays=3)

                        event_profile = build_pattern_profile(
                            ticker, category.value, windows, event_similarity_results,
                        )
                        event_baseline = compute_baseline_stats(df, horizon_days=days)
                        event_signal_stats = compute_signal_stats(event_profile, event_baseline)
                        display_pattern_profile(event_profile, signal_stats=event_signal_stats)

                        # Event forecast
                        console.print()
                        _run_event_forecast(event_similarity_results, target_window,
                                            ticker, catalog, category, dp)
        else:
            console.print(f"\n  [dim]No {cat_label} events found for {ticker} — skipping event analysis[/dim]")

    # ══════════════════════════════════════════════════════════════════════
    #  Section 3: Pattern Scan (event-free sliding window)
    # ══════════════════════════════════════════════════════════════════════
    console.print()
    console.print(Rule("[bold magenta]Pattern Scan"))

    scan_results = []
    scan_fwd: dict = {}

    if len(df) >= days * 2:
        with Progress(SpinnerColumn(), TextColumn("[bold blue]Scanning patterns...")) as progress:
            progress.add_task("scan", total=None)
            scan_results = sliding_window_scan(df, window_size=days, step=1, top_n=top_n)

        if scan_results:
            display_similarity_results(scan_results, top_n=top_n)

            # Overlay chart
            target_df = df.iloc[-days:].copy()
            target_df["rel_day"] = range(days)
            target_ref = target_df["Close"].iloc[0]
            target_df["Close_norm"] = ((target_df["Close"] / target_ref) - 1) * 100
            display_comparison_chart(target_df, scan_results, target_label="Current", max_overlays=3)

            # Forward returns table
            console.print()
            _display_scan_forward_returns(df, scan_results, days)

            # Forward averages for signal synthesis
            scan_fwd = _compute_scan_forward_avg(df, scan_results, days)

            # Scan forecast
            console.print()
            _display_forecast(df, scan_results, days, ticker)
        else:
            console.print("  [dim]No similar patterns found in scan[/dim]")
    else:
        console.print(f"  [dim]Insufficient data for {days}-day scan[/dim]")

    # ══════════════════════════════════════════════════════════════════════
    #  Section 4: Signal Summary
    # ══════════════════════════════════════════════════════════════════════
    console.print()
    console.print(Rule("[bold magenta]Signal Summary"))

    signal = _build_dashboard_signal(event_profile, scan_results, scan_fwd, vp, regime_result,
                                     event_stats=event_signal_stats, baseline=event_baseline)
    display_dashboard_signal(signal, ticker)

    # ── Export ─────────────────────────────────────────────────────────────
    if export_json:
        export_data = _build_dashboard_export(
            ticker, signal, event_profile, scan_results, scan_fwd,
            vp, sr_levels, vp_profile, regime_result,
        )
        path = Path(export_json)
        path.write_text(json.dumps(export_data, indent=2, default=str))
        console.print(f"\n[green]✓ Exported dashboard to {path}[/green]")
    else:
        console.print(
            "\n[dim]Tip: Use --export-json dash.json to export combined results[/dim]"
        )


# ── FLY command ────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("ticker")
@click.option("--width", "-w", type=float, default=None,
              help="Fixed wing width (disables width search; keeps R:R-ceiling check)")
@click.option("--select", "select_mode", type=click.Choice(["pop", "payout"]),
              default="payout", show_default=True,
              help="Width objective: 'payout' walks the R:R ladder against the "
                   "debit ceiling width/(min-rr+1); 'pop' searches for the "
                   "highest probability-of-profit positive-EV fly instead")
@click.option("--target-pop", default=0.55, show_default=True,
              help="Target probability of profit for --select pop (the engine "
                   "maximizes POP among positive-EV flies; flags when below target)")
@click.option("--min-rr", default=5.0, show_default=True,
              help="Minimum structural risk:reward in payout mode (debit ceiling width/(rr+1))")
@click.option("--band", default=1.5, show_default=True,
              help="Pin search band as % from spot in the drift direction")
@click.option("--min-dte", default=2, show_default=True, help="Minimum days to expiry")
@click.option("--max-dte", default=5, show_default=True, help="Maximum days to expiry")
@click.option("--account", type=float, default=None,
              help="Account size in dollars (sizing shown in dollars as well as %)")
@click.option("--expiry", "expiry_str", default=None,
              help="Explicit expiry (YYYY-MM-DD), overrides DTE window")
@click.option("--chain-source", type=click.Choice(["auto", "cboe", "massive", "yfinance"]),
              default="auto", show_default=True,
              help="Options chain source (auto = Massive when an API key is "
                   "configured via `qpat config set massive-api-key`, else "
                   "CBOE's free delayed feed; yfinance is the failure fallback)")
@click.option("--log", "log_entry", is_flag=True,
              help="Append the recommendation to the forward-test journal "
                   "(~/.qpat/fly_journal.jsonl); score later with `qpat journal`")
@click.option("--cron", is_flag=True,
              help="Scheduler mode: exit silently outside Mon-Fri 15:15-16:00 ET "
                   "— after-close and wake-coalesced runs would journal quotes "
                   "nobody can trade")
@click.option("--json", "as_json", is_flag=True,
              help="Emit the recommendation as JSON (for piping)")
@common_options
def fly(ticker, width, select_mode, target_pop, min_rr, band, min_dte, max_dte,
        account, expiry_str, chain_source, log_entry, cron, as_json, provider, verbose):
    """
    Recommend a 3-Day Pin Fly: a 2-5 DTE butterfly bodied on the highest
    open-interest pin strike near spot.

    Drift (5/20 EMA + 3-session momentum) picks the band direction and the
    option right; the expiry with the heaviest pin OI wins. Wing width is
    chosen for headline risk:reward by default (--select payout: the 5→3→2
    ladder against width/(min_rr+1), targeting 1:5), or for probability of
    profit (--select pop: the widest positive-EV fly that tends to actually
    pin). Skips or half-sizes around CPI/PPI/FOMC/NFP prints in the window.

    Output is analysis, not financial advice — no orders are placed.

    Examples:

      qpat fly SPY

      qpat fly SPY --min-rr 15 --account 25000

      qpat fly SPY --select pop --target-pop 0.6

      qpat fly QQQ -w 5 --expiry 2026-06-18

      qpat fly SPY --json | jq .legs
    """
    from .butterfly import ET, in_fly_log_window, recommend_fly
    from .display import display_fly

    setup_logging(verbose)
    ticker = ticker.upper()
    if cron and not in_fly_log_window(datetime.now(ET)):
        return
    expiry_override = (datetime.strptime(expiry_str, "%Y-%m-%d").date()
                       if expiry_str else None)

    if not as_json:
        if width:
            mode_note = f"Width: {width:g} (fixed) | Min R:R: 1:{min_rr:g}"
        elif select_mode == "pop":
            mode_note = f"Select: POP (target {target_pop:.0%}) | Width: searched"
        else:
            mode_note = f"Select: payout (min R:R 1:{min_rr:g}) | Width: adaptive 5→3→2"
        console.print(f"\n[bold cyan]⚡ 3-Day Pin Fly: {ticker}[/bold cyan]")
        console.print(f"   Band: {band}% | DTE: {min_dte}-{max_dte} | {mode_note}\n")

    try:
        rec = recommend_fly(
            ticker,
            min_rr=min_rr,
            band_pct=band,
            min_dte=min_dte,
            max_dte=max_dte,
            fixed_width=width,
            account=account,
            expiry_override=expiry_override,
            chain_source=chain_source,
            select=select_mode,
            target_pop=target_pop,
        )
    except Exception as e:
        if as_json:
            click.echo(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(rec.to_dict(), indent=2))
    else:
        display_fly(rec)

    if log_entry:
        from .journal import log_recommendation, JOURNAL_PATH

        entry, appended = log_recommendation(rec)
        if entry is None:
            msg = "Not journaled — no pin/expiry in this recommendation."
        elif appended:
            msg = f"Journaled to {JOURNAL_PATH} — score after expiry with: qpat journal"
        else:
            msg = "Already journaled today (same ticker/expiry/pin) — skipped."
        if as_json:
            click.echo(msg, err=True)
        else:
            console.print(f"  [dim]{msg}[/dim]\n")


# ── SWING command ───────────────────────────────────────────────────────────────

SWING_LOG_PATH = Path.home() / ".qpat" / "swing_journal.jsonl"


@cli.command()
@click.argument("ticker", default="SPY")
@click.option("--json", "as_json", is_flag=True, help="Emit the signal as JSON")
@click.option("--notify", "do_notify", is_flag=True,
              help="Send the signal to Telegram (qpat config set telegram-bot-token / telegram-chat-id)")
@click.option("--cron", is_flag=True,
              help="Scheduler mode: exit silently before the US close or on non-trading days")
@click.option("--log/--no-log", "do_log", default=True,
              help="Append the signal to ~/.qpat/swing_journal.jsonl (one per ticker per day)")
@click.option("--score", "do_score", is_flag=True,
              help="Score the journal: replay every signal from the next session's open")
def swing(ticker, as_json, do_notify, cron, do_log, do_score):
    """End-of-day swing signal for TICKER (default SPY): 2-10 day option swings.

    Evaluates the completed daily bar — trend (20/50 EMA), pullback
    reversals and S/R breakouts, confirmed by volumetrics (RVOL, OBV) —
    and emits BUY CALLS / BUY PUTS / stand-aside with an ATR-based stop,
    target, and a sized option contract (~30 DTE, ~0.60 delta). Meant to
    run nightly after the close via launchd with --notify --cron.
    """
    from .swing import (DTE_MAX, DTE_MIN, ET, MAX_HOLD_DAYS, SESSION_CLOSE_ET,
                        evaluate_swing, format_swing_message, log_swing,
                        pick_option)

    ticker = ticker.upper()
    now_et = datetime.now(ET)
    if do_score:
        _swing_score(ticker, now_et, as_json)
        return
    if cron and (now_et.weekday() >= 5 or now_et.time() < SESSION_CLOSE_ET):
        return

    warnings: list[str] = []
    dp = get_provider("yfinance")
    try:
        df = dp.get_daily_ohlcv(ticker, now_et.date() - timedelta(days=400),
                                now_et.date())
    except Exception as e:
        if cron:
            click.echo(f"swing: daily fetch failed: {e}", err=True)
            sys.exit(1)
        console.print(f"[red]Daily data error: {e}[/red]")
        sys.exit(1)

    # Mid-session, yfinance already returns today's partial bar — evaluate
    # only completed bars so the signal can't change before the close.
    if now_et.weekday() < 5 and now_et.time() < SESSION_CLOSE_ET \
            and df.index[-1].date() == now_et.date():
        df = df.iloc[:-1]
        warnings.append("session in progress — evaluating yesterday's completed bar")
    last_bar = df.index[-1].date()
    if cron and last_bar != now_et.date():
        return  # holiday / wake-coalesced run: no fresh bar, dedup covers the rest

    sr_levels = find_support_resistance(df.iloc[-130:], window=5, num_levels=5)

    # Scheduled macro prints inside the hold window are exit-risk — warn.
    try:
        from .events import EventCatalog
        upcoming = EventCatalog().search(
            ticker=ticker, start=last_bar + timedelta(days=1),
            end=last_bar + timedelta(days=MAX_HOLD_DAYS * 2))
        for ev in upcoming:
            warnings.append(f"{ev.name} ({ev.date}) lands inside the hold window")
    except Exception:
        pass  # a broken calendar must never block the signal

    sig = evaluate_swing(ticker, df, sr_levels=sr_levels, warnings=warnings)

    if sig.direction in ("long", "short") and not sig.stand_aside:
        try:
            from .options_data import fetch_chains, get_options_provider
            _, chains, src_warns = fetch_chains(
                get_options_provider("auto"), ticker,
                sig.as_of + timedelta(days=DTE_MIN),
                sig.as_of + timedelta(days=DTE_MAX))
            sig.option = pick_option(chains, sig.close, sig.direction, sig.as_of)
            sig.warnings.extend(src_warns)
            if sig.option is None:
                sig.warnings.append("no suitable contract in the 21-50 DTE "
                                    "window — pick ~30 DTE Δ0.60 manually")
        except Exception as e:
            sig.warnings.append(f"option ticket unavailable ({e}) — "
                                "signal stands on the shares")

    if do_log:
        log_swing(sig, SWING_LOG_PATH)

    if as_json:
        click.echo(json.dumps(sig.to_dict(), indent=2))
    elif not cron:
        from .display import display_swing
        display_swing(sig)

    if do_notify:
        from .notify import TelegramError, send_telegram
        try:
            send_telegram(format_swing_message(sig))
        except TelegramError as e:
            if cron:
                click.echo(f"swing: {e}", err=True)
            else:
                console.print(f"[red]{e}[/red]")
            sys.exit(1)


def _swing_score(ticker: str, now_et, as_json: bool) -> None:
    """Replay every journaled swing signal from the next session's open."""
    from .data import YFinanceProvider
    from .swing import SESSION_CLOSE_ET, load_swing_journal, score_swing_journal

    entries = [e for e in load_swing_journal(SWING_LOG_PATH)
               if e.get("ticker") == ticker]
    directional = [e for e in entries if e.get("direction") in ("long", "short")]
    if not directional:
        console.print(f"[yellow]No directional {ticker} swing signals journaled "
                      "yet — the nightly cron fills ~/.qpat/swing_journal.jsonl.[/yellow]")
        return

    first = min(date.fromisoformat(e["as_of"]) for e in directional)
    # Unadjusted bars: journaled levels are raw prices — adjusted history
    # would shift under them at every ex-dividend date.
    df = YFinanceProvider().get_daily_ohlcv(
        ticker, first, now_et.date(), auto_adjust=False)
    # Only completed sessions count; drop today's partial bar mid-session.
    if now_et.weekday() < 5 and now_et.time() < SESSION_CLOSE_ET \
            and df.index[-1].date() == now_et.date():
        df = df.iloc[:-1]

    def get_bars(tkr, as_of_iso):
        after = df[df.index.date > date.fromisoformat(as_of_iso)]
        return after if not after.empty else None

    stats = score_swing_journal(entries, get_bars)
    if as_json:
        click.echo(json.dumps(stats, indent=2))
    else:
        from .display import display_swing_score
        display_swing_score(ticker, stats)


# ── JOURNAL command ─────────────────────────────────────────────────────────────

@cli.command()
@click.option("--ticker", "-t", default=None, help="Only show entries for this ticker")
@click.option("--json", "as_json", is_flag=True, help="Emit scored journal as JSON")
@common_options
def journal(ticker, as_json, provider, verbose):
    """
    Score the pin-fly forward-test journal.

    Entries logged with `qpat fly --log` are scored once their expiry has
    passed: settle distance from the recommended pin for every entry, and
    fly P&L at the mid debit for priced PASS entries. Settlement uses the
    expiry-day close (exact for PM-settled SPY/QQQ/equities; AM-settled
    index options like SPX differ).
    """
    from .journal import load_journal, score_journal, summarize
    from .display import display_journal

    setup_logging(verbose)
    entries = load_journal()
    if ticker:
        entries = [e for e in entries if e.get("ticker") == ticker.upper()]
    if not entries:
        console.print("[yellow]Journal is empty — log recommendations with: qpat fly TICKER --log[/yellow]")
        return

    dp = get_provider("yfinance")

    def get_close(tkr: str, day: date) -> Optional[float]:
        # Unadjusted close: journaled strikes are raw prices, and adjusted
        # history shifts under them at every ex-dividend date, silently
        # re-scoring old entries.
        for symbol in (tkr, f"^{tkr}"):
            try:
                df = dp.get_daily_ohlcv(symbol, day - timedelta(days=7), day,
                                        auto_adjust=False)
            except ValueError:
                continue
            if df.index[-1].date() == day:
                return float(df["Close"].iloc[-1])
            return None
        return None

    scored, pending = score_journal(entries, get_close)
    stats = summarize(scored)

    if as_json:
        click.echo(json.dumps(
            {"summary": stats, "scored": scored, "pending": pending}, indent=2))
    else:
        display_journal(scored, pending, stats)


# ── BACKTEST command ────────────────────────────────────────────────────────────

@cli.command()
@click.argument("ticker")
@click.option("--event-type", "-e", default="all",
              type=click.Choice(["all", "cpi", "ppi", "fomc", "nfp", "earnings",
                                 "election", "geopolitical", "gdp", "retail_sales",
                                 "opec", "crypto", "potus"]),
              help="Event category for the event leg ('all' pools cpi/ppi/fomc/nfp)")
@click.option("--horizon", default=10, show_default=True,
              help="Trading days each signal is scored over")
@click.option("--mode", type=click.Choice(["both", "events", "scan"]),
              default="both", show_default=True, help="Which signal types to replay")
@click.option("--window", default=10, show_default=True,
              help="Scan pattern window in trading days")
@click.option("--step", default=None, type=int,
              help="Trading days between scan as-of dates (default: horizon, "
                   "so scoring windows never overlap)")
@click.option("--lookback", default=2000, show_default=True,
              help="Calendar days of history for the scan leg")
@click.option("--min-history", default=5, show_default=True,
              help="Prior events required before the event signal is scored")
@click.option("--export-json", "-o", default=None, help="Export full results to JSON")
@common_options
def backtest(ticker, event_type, horizon, mode, window, step, lookback,
             min_history, export_json, provider, verbose):
    """
    Walk-forward backtest of qpat's directional signals.

    Replays history: at each as-of date the signal is rebuilt from only the
    data available then, and its direction is scored against the realized
    next-N-day return. The hit rate is tested against the majority-class
    baseline (in a market that rose 63% of windows, "always bullish"
    already hits 63% — the signal must beat that).

    This is the command that answers "how much should I trust qpat?" —
    believe its p-values over any single signal's confidence.

    Examples:

      qpat backtest SPY

      qpat backtest SPY -e fomc --horizon 5

      qpat backtest NVDA --mode scan --lookback 3000 -o bt.json
    """
    from .backtest import run_backtest
    from .display import display_backtest

    setup_logging(verbose)
    ticker = ticker.upper()

    if event_type == "all":
        categories = [EventCategory.CPI, EventCategory.PPI,
                      EventCategory.FOMC, EventCategory.NFP]
    else:
        categories = [EventCategory(event_type)]

    cat_label = "+".join(c.value for c in categories)
    console.print(f"\n[bold cyan]⚡ Walk-forward backtest: {ticker}[/bold cyan]")
    console.print(f"   Events: {cat_label} | Horizon: {horizon}d | Mode: {mode}\n")

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}")) as progress:
        task = progress.add_task("Backtesting...", total=None)
        reports = run_backtest(
            ticker,
            categories=categories,
            horizon=horizon,
            mode=mode,
            window_size=window,
            step=step,
            lookback_days=lookback,
            min_history=min_history,
            progress_cb=lambda msg: progress.update(task, description=f"Backtesting: {msg}"),
        )

    display_backtest(reports, ticker)

    if export_json:
        path = Path(export_json)
        path.write_text(json.dumps([r.to_dict() for r in reports], indent=2, default=str))
        console.print(f"[green]✓ Exported to {path}[/green]")


# ── POTUS command ──────────────────────────────────────────────────────────────

@cli.group()
def potus():
    """View and sync presidential actions from White House RSS feeds."""
    pass


@potus.command("schedule")
@click.option("--feed", "-f", type=click.Choice(["actions", "briefings", "all"]),
              default="actions", help="Which WH feed to pull")
@click.option("--limit", "-n", default=20, help="Number of items to show")
def potus_schedule(feed, limit):
    """View latest presidential actions from the White House RSS feed."""
    from .potus import fetch_potus_feed, FEED_URLS

    feeds = list(FEED_URLS) if feed == "all" else [feed]
    all_items = []
    for f in feeds:
        try:
            items = fetch_potus_feed(f)
            all_items.extend(items)
        except Exception as e:
            console.print(f"[red]Error fetching {f} feed: {e}[/red]")

    if not all_items:
        console.print("[yellow]No items retrieved from RSS feed[/yellow]")
        return

    # Sort by date descending
    all_items.sort(key=lambda x: x.get("date") or date.min, reverse=True)
    display_potus_schedule(all_items[:limit])
    console.print(f"\n  Showing {min(limit, len(all_items))} of {len(all_items)} items")


@potus.command("sync")
@click.option("--feed", "-f", type=click.Choice(["actions", "briefings", "all"]),
              default="actions", help="Which WH feed to sync")
@click.option("--market-only", "-m", is_flag=True, help="Only sync market-relevant actions")
def potus_sync(feed, market_only):
    """Sync presidential actions from RSS to local cache."""
    from .potus import sync_potus_events

    console.print(f"[bold cyan]Syncing POTUS events from {feed} feed...[/bold cyan]")
    if market_only:
        console.print("  Filtering for market-relevant actions only")

    try:
        added = sync_potus_events(feed=feed, market_only=market_only)
        if added:
            console.print(f"[green]  Added {len(added)} new events to cache[/green]")
            for evt in added[:10]:
                console.print(f"    + {evt.date} — {evt.name}")
            if len(added) > 10:
                console.print(f"    ... and {len(added) - 10} more")
        else:
            console.print("[dim]  No new events to add (cache is up to date)[/dim]")
    except Exception as e:
        console.print(f"[red]Error syncing: {e}[/red]")


@potus.command("list")
def potus_list():
    """Show all POTUS events (built-in + cached)."""
    catalog = EventCatalog()
    events = catalog.filter_by_category(EventCategory.POTUS)
    if not events:
        console.print("[yellow]No POTUS events found[/yellow]")
        return
    display_event_list(events)
    console.print(f"\n  Total: {len(events)} POTUS events")


if __name__ == "__main__":
    cli()
