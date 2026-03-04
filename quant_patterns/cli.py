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
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from .analysis import (
    analyze_volume_price,
    build_pattern_profile,
    compare_windows,
    export_for_agent,
    find_support_resistance,
    sliding_window_scan,
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
    display_event_list,
    display_pattern_profile,
    display_scan_forecast,
    display_similarity_results,
    display_support_resistance,
    display_ticker_info,
    display_volume_price_profile,
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
@common_options
def analyze(ticker, event_type, days_before, days_after, target_date, top_n,
            export_json, event_ticker, provider, verbose):
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

    # Get relevant events
    events = catalog.search(category=category, ticker=search_ticker)
    if not events:
        console.print(f"[red]No {event_type} events found for {ticker}[/red]")
        return

    console.print(f"  Found {len(events)} relevant events\n")
    display_event_list(events)

    # Fetch target window (most recent behavior or specified date)
    if target_date:
        t_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    else:
        t_date = date.today()

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Fetching target window...")) as progress:
        task = progress.add_task("fetch", total=None)
        try:
            # For target, get recent data leading up to target_date
            target_window = fetch_event_window(dp, ticker, t_date, days_before, days_after)
            target_norm = normalize_window(target_window)
        except Exception as e:
            console.print(f"[red]Error fetching target data: {e}[/red]")
            return

    console.print(f"\n  Target window: {target_window.index[0].date()} → {target_window.index[-1].date()}")
    console.print(f"  {len(target_window)} trading days, current close: ${target_window['Close'].iloc[-1]:.2f}\n")

    # Fetch and compare each historical event
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
    try:
        # Get broader data for S/R
        broad_start = t_date - timedelta(days=180)
        broad_df = dp.get_daily_ohlcv(ticker, broad_start, t_date)
        sr_levels = find_support_resistance(broad_df)
        current_price = target_window["Close"].iloc[-1]
        display_support_resistance(sr_levels, current_price=current_price)

        # ASCII chart with S/R
        chart = ascii_price_chart(
            target_window,
            title=f"{ticker} Price (Target Window)",
            support_resistance=sr_levels,
        )
        console.print(f"\n{chart}\n")
    except Exception as e:
        logger.warning(f"S/R analysis error: {e}")
        sr_levels = []

    # Volume-Price Authenticity
    console.print()
    vp_profile = analyze_volume_price(target_window)
    if vp_profile:
        vp_profile.ticker = ticker
        display_volume_price_profile(vp_profile, show_daily=True)

    # Build and display profile
    profile = build_pattern_profile(ticker, event_type, windows, similarity_results)
    display_pattern_profile(profile)

    # Day-by-day forecast
    console.print()
    current_price = float(target_window["Close"].iloc[-1])
    _build_event_forecast(similarity_results, current_price, ticker,
                          start_date=target_window.index[-1].date())

    # Export
    if export_json:
        export_data = export_for_agent(profile, sr_levels, target_window, volume_price=vp_profile)
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


def _build_event_forecast(similarity_results, current_price, ticker, start_date=None):
    """Build a day-by-day forecast from event-based matches' post-event returns."""
    top = sorted(similarity_results, key=lambda s: s.composite_score, reverse=True)[:5]

    forward_returns_by_day: dict[int, list[tuple[float, float]]] = {}

    for r in top:
        wd = r.window_data
        if wd is None or "rel_day" not in wd.columns or "Close" not in wd.columns:
            continue

        post = wd[wd["rel_day"] >= 0].sort_values("rel_day")
        if len(post) < 2:
            continue

        weight = r.composite_score
        closes = post["Close"].values

        for i in range(1, len(closes)):
            daily_ret = (closes[i] / closes[i - 1] - 1) * 100
            forward_returns_by_day.setdefault(i, []).append((daily_ret, weight))

    if not forward_returns_by_day:
        return

    forecast = []
    projected = current_price
    for day in sorted(forward_returns_by_day.keys()):
        entries = forward_returns_by_day[day]
        total_weight = sum(w for _, w in entries)
        if total_weight == 0:
            break
        avg_ret = sum(ret * w for ret, w in entries) / total_weight
        projected = projected * (1 + avg_ret / 100)
        forecast.append({
            "day": day,
            "price": projected,
            "change_pct": avg_ret,
            "contributors": len(entries),
        })

    if forecast:
        display_scan_forecast(forecast, ticker, current_price, start_date=start_date)


def _display_forecast(df, results, window_size: int, ticker: str):
    """Build a weighted day-by-day price forecast from top matches' forward returns."""
    import pandas as pd

    current_price = float(df["Close"].iloc[-1])

    # Collect forward daily returns for each match, weighted by composite score
    forward_returns_by_day: dict[int, list[tuple[float, float]]] = {}  # day -> [(return, weight)]

    for r in results:
        if r.event_date is None:
            continue
        match_start_idx = df.index.searchsorted(pd.Timestamp(r.event_date))
        match_end_idx = match_start_idx + window_size

        if match_end_idx >= len(df):
            continue

        end_price = float(df["Close"].iloc[match_end_idx - 1])
        weight = r.composite_score

        prev_price = end_price
        for d in range(1, window_size + 1):
            fwd_idx = match_end_idx + d - 1
            if fwd_idx >= len(df):
                break
            fwd_price = float(df["Close"].iloc[fwd_idx])
            daily_ret = (fwd_price / prev_price - 1) * 100
            forward_returns_by_day.setdefault(d, []).append((daily_ret, weight))
            prev_price = fwd_price

    if not forward_returns_by_day:
        return

    # Build forecast: weighted average return per day, applied sequentially
    forecast = []
    projected = current_price
    for day in sorted(forward_returns_by_day.keys()):
        entries = forward_returns_by_day[day]
        total_weight = sum(w for _, w in entries)
        if total_weight == 0:
            break
        avg_ret = sum(r * w for r, w in entries) / total_weight
        projected = projected * (1 + avg_ret / 100)
        forecast.append({
            "day": day,
            "price": projected,
            "change_pct": avg_ret,
            "contributors": len(entries),
        })

    if forecast:
        last_date = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]
        display_scan_forecast(forecast, ticker, current_price, start_date=last_date)


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

    events_list = catalog.search(category=category, ticker=search_ticker)
    if not events_list:
        console.print(f"[red]No events found[/red]")
        return

    console.print(f"[bold cyan]Exporting {ticker} × {event_type} analysis...[/bold cyan]\n")

    display_ticker_info(fetch_ticker_info(ticker))
    console.print()

    t_date = date.today()
    target_window = fetch_event_window(dp, ticker, t_date, days_before, days_after)
    target_norm = normalize_window(target_window)

    windows = []
    similarity_results = []

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}")) as progress:
        task = progress.add_task("Processing...", total=len(events_list))
        for event in events_list:
            progress.update(task, description=f"Processing {event.name}...")
            try:
                hw = fetch_event_window(dp, ticker, event.date, days_before, days_after)
                hn = normalize_window(hw)
                windows.append(hw)
                result = compare_windows(target_norm, hn, event.name, event.date)
                result.window_data = hn
                similarity_results.append(result)
            except Exception as e:
                logger.warning(f"Skipping {event.name}: {e}")
            progress.advance(task)

    # S/R
    broad_df = dp.get_daily_ohlcv(ticker, t_date - timedelta(days=180), t_date)
    sr_levels = find_support_resistance(broad_df)

    profile = build_pattern_profile(ticker, event_type, windows, similarity_results)

    # Volume-Price Authenticity
    vp_profile = analyze_volume_price(target_window)
    if vp_profile:
        vp_profile.ticker = ticker
        display_volume_price_profile(vp_profile, show_daily=False)

    # Day-by-day forecast
    if similarity_results:
        console.print()
        current_price = float(target_window["Close"].iloc[-1])
        _build_event_forecast(similarity_results, current_price, ticker,
                          start_date=target_window.index[-1].date())

    export_data = export_for_agent(profile, sr_levels, target_window, volume_price=vp_profile)

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
    all_events = catalog.search(category=category, ticker=ticker)
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

    windows = []
    similarity_results = []

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}")) as progress:
        task = progress.add_task("Comparing...", total=len(events_list))
        for event in events_list:
            progress.update(task, description=f"{event.name}...")
            try:
                hw = fetch_event_window(dp, ticker, event.date, days_before, days_after)
                hn = normalize_window(hw)
                windows.append(hw)
                r = compare_windows(target_norm, hn, event.name, event.date)
                r.window_data = hn
                similarity_results.append(r)
            except Exception:
                pass
            progress.advance(task)

    if similarity_results:
        display_similarity_results(similarity_results, top_n=len(similarity_results))
        top = sorted(similarity_results, key=lambda s: s.composite_score, reverse=True)
        display_comparison_chart(target_norm, top, max_overlays=3)

    # S/R
    try:
        broad_df = dp.get_daily_ohlcv(ticker, t_date - timedelta(days=180), t_date)
        sr_levels = find_support_resistance(broad_df)
        display_support_resistance(sr_levels, target_window["Close"].iloc[-1])
    except Exception:
        sr_levels = []

    # Volume-Price Authenticity
    console.print()
    vp_profile = analyze_volume_price(target_window)
    if vp_profile:
        vp_profile.ticker = ticker
        display_volume_price_profile(vp_profile, show_daily=True)

    if windows and similarity_results:
        profile = build_pattern_profile(ticker, cat_choice, windows, similarity_results)
        display_pattern_profile(profile)

        # Day-by-day forecast
        console.print()
        current_price = float(target_window["Close"].iloc[-1])
        _build_event_forecast(similarity_results, current_price, ticker,
                          start_date=target_window.index[-1].date())

    # Export option
    if Prompt.ask("\n[bold]Export to JSON?", choices=["y", "n"], default="n") == "y":
        out = Prompt.ask("[bold]Output file", default=f"{ticker.lower()}_{cat_choice}_analysis.json")
        if windows and similarity_results:
            export_data = export_for_agent(profile, sr_levels, target_window, volume_price=vp_profile)
            Path(out).write_text(json.dumps(export_data, indent=2, default=str))
            console.print(f"[green]✓ Exported to {out}[/green]")


if __name__ == "__main__":
    cli()
