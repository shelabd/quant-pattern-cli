"""
Rich terminal display for analysis results.

ASCII price charts, colored tables, sparklines, and formatted output.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule
from rich import box

from .analysis import Level, SimilarityResult, PatternProfile, VolumePriceProfile
from .events import MarketEvent, EventCategory, EVENT_CATEGORY_LABELS

console = Console()


# ── Ticker Info ──────────────────────────────────────────────────────────────────

def display_ticker_info(info: dict):
    """Display ticker metadata as a compact panel."""
    name = info.get("name") or "Unknown"
    quote_type = info.get("quote_type") or ""
    sector = info.get("sector")
    industry = info.get("industry")
    exchange = info.get("exchange")
    market_cap = info.get("market_cap")
    currency = info.get("currency") or "USD"
    description = info.get("description")

    # Type badge
    type_map = {"ETF": "ETF", "EQUITY": "Stock", "CRYPTOCURRENCY": "Crypto",
                "MUTUALFUND": "Fund", "INDEX": "Index", "FUTURE": "Futures"}
    type_label = type_map.get(quote_type, quote_type or "")

    # Format market cap
    cap_str = None
    if market_cap:
        if market_cap >= 1_000_000_000_000:
            cap_str = f"${market_cap / 1_000_000_000_000:.2f}T"
        elif market_cap >= 1_000_000_000:
            cap_str = f"${market_cap / 1_000_000_000:.2f}B"
        elif market_cap >= 1_000_000:
            cap_str = f"${market_cap / 1_000_000:.1f}M"
        else:
            cap_str = f"${market_cap:,.0f}"

    # Build detail line
    details = []
    if type_label:
        details.append(f"[yellow]{type_label}[/yellow]")
    if sector and industry:
        details.append(f"{sector} · {industry}")
    elif sector:
        details.append(sector)
    if exchange:
        details.append(exchange)
    if cap_str:
        details.append(f"Mkt Cap: {cap_str} {currency}")

    lines = f"[bold bright_white]{name}[/bold bright_white]"
    if details:
        lines += f"\n{'  ·  '.join(details)}"
    if description:
        lines += f"\n[dim]{description[:200]}[/dim]"

    console.print(Panel(lines, border_style="cyan", padding=(0, 1)))


# ── Utilities ───────────────────────────────────────────────────────────────────

def _sparkline(values: list[float], width: int = 30) -> str:
    """Generate a Unicode sparkline."""
    if not values or len(values) < 2:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1

    # Resample to width
    if len(values) > width:
        indices = np.linspace(0, len(values) - 1, width, dtype=int)
        values = [values[i] for i in indices]

    return "".join(blocks[min(8, int((v - mn) / rng * 8))] for v in values)


def _pct_color(val: float) -> str:
    """Return color name based on percentage value."""
    if val > 0.5:
        return "green"
    elif val > 0:
        return "bright_green"
    elif val > -0.5:
        return "bright_red"
    else:
        return "red"


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "bright_green"
    elif score >= 0.6:
        return "green"
    elif score >= 0.4:
        return "yellow"
    else:
        return "red"


# ── ASCII Chart ─────────────────────────────────────────────────────────────────

def ascii_price_chart(
    df: pd.DataFrame,
    title: str = "Price",
    height: int = 15,
    width: int = 70,
    show_volume: bool = True,
    support_resistance: Optional[list[Level]] = None,
) -> str:
    """Render an ASCII OHLC chart with optional S/R levels."""
    if df.empty:
        return "[No data]"

    close = df["Close"].values
    dates = df.index
    n = len(close)

    mn = float(np.min(close))
    mx = float(np.max(close))
    rng = mx - mn if mx != mn else 1
    price_step = rng / height

    # Build chart grid
    chart = [[" " for _ in range(width)] for _ in range(height + 1)]

    # Plot price line
    for i in range(n):
        x = int(i / max(n - 1, 1) * (width - 1))
        y = int((close[i] - mn) / rng * height)
        y = min(height, max(0, y))
        row = height - y
        if i > 0:
            prev_x = int((i - 1) / max(n - 1, 1) * (width - 1))
            prev_y = int((close[i - 1] - mn) / rng * height)
            prev_y = min(height, max(0, prev_y))
            prev_row = height - prev_y
            # Connect with line
            if prev_x < x:
                for cx in range(prev_x + 1, x):
                    frac = (cx - prev_x) / (x - prev_x)
                    cy = int(prev_row + frac * (row - prev_row))
                    cy = min(height, max(0, cy))
                    chart[cy][cx] = "─"
        chart[row][x] = "●"

    # Add S/R levels
    sr_markers = {}
    if support_resistance:
        for level in support_resistance:
            if mn <= level.price <= mx:
                y = int((level.price - mn) / rng * height)
                row = height - y
                row = min(height, max(0, row))
                marker = "S" if level.kind == "support" else "R"
                sr_markers[row] = (marker, level.price)
                for x in range(0, width, 3):
                    if chart[row][x] == " ":
                        chart[row][x] = "·"

    # Build output
    lines = []
    lines.append(f"  {title}")
    lines.append(f"  {'─' * width}")

    for r in range(height + 1):
        price_at_row = mx - r * price_step
        label = f"{price_at_row:>8.2f} │"
        row_str = "".join(chart[r])
        sr_note = ""
        if r in sr_markers:
            marker, price = sr_markers[r]
            sr_note = f"  ◄ {marker} {price:.2f}"
        lines.append(f"{label}{row_str}{sr_note}")

    # X-axis
    lines.append(f"{'':>9}└{'─' * width}")
    if len(dates) >= 2:
        start_label = dates[0].strftime("%Y-%m-%d")
        end_label = dates[-1].strftime("%Y-%m-%d")
        mid_idx = len(dates) // 2
        mid_label = dates[mid_idx].strftime("%m-%d")
        spacing = width - len(start_label) - len(end_label)
        lines.append(f"{'':>10}{start_label}{mid_label:^{spacing}}{end_label}")

    return "\n".join(lines)


# ── Display Functions ───────────────────────────────────────────────────────────

def display_event_list(events: list[MarketEvent]):
    """Display a table of events."""
    table = Table(
        title="Event Catalog",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Date", style="white", width=12)
    table.add_column("Category", style="yellow", width=14)
    table.add_column("Name", style="bright_white", width=30)
    table.add_column("Ticker", style="cyan", width=8)
    table.add_column("Description", style="dim", width=35)

    for i, e in enumerate(events, 1):
        table.add_row(
            str(i),
            e.date.isoformat(),
            e.category.value.upper(),
            e.name,
            e.ticker_specific or "BROAD",
            e.description[:35],
        )

    console.print(table)


def display_support_resistance(levels: list[Level], current_price: Optional[float] = None):
    """Display S/R levels in a formatted table."""
    table = Table(
        title="Support & Resistance Levels",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Type", width=12)
    table.add_column("Price", justify="right", width=12)
    table.add_column("Touches", justify="center", width=9)
    table.add_column("Strength", width=20)
    table.add_column("Date Range", width=25)

    for l in levels:
        color = "green" if l.kind == "support" else "red"
        strength_bar = "█" * int(l.strength * 10) + "░" * (10 - int(l.strength * 10))

        date_range = ""
        if l.first_date and l.last_date:
            date_range = f"{l.first_date} → {l.last_date}"

        # Highlight if near current price
        price_style = f"bold {color}"
        if current_price and abs(l.price - current_price) / current_price < 0.02:
            price_style = f"bold {color} on dark_red" if l.kind == "resistance" else f"bold {color} on dark_green"

        table.add_row(
            Text(l.kind.upper(), style=color),
            Text(f"${l.price:.2f}", style=price_style),
            str(l.touches),
            Text(f"{strength_bar} {l.strength:.2f}", style=color),
            date_range,
        )

    console.print(table)


def display_similarity_results(results: list[SimilarityResult], top_n: int = 10):
    """Display ranked similarity results."""
    table = Table(
        title=f"Top {min(top_n, len(results))} Pattern Matches",
        box=box.ROUNDED,
        header_style="bold blue",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Event", width=28)
    table.add_column("Date", width=12)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Label", width=14)
    table.add_column("Corr", justify="right", width=8)
    table.add_column("Dir%", justify="right", width=8)
    table.add_column("DTW", justify="right", width=8)
    table.add_column("Sparkline", width=25)

    for i, r in enumerate(results[:top_n], 1):
        score_color = _score_color(r.composite_score)
        spark = ""
        if r.window_data is not None and "Close_norm" in r.window_data.columns:
            spark = _sparkline(r.window_data["Close_norm"].tolist())

        table.add_row(
            str(i),
            r.event_name[:28],
            r.event_date.isoformat() if r.event_date else "?",
            Text(f"{r.composite_score:.3f}", style=f"bold {score_color}"),
            Text(r.score_label, style=score_color),
            f"{r.correlation:.3f}",
            f"{r.direction_match:.1%}",
            f"{r.dtw_distance:.2f}",
            spark,
        )

    console.print(table)


def display_pattern_profile(profile: PatternProfile):
    """Display the aggregated pattern profile."""
    console.print(Rule(f"[bold cyan]Pattern Profile: {profile.ticker} × {profile.category.upper()}"))
    console.print()

    # Summary stats
    stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    stats_table.add_column("Metric", style="dim", width=28)
    stats_table.add_column("Value", width=20)

    rows = [
        ("Events Analyzed", str(profile.num_events)),
        ("Avg Return Before Event", f"{profile.avg_return_before:+.3f}%"),
        ("Avg Return Event Day", f"{profile.avg_return_event_day:+.3f}%"),
        ("Avg Return After Event", f"{profile.avg_return_after:+.3f}%"),
        ("Median Return After", f"{profile.median_return_after:+.3f}%"),
        ("Positive After %", f"{profile.positive_after_pct:.1f}%"),
        ("Avg Daily Volatility", f"{profile.avg_volatility:.4f}"),
        ("Avg Volume Change (Event Day)", f"{profile.avg_volume_change:+.1f}%"),
    ]

    for metric, value in rows:
        color = "white"
        if "Return After" in metric or "Return Event" in metric:
            val = float(value.replace("%", "").replace("+", ""))
            color = "green" if val > 0 else "red"
        stats_table.add_row(metric, Text(value, style=color))

    console.print(stats_table)

    # Signal
    direction = "BULLISH" if profile.avg_return_after > 0 else "BEARISH"
    dir_color = "green" if profile.avg_return_after > 0 else "red"
    confidence = min(1.0, profile.positive_after_pct / 100 if profile.avg_return_after > 0
                     else (100 - profile.positive_after_pct) / 100)

    signal_text = Text()
    signal_text.append(f"\n  Signal: ", style="bold white")
    signal_text.append(f"{direction}", style=f"bold {dir_color}")
    signal_text.append(f" | Confidence: ", style="bold white")
    signal_text.append(f"{confidence:.1%}", style=f"bold {_score_color(confidence)}")
    signal_text.append(f" | Historical Edge: ", style="bold white")
    edge_color = "green" if profile.avg_return_after > 0 else "red"
    signal_text.append(f"{profile.avg_return_after:+.3f}%", style=f"bold {edge_color}")

    console.print(Panel(signal_text, title="[bold]Trading Signal[/bold]", border_style="cyan"))


def display_comparison_chart(
    target: pd.DataFrame,
    matches: list[SimilarityResult],
    target_label: str = "Current",
    max_overlays: int = 3,
):
    """Display overlayed normalized price charts comparing target to historical matches."""
    console.print(Rule("[bold cyan]Normalized Price Overlay"))

    if target.empty or "Close_norm" not in target.columns:
        console.print("[red]No normalized target data to display[/red]")
        return

    height = 18
    width = 65
    all_series = [("Current", target["Close_norm"].values)]

    for m in matches[:max_overlays]:
        if m.window_data is not None and "Close_norm" in m.window_data.columns:
            label = m.event_name[:15]
            all_series.append((label, m.window_data["Close_norm"].values))

    # Find global range
    all_vals = np.concatenate([s for _, s in all_series])
    all_vals = all_vals[~np.isnan(all_vals)]
    mn = float(np.min(all_vals))
    mx = float(np.max(all_vals))
    rng = mx - mn if mx != mn else 1

    markers = "●◆▲■"
    colors = ["bright_white", "bright_yellow", "bright_cyan", "bright_magenta"]

    # Build chart
    chart = [[" " for _ in range(width)] for _ in range(height + 1)]
    legend_parts = []

    for si, (label, values) in enumerate(all_series):
        marker = markers[si % len(markers)]
        legend_parts.append(f"  {marker} {label}")

        values_clean = values[~np.isnan(values)]
        n = len(values_clean)
        if n < 2:
            continue

        for i in range(n):
            x = int(i / max(n - 1, 1) * (width - 1))
            y = int((values_clean[i] - mn) / rng * height)
            y = min(height, max(0, y))
            row = height - y
            chart[row][x] = marker

    # Render
    price_step = rng / height
    lines = []
    lines.append("  Normalized Returns (% from event day)")
    lines.append(f"  {'─' * width}")

    for r in range(height + 1):
        price_at_row = mx - r * price_step
        label = f"{price_at_row:>7.2f}% │"
        row_str = "".join(chart[r])
        lines.append(f"{label}{row_str}")

    lines.append(f"{'':>9}└{'─' * width}")
    lines.append(f"{'':>10}← Before Event ───── Event Day (0) ───── After Event →")
    lines.append("")
    lines.append("  Legend: " + "  ".join(legend_parts))

    console.print(Panel("\n".join(lines), border_style="blue"))


def _next_trading_day(d: date, n: int) -> date:
    """Advance n trading days from date d (skip weekends)."""
    current = d
    days_added = 0
    while days_added < n:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            days_added += 1
    return current


def display_scan_forecast(forecast: list[dict], ticker: str, current_price: float,
                          start_date: Optional[date] = None):
    """
    Display a day-by-day price forecast table.

    Each entry in forecast: {day, price, change_pct, contributors}
    """
    if start_date is None:
        start_date = date.today()

    table = Table(
        title=f"Price Forecast — {ticker} (based on top historical matches)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Day", justify="center", width=5)
    table.add_column("Date", width=12)
    table.add_column("Projected Price", justify="right", width=16)
    table.add_column("Change", justify="right", width=10)
    table.add_column("Cumulative", justify="right", width=10)
    table.add_column("Trend", width=30)

    cum_pct = 0.0
    prices = []
    for entry in forecast:
        day = entry["day"]
        price = entry["price"]
        change_pct = entry["change_pct"]
        cum_pct = ((price / current_price) - 1) * 100
        forecast_date = _next_trading_day(start_date, day)

        prices.append(price)
        color = "green" if change_pct >= 0 else "red"
        cum_color = "green" if cum_pct >= 0 else "red"

        # Mini trend bar
        bar_len = min(20, int(abs(cum_pct) * 4))
        if cum_pct >= 0:
            trend = f"[green]{'▓' * bar_len}{'░' * (20 - bar_len)}[/green]"
        else:
            trend = f"[red]{'▓' * bar_len}{'░' * (20 - bar_len)}[/red]"

        table.add_row(
            f"+{day}",
            forecast_date.strftime("%Y-%m-%d"),
            f"${price:.2f}",
            Text(f"{change_pct:+.2f}%", style=color),
            Text(f"{cum_pct:+.2f}%", style=cum_color),
            trend,
        )

    console.print(table)

    # Summary line
    if prices:
        final = prices[-1]
        total_pct = ((final / current_price) - 1) * 100
        direction = "higher" if total_pct > 0 else "lower"
        dir_color = "green" if total_pct > 0 else "red"
        console.print(
            f"\n  [bold]Projection:[/bold] ${current_price:.2f} → "
            f"[{dir_color}]${final:.2f} ({total_pct:+.2f}% {direction})[/{dir_color}] "
            f"over {len(prices)} trading days"
        )
        console.print(
            "  [dim]Based on weighted average of top matches' forward returns. "
            "Past patterns do not guarantee future results.[/dim]\n"
        )


def display_agent_export(data: dict):
    """Display the JSON export preview for quant agent."""
    import json
    formatted = json.dumps(data, indent=2, default=str)
    console.print(Panel(formatted, title="[bold green]Agent Export (JSON)[/bold green]", border_style="green"))


def display_volume_price_profile(profile: VolumePriceProfile, show_daily: bool = True):
    """Display volume-price authenticity analysis."""
    # Color-code authenticity score
    score = profile.authenticity_score
    if score >= 0.6:
        score_color = "bright_green"
    elif score >= 0.4:
        score_color = "yellow"
    else:
        score_color = "red"

    # Classification color
    cls_map = {
        "Organic": "bright_green",
        "Likely Synthetic": "red",
        "Accumulation Phase": "yellow",
        "Distribution Phase": "yellow",
        "Mixed": "white",
    }
    cls_color = cls_map.get(profile.classification, "white")

    # Summary panel
    summary = Text()
    summary.append("  Authenticity Score: ", style="bold white")
    summary.append(f"{score:.3f}", style=f"bold {score_color}")
    summary.append(f"  |  ", style="dim")
    summary.append("Classification: ", style="bold white")
    summary.append(profile.classification, style=f"bold {cls_color}")
    summary.append(f"\n  Vol Confirmation: ", style="bold white")
    summary.append(f"{profile.volume_confirmation_pct:.1f}%", style="white")
    summary.append(f"  |  ", style="dim")
    summary.append("Avg RVOL: ", style="bold white")
    rvol_color = "green" if profile.avg_relative_volume > 1.0 else "yellow" if profile.avg_relative_volume > 0.7 else "red"
    summary.append(f"{profile.avg_relative_volume:.2f}x", style=rvol_color)
    summary.append(f"  |  ", style="dim")
    summary.append(f"High Vol Days: ", style="bold white")
    summary.append(f"{profile.high_volume_days}", style="green")
    summary.append(f"  |  ", style="dim")
    summary.append(f"Low Vol Days: ", style="bold white")
    summary.append(f"{profile.low_volume_days}", style="red")

    console.print(Panel(
        summary,
        title="[bold]Volume-Price Authenticity[/bold]",
        border_style="magenta",
        padding=(0, 1),
    ))

    # Per-day table
    if show_daily and profile.daily_metrics:
        table = Table(
            box=box.SIMPLE_HEAVY,
            header_style="bold magenta",
            show_lines=False,
        )
        table.add_column("Day", justify="center", width=5)
        table.add_column("Date", width=12)
        table.add_column("Price Chg", justify="right", width=10)
        table.add_column("RVOL", justify="right", width=7)
        table.add_column("Efficiency", justify="right", width=10)
        table.add_column("Type", width=14)
        table.add_column("Confirmed", justify="center", width=9)

        type_colors = {
            "organic": "green",
            "synthetic": "red",
            "accumulation": "yellow",
            "distribution": "yellow",
            "neutral": "dim",
        }

        for d in profile.daily_metrics:
            color = type_colors.get(d.classification, "white")
            chg_color = "green" if d.price_change_pct >= 0 else "red"
            rvol_style = "green" if d.relative_volume > 1.2 else "red" if d.relative_volume < 0.5 else "white"
            confirm_str = Text("Y" if d.volume_confirms_price else "N",
                               style="green" if d.volume_confirms_price else "red")

            table.add_row(
                str(d.rel_day),
                str(d.date),
                Text(f"{d.price_change_pct:+.2f}%", style=chg_color),
                Text(f"{d.relative_volume:.2f}x", style=rvol_style),
                f"{d.move_efficiency:.2f}",
                Text(d.classification.upper(), style=f"bold {color}"),
                confirm_str,
            )

        console.print(table)


def display_categories():
    """Display available event categories."""
    table = Table(title="Event Categories", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Category", width=15)
    table.add_column("Description", width=45)

    for cat in EventCategory:
        table.add_row(cat.value, EVENT_CATEGORY_LABELS.get(cat, ""))

    console.print(table)
