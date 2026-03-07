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

from .analysis import Level, SimilarityResult, PatternProfile, VolumePriceProfile, VolumeProfile
from .events import MarketEvent, EventCategory, EVENT_CATEGORY_LABELS
from .regime import RegimeResult

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
    """Display ranked similarity results sorted by score descending."""
    sorted_results = sorted(results, key=lambda s: s.composite_score, reverse=True)
    show_n = min(top_n, len(sorted_results))
    title = f"All {show_n} Pattern Matches (by score)" if show_n == len(sorted_results) else f"Top {show_n} Pattern Matches"
    table = Table(
        title=title,
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

    for i, r in enumerate(sorted_results[:show_n], 1):
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
                          start_date: Optional[date] = None,
                          actuals: Optional[dict] = None):
    """
    Display a day-by-day price forecast table.

    Each entry in forecast: {day, price, change_pct, contributors}
    Enhanced entries may also contain: {low_25, high_75, low_min, high_max, agree_pct, confidence}
    actuals: optional dict mapping date -> actual close price for backtesting
    """
    if start_date is None:
        start_date = date.today()

    has_actuals = actuals is not None and len(actuals) > 0
    enhanced = len(forecast) > 0 and "low_25" in forecast[0]
    today = date.today()

    table = Table(
        title=f"Price Forecast — {ticker} (based on top historical matches)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Day", justify="center", width=5)
    table.add_column("Date", width=12)
    table.add_column("Predicted", justify="right", width=14)
    if has_actuals:
        table.add_column("Actual", justify="right", width=14)
        table.add_column("Miss", justify="right", width=9)
    table.add_column("Change", justify="right", width=10)
    table.add_column("Cumulative", justify="right", width=10)
    if enhanced:
        table.add_column("Range (25-75%)", justify="center", width=16)
        table.add_column("Agree", justify="center", width=7)
        table.add_column("Conf", justify="center", width=7)
    table.add_column("Trend", width=24)

    cum_pct = 0.0
    prices = []
    agree_total = 0.0
    agree_count = 0
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

        is_past = forecast_date <= today

        # Build row
        row = [f"+{day}"]

        if is_past and has_actuals:
            row.append(Text(forecast_date.strftime("%Y-%m-%d"), style="bold"))
        else:
            row.append(forecast_date.strftime("%Y-%m-%d"))

        row.append(f"${price:.2f}")

        if has_actuals:
            actual = actuals.get(forecast_date)
            if actual is not None:
                miss_pct = ((price - actual) / actual) * 100
                miss_color = "green" if abs(miss_pct) < 0.5 else "yellow" if abs(miss_pct) < 1.5 else "red"
                row.append(Text(f"${actual:.2f}", style="bold"))
                row.append(Text(f"{miss_pct:+.1f}%", style=miss_color))
            else:
                row.append(Text("—", style="dim"))
                row.append(Text("—", style="dim"))

        row.extend([
            Text(f"{change_pct:+.2f}%", style=color),
            Text(f"{cum_pct:+.2f}%", style=cum_color),
        ])

        if enhanced:
            low25 = entry.get("low_25", price)
            high75 = entry.get("high_75", price)
            spread = ((high75 - low25) / current_price) * 100
            spread_color = "green" if spread < 1 else "yellow" if spread < 2.5 else "red"
            row.append(Text(f"${low25:.0f}-${high75:.0f}", style=spread_color))

            agree = entry.get("agree_pct", 100)
            agree_color = "green" if agree >= 80 else "yellow" if agree >= 60 else "red"
            row.append(Text(f"{agree:.0f}%", style=agree_color))
            agree_total += agree
            agree_count += 1

            conf = entry.get("confidence", 1.0)
            filled = int(conf * 5)
            dots = "●" * filled + "○" * (5 - filled)
            conf_color = "green" if conf >= 0.7 else "yellow" if conf >= 0.5 else "red"
            row.append(Text(dots, style=conf_color))

        row.append(trend)

        table.add_row(*row, end_section=(is_past and has_actuals and
            _next_trading_day(start_date, day + 1) > today))

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

        if enhanced and agree_count > 0:
            last_entry = forecast[-1]
            full_low = last_entry.get("low_min", final)
            full_high = last_entry.get("high_max", final)
            avg_agree = agree_total / agree_count
            console.print(
                f"  [dim]Full range: ${full_low:.2f} – ${full_high:.2f} | "
                f"Avg consensus: {avg_agree:.0f}%[/dim]"
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


def display_volume_profile(vp: VolumeProfile):
    """Display volume profile with ASCII histogram, key levels, AVWAPs, and signal."""
    # ── Summary Panel ──
    pos_labels = {
        "above_vah": ("ABOVE Value Area", "bright_green"),
        "in_value_area": ("INSIDE Value Area", "yellow"),
        "below_val": ("BELOW Value Area", "red"),
    }
    pos_label, pos_color = pos_labels.get(vp.position, ("Unknown", "white"))

    summary = Text()
    summary.append("  Current Price: ", style="bold white")
    summary.append(f"${vp.current_price:.2f}", style="bold bright_white")
    summary.append("  |  Position: ", style="dim")
    summary.append(pos_label, style=f"bold {pos_color}")
    summary.append("\n  POC: ", style="bold white")
    summary.append(f"${vp.poc_price:.2f}", style="bold cyan")
    poc_dir = "above" if vp.poc_distance_pct > 0 else "below"
    poc_color = "green" if vp.poc_distance_pct > 0 else "red"
    summary.append(f" ({abs(vp.poc_distance_pct):.2f}% {poc_dir})", style=poc_color)
    summary.append("  |  VAH: ", style="dim")
    summary.append(f"${vp.vah_price:.2f}", style="bright_red")
    summary.append("  |  VAL: ", style="dim")
    summary.append(f"${vp.val_price:.2f}", style="bright_green")
    summary.append("\n  Period: ", style="bold white")
    summary.append(f"{vp.start_date} → {vp.end_date}", style="white")
    summary.append(f"  |  Bins: {vp.num_bins}", style="dim")

    console.print(Panel(
        summary,
        title=f"[bold]Volume Profile — {vp.ticker}[/bold]",
        border_style="cyan",
        padding=(0, 1),
    ))

    # ── ASCII Volume Profile Histogram ──
    # Show condensed bins (group into ~30 display rows)
    display_rows = 30
    bins = vp.bins
    num_bins = len(bins)
    group_size = max(1, num_bins // display_rows)

    grouped = []
    for i in range(0, num_bins, group_size):
        chunk = bins[i:i + group_size]
        total_vol = sum(b.volume for b in chunk)
        total_pct = sum(b.pct_of_total for b in chunk)
        grouped.append({
            "price_low": chunk[0].price_low,
            "price_high": chunk[-1].price_high,
            "price_mid": (chunk[0].price_low + chunk[-1].price_high) / 2,
            "volume": total_vol,
            "pct": total_pct,
        })

    max_pct = max(g["pct"] for g in grouped) if grouped else 1
    bar_width = 50

    lines = []
    lines.append("  Volume Profile (horizontal histogram)")
    lines.append(f"  {'─' * (bar_width + 22)}")

    for g in reversed(grouped):  # top price = first row
        bar_len = int(g["pct"] / max_pct * bar_width) if max_pct > 0 else 0
        price_mid = g["price_mid"]

        # Mark special levels
        marker = " "
        style_note = ""

        is_poc = abs(price_mid - vp.poc_price) <= (g["price_high"] - g["price_low"])
        is_vah = abs(price_mid - vp.vah_price) <= (g["price_high"] - g["price_low"])
        is_val = abs(price_mid - vp.val_price) <= (g["price_high"] - g["price_low"])
        is_current = g["price_low"] <= vp.current_price <= g["price_high"]

        if is_poc:
            style_note = " ◄ POC"
        elif is_vah:
            style_note = " ◄ VAH"
        elif is_val:
            style_note = " ◄ VAL"

        if is_current:
            marker = "►"
            style_note += " ◄ PRICE"

        # Color: value area bins in yellow, outside in dim
        in_va = g["price_low"] >= vp.val_price and g["price_high"] <= vp.vah_price
        if is_poc:
            bar_str = "█" * bar_len + "░" * (bar_width - bar_len)
        elif in_va:
            bar_str = "▓" * bar_len + "░" * (bar_width - bar_len)
        else:
            bar_str = "▒" * bar_len + "░" * (bar_width - bar_len)

        lines.append(f"  {marker}{price_mid:>8.2f} │{bar_str}{style_note}")

    lines.append(f"  {'':>9}└{'─' * bar_width}")
    lines.append(f"  {'':>10}Volume ──────────────────────────────────────────►")
    lines.append(f"  {'':>10}█ POC  ▓ Value Area (70%)  ▒ Low Volume  ► Current Price")

    console.print(Panel("\n".join(lines), border_style="blue"))

    # ── Key Levels Table ──
    table = Table(
        title="Key Volume Levels",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Level", width=22)
    table.add_column("Price", justify="right", width=12)
    table.add_column("Distance", justify="right", width=12)
    table.add_column("Significance", width=40)

    # POC
    table.add_row(
        Text("Point of Control (POC)", style="bold cyan"),
        Text(f"${vp.poc_price:.2f}", style="bold cyan"),
        Text(f"{vp.poc_distance_pct:+.2f}%", style=poc_color),
        "Highest volume node — strongest mean-reversion magnet",
    )
    # VAH
    vah_dist = ((vp.current_price - vp.vah_price) / vp.vah_price) * 100
    vah_color = "green" if vah_dist > 0 else "red"
    table.add_row(
        Text("Value Area High (VAH)", style="bright_red"),
        Text(f"${vp.vah_price:.2f}", style="bright_red"),
        Text(f"{vah_dist:+.2f}%", style=vah_color),
        "Upper boundary of 70% volume zone — resistance",
    )
    # VAL
    val_dist = ((vp.current_price - vp.val_price) / vp.val_price) * 100
    val_color = "green" if val_dist > 0 else "red"
    table.add_row(
        Text("Value Area Low (VAL)", style="bright_green"),
        Text(f"${vp.val_price:.2f}", style="bright_green"),
        Text(f"{val_dist:+.2f}%", style=val_color),
        "Lower boundary of 70% volume zone — support",
    )

    console.print(table)

    # ── Anchored VWAPs ──
    if vp.anchored_vwaps:
        vwap_table = Table(
            title="Anchored VWAPs",
            box=box.ROUNDED,
            header_style="bold blue",
        )
        vwap_table.add_column("Anchor", width=28)
        vwap_table.add_column("Date", width=12)
        vwap_table.add_column("VWAP Price", justify="right", width=12)
        vwap_table.add_column("Distance", justify="right", width=12)
        vwap_table.add_column("Position", width=20)

        for av in vp.anchored_vwaps:
            dist_color = "green" if av.distance_pct > 0 else "red"
            pos = "Price ABOVE" if av.distance_pct > 0 else "Price BELOW"
            pos_style = "bold green" if av.distance_pct > 0 else "bold red"

            vwap_table.add_row(
                av.anchor_label,
                av.anchor_date.isoformat(),
                Text(f"${av.vwap_price:.2f}", style="bold white"),
                Text(f"{av.distance_pct:+.2f}%", style=dist_color),
                Text(pos, style=pos_style),
            )

        console.print(vwap_table)

    # ── Signal Panel ──
    signal_text = Text()
    signal_parts = vp.signal.split(" | ")
    for i, part in enumerate(signal_parts):
        if i > 0:
            signal_text.append("\n")
        # Color-code based on content
        if "ABOVE" in part or "bullish" in part.lower():
            signal_text.append(f"  {part}", style="bright_green")
        elif "BELOW" in part or "bearish" in part.lower():
            signal_text.append(f"  {part}", style="red")
        elif "AT the POC" in part:
            signal_text.append(f"  {part}", style="bold cyan")
        else:
            signal_text.append(f"  {part}", style="yellow")

    console.print(Panel(
        signal_text,
        title="[bold]Volume Profile Signal[/bold]",
        border_style="cyan",
        padding=(0, 1),
    ))


def display_potus_schedule(items: list[dict]):
    """Display RSS feed items from White House feeds."""
    table = Table(
        title="White House — Presidential Actions",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Date", style="white", width=12)
    table.add_column("Type", style="yellow", width=18)
    table.add_column("Title", style="bright_white", width=50)
    table.add_column("Source", style="dim", width=10)

    for i, item in enumerate(items, 1):
        d = item.get("date")
        date_str = d.isoformat() if d else "—"
        category = item.get("category", "")[:18]
        title = item.get("title", "")[:50]
        source = "WH.gov"
        table.add_row(str(i), date_str, category, title, source)

    console.print(table)


def display_categories():
    """Display available event categories."""
    table = Table(title="Event Categories", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Category", width=15)
    table.add_column("Description", width=45)

    for cat in EventCategory:
        table.add_row(cat.value, EVENT_CATEGORY_LABELS.get(cat, ""))

    console.print(table)


# ── Regime Display ──────────────────────────────────────────────────────────

def _regime_color(label: str) -> str:
    """Return Rich color for regime label."""
    colors = {
        "Bull-Trend": "bright_green",
        "Bear-Trend": "red",
        "Low-Vol-Range": "yellow",
        "High-Vol-Stress": "bright_red",
    }
    return colors.get(label, "white")


def display_regime_summary(result: RegimeResult):
    """Panel with current regime label + confidence, probability bar chart."""
    color = _regime_color(result.current_regime)
    conf = result.probabilities.get(result.current_regime, 0)

    summary = Text()
    summary.append("  Current Regime: ", style="bold white")
    summary.append(result.current_regime, style=f"bold {color}")
    summary.append(f"  ({conf:.1%} confidence)", style="dim")
    summary.append("\n  Observations: ", style="bold white")
    summary.append(f"{result.n_observations}", style="white")
    summary.append("  |  ", style="dim")
    summary.append("Converged: ", style="bold white")
    conv_color = "green" if result.converged else "yellow"
    summary.append(f"{'Yes' if result.converged else 'No'}", style=conv_color)

    # Probability distribution as bar chart
    summary.append("\n\n  Regime Probabilities:\n", style="bold white")
    for label in ["Bull-Trend", "Low-Vol-Range", "Bear-Trend", "High-Vol-Stress"]:
        prob = result.probabilities.get(label, 0)
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        lc = _regime_color(label)
        summary.append(f"    {label:<18}", style=lc)
        summary.append(f" {bar} ", style=lc)
        summary.append(f"{prob:.1%}\n", style="white")

    console.print(Panel(
        summary,
        title=f"[bold]Market Regime — {result.ticker}[/bold]",
        border_style="cyan",
        padding=(0, 1),
    ))


def display_regime_states(result: RegimeResult):
    """Table of state characteristics."""
    table = Table(
        title="Regime State Characteristics",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Regime", width=18)
    table.add_column("Mean Return", justify="right", width=14)
    table.add_column("Mean Volatility", justify="right", width=14)
    table.add_column("VIX Ratio", justify="right", width=10)
    table.add_column("Frequency", justify="right", width=10)
    table.add_column("", width=22)

    for s in sorted(result.states, key=lambda x: x.mean_return, reverse=True):
        color = _regime_color(s.label)
        # Annualize: mean daily log return * 252
        ann_ret = s.mean_return * 252 * 100
        ann_vol = s.mean_volatility * np.sqrt(252) * 100 if s.mean_volatility > 0 else 0

        freq_bar = "█" * int(s.frequency_pct / 5) + "░" * (20 - int(s.frequency_pct / 5))

        ret_color = "green" if ann_ret > 0 else "red"
        table.add_row(
            Text(s.label, style=f"bold {color}"),
            Text(f"{ann_ret:+.1f}%/yr", style=ret_color),
            Text(f"{ann_vol:.1f}%/yr", style="white"),
            f"{s.mean_vix_ratio:.2f}",
            f"{s.frequency_pct:.1f}%",
            Text(freq_bar, style=color),
        )

    console.print(table)


def display_regime_chart(result: RegimeResult):
    """ASCII price chart with regime-letter timeline strip below."""
    history = result.regime_history
    if history.empty:
        return

    close = history["Close"].values
    labels = history["regime_label"].values
    dates = history.index
    n = len(close)

    height = 15
    width = 70

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
        chart[row][x] = "●"

    # Build regime timeline strip
    label_map = {"Bull-Trend": "B", "Bear-Trend": "D", "Low-Vol-Range": "R", "High-Vol-Stress": "S"}
    regime_strip = [" "] * width
    for i in range(n):
        x = int(i / max(n - 1, 1) * (width - 1))
        letter = label_map.get(labels[i], "?")
        regime_strip[x] = letter

    # Render
    lines = []
    lines.append(f"  {result.ticker} Price + Regime Timeline")
    lines.append(f"  {'─' * width}")

    for r in range(height + 1):
        price_at_row = mx - r * price_step
        label_str = f"{price_at_row:>8.2f} │"
        row_str = "".join(chart[r])
        lines.append(f"{label_str}{row_str}")

    lines.append(f"{'':>9}└{'─' * width}")

    # Regime strip with colors
    lines.append(f"{'':>9} {''.join(regime_strip)}")
    lines.append(f"{'':>9} B=Bull  D=Bear  R=Range  S=Stress")

    # Date axis
    if len(dates) >= 2:
        start_label = dates[0].strftime("%Y-%m-%d")
        end_label = dates[-1].strftime("%Y-%m-%d")
        spacing = width - len(start_label) - len(end_label)
        lines.append(f"{'':>10}{start_label}{' ' * max(0, spacing)}{end_label}")

    console.print(Panel("\n".join(lines), border_style="blue"))


def display_regime_conditional_winrates(ticker: str, category: str, regime_winrates: dict):
    """Table of win rate / avg return / sample size per regime.

    regime_winrates: {label: {"win_rate": float, "avg_return": float, "count": int}}
    """
    table = Table(
        title=f"Regime-Conditional Win Rates — {ticker} × {category.upper()}",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Regime", width=18)
    table.add_column("Win Rate", justify="right", width=10)
    table.add_column("Avg Return", justify="right", width=12)
    table.add_column("Sample", justify="center", width=8)
    table.add_column("", width=22)

    for label in ["Bull-Trend", "Low-Vol-Range", "Bear-Trend", "High-Vol-Stress"]:
        data = regime_winrates.get(label)
        if data is None:
            continue
        color = _regime_color(label)
        wr = data["win_rate"]
        avg_ret = data["avg_return"]
        count = data["count"]

        wr_color = "green" if wr > 50 else "yellow" if wr > 40 else "red"
        ret_color = "green" if avg_ret > 0 else "red"

        bar_len = int(wr / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        table.add_row(
            Text(label, style=f"bold {color}"),
            Text(f"{wr:.1f}%", style=wr_color),
            Text(f"{avg_ret:+.3f}%", style=ret_color),
            str(count),
            Text(bar, style=wr_color),
        )

    console.print(table)
