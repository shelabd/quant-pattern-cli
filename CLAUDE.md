# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`qpat` (Quant Pattern CLI) — a Python CLI that analyzes historical price behavior around key market events (CPI, FOMC, earnings, etc.), finds similar historical patterns via multi-metric similarity scoring, and exports structured signals for a downstream quant agent.

## Build & Run

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run the CLI
qpat analyze SPY -e fomc
qpat compare SPY --cs 2026-02-20 --ce 2026-02-28 --hs 2020-02-20 --he 2020-02-28
qpat sr NVDA --lookback 365
qpat events list -c cpi
qpat export SPY -e cpi -o spy_cpi.json
qpat interactive

# Lint
ruff check .

# Tests
pytest
```

For IBKR provider: `pip install -e ".[ibkr]"` (requires TWS/IB Gateway running locally).

## Architecture

The package is `quant_patterns/` (flat, 5 modules). Entry point: `cli:cli` registered as `qpat` console script.

**Data flow:** CLI command → `EventCatalog.search()` filters events → `fetch_event_window()` pulls OHLCV around each event date → `normalize_window()` converts to % change from event-day close → `compare_windows()` scores similarity → `build_pattern_profile()` aggregates → `export_for_agent()` produces JSON.

### Module Responsibilities

- **cli.py** — Click command group (`analyze`, `compare`, `sr`, `events`, `export`, `interactive`). Orchestrates the pipeline; all commands follow the same fetch→normalize→compare→display flow.
- **data.py** — `DataProvider` ABC with `YFinanceProvider` (default) and `IBKRProvider`. Key helpers: `fetch_event_window()` extracts a trading-day window around a date with weekend/holiday buffer; `normalize_window()` produces `Close_norm` (% change from rel_day=0).
- **events.py** — `EventCategory` enum (11 categories), `MarketEvent` dataclass, `EventCatalog` with ~60 built-in events. Custom events persist to `~/.qpat/custom_events.json`.
- **analysis.py** — `compare_windows()` computes 5 metrics (Pearson correlation, Euclidean distance, banded DTW, direction match %, volatility ratio) combined into a weighted composite score (0.30/0.20/0.20/0.20/0.10); distance scores are normalized by the windows' combined dispersion so scores are comparable across volatility regimes. `find_support_resistance()` uses `scipy.signal.argrelextrema` with clustering. `build_pattern_profile()` aggregates returns/volatility/volume across event windows. `compute_signal_stats()` is the single source of truth for the trading signal: Wilson-lower-bound confidence (shrinks with sample size) and a one-sided binomial p-value against the ticker's unconditional base rate from `compute_baseline_stats()`. `export_for_agent()` produces the JSON schema for quant agent consumption.
- **display.py** — Rich terminal output. Exports a shared `console` instance. ASCII price charts with S/R overlay, sparklines, colored similarity tables, pattern profiles with trading signal panels.

### Key Data Structures

- `SimilarityResult` — per-event comparison output with all 5 metrics + composite score
- `PatternProfile` — aggregated stats across all event windows (avg returns, win rate, volatility, volume, raw `returns_after_list`)
- `SignalStats` — statistically grounded signal: direction, Wilson-shrunk confidence, binomial p-value, baseline comparison / excess edge
- `BaselineStats` — unconditional N-day forward-return distribution (the signal's null hypothesis)
- `Level` — support/resistance level with price, touches, strength, date range
- DataFrames carry a `rel_day` column (0 = event day) and `*_norm` columns after normalization

### Composite Score Weights

Correlation: 0.30, Euclidean: 0.20, DTW: 0.20, Direction match: 0.20, Volatility ratio: 0.10

## Conventions

- All ticker symbols are uppercased at command boundaries
- Provider pattern: ABC `DataProvider` → concrete implementations via `get_provider()` factory
- Events that are `ticker_specific=None` are "broad market" events and match any ticker query
- The `rel_day` column is the canonical way to align windows across different dates
- `fetch_event_window()` raises when no trading data exists within 7 days of the event date (future events from the synced macro calendar, or events predating the ticker's history) — analysis commands also pass `end=date.today()` to `catalog.search()` so unhappened events never enter comparisons
- Rich `console` is the single output sink — imported from `display.py` everywhere
