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
- **backtest.py** — Walk-forward backtester (`qpat backtest`). Pure logic except `run_backtest()`: replays the event signal (rebuilt from prior events only via `signal_stats_from_returns`) and the scan signal (trailing-history sliding scan, score²-weighted forward returns) through history, then scores hit rates against the majority-class baseline with a binomial test plus a confidence-calibration table. Defaults avoid overlapping scoring windows (step = horizon) and flag overlap when forced, since autocorrelated outcomes make p-values optimistic.
- **butterfly.py** — Pin butterfly engine (`qpat fly`). Pure logic, no Rich/Click: drift detection (5/20 EMA + 3-session momentum), gamma-weighted OI pin scoring with round-number bonus (prefers the chain's server-side `gamma` column when present, else local Black-Scholes), OI-aware expiry selection, event-skip warnings. Network only inside `recommend_fly()`; chains come from options_data's provider layer. Falls back to volume-weighted scoring when a fresh weekly chain reports zero OI. Recommends only — never routes orders. **Width selection has two modes (`--select`):** the default **payout mode** uses the adaptive ladder (5→3→2) against the debit ceiling `width/(min_rr+1)`, maximizing headline risk:reward (targets 1:5). The optional **POP mode** (`select_width_by_pop()` over `candidate_widths()`) prices every symmetric width listed in the chain and picks the highest probability-of-profit fly that is still *positive-EV* and clears a modest 1:1 R:R floor (the +EV floor stops "max POP" from buying ever-wider wings; the R:R floor blocks deep-ITM boxes) — high-POP flies are wide, so it effectively drops the 1:5 gate; it logs NO TRADE when the best fly is below `--target-pop` (default 0.55). An explicit `--width` keeps the R:R-ceiling check. **Expected-move model (drives POP-mode ranking; informational in payout mode):** `atm_iv()` interpolates ATM IV, `event_vol_addon()` adds a per-category macro vol bump (`EVENT_VOL_ADDON` table) for FOMC/CPI/PPI/NFP landing *strictly after today* through expiry (a same-day, already-printed event adds no forward uncertainty), `expected_move()` combines them in quadrature, and `prob_in_profit()`/`fly_expected_value()` (normal approx via `statistics.NormalDist`) give POP and EV against the breakevens — a fly whose ±1σ band exceeds its breakeven half-width is warned as likely to finish at a loss.
- **options_data.py** — Options chain providers for `qpat fly`. `OptionsChainProvider` ABC with `CboeOptionsProvider` (default: CBOE's free unauthenticated delayed-quotes feed, one GET per ticker with OI/quotes/IV/greeks, OCC-symbol parsing folds weekly roots like SPXW into the chain, retries index tickers under `_TICKER`), `MassiveOptionsProvider` (paid upgrade — massive.com, the rebranded Polygon.io: paginated `/v3/snapshot/options/{ticker}` sweep, Bearer auth), and `YFinanceOptionsProvider` (failure fallback wrapping `normalize_chain`). All map into `butterfly.CHAIN_COLUMNS` + a `gamma` column via the shared `_sides_to_frame()`. `get_options_provider("auto")` picks Massive when a key exists (`MASSIVE_API_KEY` env or `qpat config set massive-api-key`), else CBOE; `fetch_chains_with_fallback()` degrades paid/CBOE errors to yfinance with a warning. The mapping functions (`massive_chain_frame`, `cboe_chain_rows`) are pure and offline-testable.
- **journal.py** — Forward-test journal for `qpat fly` (`--log` flag + `qpat journal` command). Appends recommendation snapshots to `~/.qpat/fly_journal.jsonl` (same-day same-pin dedup), scores expired entries against the expiry-day close: pin accuracy (settle distance) for every entry, fly P&L (`payoff = max(0, width - |settle - body|)`) for priced PASS entries. Pure logic except jsonl IO; settle prices injected as a callable (`score_journal(entries, get_close)`), so fully offline-testable. Exists because historical chains with OI on 2-5 DTE expiries aren't available free (CBOE feed is snapshot-only; DoltHub's options DB lacks OI and short-dated expiries).
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
