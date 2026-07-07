# ⚡ Quant Pattern CLI (`qpat`)

Historical price pattern analysis around key market events. Find how tickers behave around CPI, FOMC, earnings, elections, geopolitical events, crypto events, and more — then compare current behavior to historical patterns. Or skip events entirely and scan all of history for similar price action.

Built to feed a downstream quant agent for informed trading decisions.

## Quick Start

```bash
# Install
pip install -e .

# Analyze SPY around FOMC decisions
qpat analyze SPY -e fomc

# Analyze NVDA around its earnings reports
qpat analyze NVDA -e earnings -b 5 -a 15

# Analyze SPY price action around NVDA earnings
qpat analyze SPY -e earnings --event-ticker NVDA -b 5 -a 15

# Analyze BTC around major crypto events (FTX, Luna, ETF approvals, etc.)
qpat analyze BTC-USD -e crypto -b 10 -a 180

# Scan history for similar price action (no events needed)
qpat scan SPY --days 10 --lookback 1000

# Compare current SPY behavior to a specific historical period
qpat compare SPY --cs 2026-02-20 --ce 2026-02-28 --hs 2020-02-20 --he 2020-02-28

# Support & Resistance levels
qpat sr NVDA --lookback 365

# 3-Day Pin Fly recommendation (butterfly on the highest-OI pin strike)
qpat fly SPY
qpat fly SPY --min-rr 15 --account 25000
qpat fly SPY --json | jq .legs

# Walk-forward backtest: do the signals actually predict, out of sample?
qpat backtest SPY
qpat backtest SPY -e fomc --horizon 5 -o bt.json

# Interactive guided mode
qpat interactive

# Export for quant agent
qpat export SPY -e cpi -o spy_cpi.json
```

## Commands

### `qpat analyze TICKER`
Full pattern analysis: fetches historical events, compares price behavior around each, shows similarity rankings, overlay charts, S/R levels, and a trading signal.

| Flag | Description |
|------|-------------|
| `-e` | Event type: `cpi`, `ppi`, `fomc`, `nfp`, `earnings`, `election`, `geopolitical`, `gdp`, `retail_sales`, `opec`, `crypto`, `custom` |
| `-b` | Trading days before event (default: 10) |
| `-a` | Trading days after event (default: 10) |
| `-t` | Target date to compare (default: today) |
| `-n` | Top N matches to display (default: 5) |
| `-et` | Use events for a different ticker (e.g. analyze SPY around NVDA earnings) |
| `-o` | Export JSON path |
| `-p` | Data provider: `yfinance` (default) or `ibkr` |

### `qpat scan TICKER`
Scan all historical data for periods most similar to recent price action — no events needed. Slides a window across history, ranks by similarity, and shows what happened next.

| Flag | Description |
|------|-------------|
| `-d` | Window size in trading days (default: 10) |
| `-l` | How many calendar days of history to scan (default: 1000) |
| `-s` | Slide step in trading days, 1=daily, 5=weekly (default: 1) |
| `-n` | Top N matches to display (default: 5) |
| `-o` | Export JSON path |
| `-p` | Data provider: `yfinance` (default) or `ibkr` |

```bash
# "SPY over the past 10 days behaved similar to when?"
qpat scan SPY --days 10 --lookback 1000

# Scan NVDA last 20 days against 5 years, step weekly for speed
qpat scan NVDA -d 20 -l 2000 -s 5 -n 10
```

### `qpat compare TICKER`
Direct comparison of two date ranges. Useful for "does current behavior look like March 2020?"

### `qpat fly TICKER`
**3-Day Pin Fly** — recommends a short-dated (2-5 DTE) butterfly whose body
sits on the highest open-interest "pin" strike near spot, targeting a
structural risk:reward of at least 1:12 (ideal 1:15). Drift from 5/20 EMA
alignment plus 3-session momentum picks the search band direction and the
option right (put fly when bearish or pinned below spot, call fly otherwise).
Strikes within 1.5% of spot are scored by gamma-weighted open interest with a
bonus for round numbers; the expiry whose pin OI concentration is highest
wins (a 2-DTE chain with a 22K wall beats a 5-DTE chain with 2K). Wing width
adapts 5→3→2 until the mid debit fits the ceiling `width/(min_rr+1)` — if
nothing fits, the verdict is NO TRADE. The ticket includes a suggested limit
price; **fills above the ceiling void the trade**. CPI/PPI/FOMC/NFP prints
inside the holding window trigger a warning and halve the sizing guidance
(base: 0.5-1% of account per fly). Output is **analysis, not financial
advice** — nothing is ever routed to a broker, and open interest is
end-of-day everywhere (OCC publishes it overnight), so verify the pin on
your broker before entry.

**Chain data sources.** By default chains come from **CBOE's free
delayed-quotes feed** — no key, no subscription: one GET returns the whole
chain with per-contract open interest, bid/ask, IV, and server-side greeks
(gamma is used directly in pin scoring instead of the local Black-Scholes
estimate). Quotes are 15-min delayed; OI is as of last close, like every
source. Index tickers (SPX, XSP, VIX, RUT) work too. With a
[Massive](https://massive.com) API key configured (the rebranded
Polygon.io; options snapshots require a paid plan), `qpat fly`
automatically upgrades to OPRA-consolidated NBBO snapshots. A
chain-source failure **errors out rather than silently degrading**:
yfinance chains report zero open interest on fresh weeklies, which would
corrupt OI-weighted pin scoring while looking authoritative. Pass
`--chain-source yfinance` to accept the degraded source explicitly (the
ticket then carries a data-quality warning).

```bash
qpat fly SPY                                  # CBOE free feed, zero setup
qpat config set massive-api-key <YOUR_KEY>    # optional paid upgrade
```

| Flag | Description |
|------|-------------|
| `-w` | Fixed wing width (disables the adaptive 5→3→2 ladder) |
| `--min-rr` | Minimum structural risk:reward (default: 12) |
| `--band` | Pin search band as % from spot (default: 1.5) |
| `--min-dte` / `--max-dte` | Expiry window in days (default: 2-5) |
| `--account` | Account size in dollars for sizing output |
| `--expiry` | Explicit expiry (YYYY-MM-DD), overrides the DTE window |
| `--chain-source` | `auto` (default), `cboe`, `massive`, or `yfinance` |
| `--log` | Append the recommendation to the forward-test journal |
| `--json` | Emit machine-readable JSON instead of the Rich ticket |

### `qpat journal`
**Forward-test scorecard for `qpat fly`.** Historical option chains with
open interest on 2-5 DTE expiries aren't available for free, so the fly
engine is validated forward instead: `qpat fly TICKER --log` snapshots
each recommendation (pin, OI, legs, debit) to `~/.qpat/fly_journal.jsonl`
— same-day duplicates are skipped, so it's safe to run repeatedly — and
`qpat journal` scores every entry whose expiry has passed against the
realized expiry-day close. You get **pin accuracy** (median settle
distance from the recommended pin, % within 0.5%) for all entries
including NO TRADEs, and **trade stats** (in-tent hit rate, win rate,
total P&L per fly at the mid debit, avg R multiple) for priced PASS
entries. Settlement uses the expiry-day close — exact for PM-settled
SPY/QQQ/equities, a proxy for AM-settled index options (SPX/RUT). Log
daily for a few weeks and you have the track record no free data source
can reconstruct.

```bash
qpat fly SPY --log        # log today's recommendation
qpat journal              # score everything that has expired
qpat journal -t SPY --json | jq .summary
```

### `qpat scalp [TICKER]`
**Intraday scalp floor & ceiling** (default SPY). Three families of evidence,
clustered into one actionable level per side: nearest-expiry **OI walls**
(biggest put-OI strike below spot / call-OI strike above, plus the
gamma-weighted "magnet"), the **ATM-IV expected move over the remaining
session** (shrinks into the close), and **price structure** (VWAP, opening
range, session and prior-day high/low/close, plus today's **volume profile**
— point of control and high-volume nodes, so levels with real acceptance
outrank prices gapped through). Candidates within 0.15% merge;
a cluster containing an OI wall snaps to the wall's strike. A **relative
volume** readout (today vs ~5 prior sessions over the same elapsed time)
tags the tape regime: ≥1.5× flags trend-day risk (fading extremes is
unreliable and counter-trend setups say so), ≤0.7× flags a quiet
range-bound tape. When weaker
structure sits *between* spot and the winning level (e.g. spot pressing the
session high while the call wall is $4 above), it shows as a **near
floor/ceiling** line — the trigger price must break before the main level
is in play.

Each snapshot also carries an **entry/exit plan** per side: long buys the
floor, short fades the ceiling (mean-reversion between the extremes — the
only trade the levels support). Entry is a ±0.05% zone around the level,
stop 0.10% beyond it, T1 at VWAP/magnet when one sits ≥1R inside the range
(else mid-range), T2 just inside the opposite level, with R-multiples for
both. A plan whose reward:risk to T2 is under 1.5 says *stand aside*
instead; counter-trend sides (against spot-vs-VWAP) and targets beyond the
remaining-session 1σ are flagged. Every snapshot appends to
`~/.qpat/scalp_journal.jsonl`. Designed to run every 30 minutes during the
session via launchd with `--notify --cron`:

| Flag | Description |
|------|-------------|
On days with a scheduled macro print (CPI, FOMC, NFP…) the message leads
with a warning — "stand aside into the print" — because scheduled events
steamroll mean-reversion levels; it keeps warning for an hour after the
print, then goes quiet.

| Flag | Description |
|------|-------------|
| `--notify` | Send the levels to Telegram (`qpat config set telegram-bot-token` / `telegram-chat-id`) |
| `--cron` | Scheduler mode: exit silently when the US market is closed |
| `--json` | Machine-readable output |
| `--no-log` | Skip the jsonl snapshot log |
| `--score` | Forward-test scorecard: mechanically replay every journaled setup (limit entry at the trigger, stop/T1 first-touch, stop wins same-bar ties, EOD mark-to-close) and report win/stop rates and R by side, trend alignment, and RVOL band — plus whether the stand-aside filter is saving money |

```bash
qpat scalp                    # SPY levels right now
qpat scalp QQQ --json
qpat scalp SPY --notify --cron   # what the 30-min launchd job runs
```

### `qpat scalp-watch [TICKER]`
**Level-touch alerts between the 30-minute updates.** Reads the latest
journaled `scalp` snapshot and compares it to the live price — no chain
fetch, so it's cheap enough to run every 60 seconds via launchd. Sends a
Telegram **touch** alert (with the full entry/stop/target plan) when price
enters a level's entry zone, and a **break** alert when price trades beyond
the setup's stop (the fade plan is invalidated). One alert per level per
day; when a 30-minute update moves a level, its alert re-arms.

| Flag | Description |
|------|-------------|
| `--cron` | Scheduler mode: exit silently when the US market is closed |
| `--no-notify` | Print alerts without sending to Telegram |

```bash
qpat scalp-watch                 # check SPY now, alert if at a level
qpat scalp-watch SPY --cron      # what the 60s launchd job runs
```

### `qpat sr TICKER`
Support & resistance detection using local extrema clustering. Shows touch count and strength.

### `qpat backtest TICKER`
**Walk-forward backtest** of qpat's directional signals — the command that
answers "how much should I trust this tool?". At each historical as-of date
the signal is rebuilt from only the data available then (prior events for the
event signal; trailing history for the pattern scan) and its direction is
scored against the realized next-N-day return. The hit rate is tested against
the **majority-class baseline**: in a market that rose 61% of windows,
"always bullish" already hits 61%, so the signal must beat that, not 50%. A
calibration table shows whether higher signal confidence actually translates
into a higher hit rate. Overlapping scoring windows are flagged because they
make p-values optimistic. Believe this command's p-values over any single
signal's confidence — and expect "NOT distinguishable from baseline" to be
the common verdict; that is the honest state of the evidence, not a bug.

| Flag | Description |
|------|-------------|
| `-e` | Event category for the event leg (default: `all` = cpi+ppi+fomc+nfp) |
| `--horizon` | Trading days each signal is scored over (default: 10) |
| `--mode` | `both`, `events`, or `scan` (default: both) |
| `--step` | Days between scan as-of dates (default: horizon → no overlap) |
| `--lookback` | Calendar days of scan history (default: 2000) |
| `--min-history` | Prior events required before scoring (default: 5) |
| `-o` | Export full results (per-signal outcomes included) to JSON |

### `qpat events list|categories|add`
Browse, search, and extend the event catalog.

### `qpat export TICKER`
JSON export for quant agent ingestion. Supports `--event-ticker` for cross-ticker analysis.

### `qpat interactive`
Guided analysis with prompts for each parameter.

## Architecture

```
quant_patterns/
├── cli.py        # Click CLI with all commands
├── data.py       # Data providers (yfinance, IBKR stub)
├── events.py     # Event catalog (80+ built-in events)
├── analysis.py   # Pattern matching, S/R, similarity scoring
└── display.py    # Rich terminal output (charts, tables, sparklines)
```

## Analysis Pipeline

### Event-based (`analyze`, `compare`, `export`)
1. **Event Selection** — Filter built-in catalog by category + ticker (or use `--event-ticker` for cross-ticker)
2. **Window Extraction** — Fetch OHLCV data around each event (±N days)
3. **Normalization** — Convert to % change from event-day close
4. **Similarity Scoring** — Compare current window to each historical window using:
   - Pearson correlation (shape similarity)
   - Euclidean distance (magnitude similarity)
   - Dynamic Time Warping (phase-invariant similarity)
   - Direction match % (up/down day alignment)
   - Volatility ratio
   - → Composite weighted score
5. **S/R Detection** — Local extrema clustering with touch-count strength
6. **Profile Aggregation** — Avg returns before/after, win rate, volume patterns
7. **Signal Generation** — Bullish/bearish + confidence from historical edge

### Event-free (`scan`)
1. **Data Fetch** — Pull full history (configurable lookback)
2. **Sliding Window** — Slide an N-day window across all historical data
3. **Same Similarity Engine** — Each window scored with the same 5-metric composite
4. **Deduplication** — Overlapping high-scoring windows are merged (keep best per cluster)
5. **Forward Returns** — Shows what happened +5d, +10d, +20d after each historical match

## JSON Export Schema (for Quant Agent)

```json
{
  "ticker": "SPY",
  "event_category": "fomc",
  "analysis_summary": {
    "num_events": 10,
    "avg_return_before_pct": -0.234,
    "avg_return_after_pct": 0.891,
    "positive_after_pct": 70.0,
    "avg_volatility": 0.0123
  },
  "support_resistance": [
    {"price": 580.50, "type": "support", "touches": 4, "strength": 0.8}
  ],
  "top_matches": [
    {"event": "FOMC Sep 2024", "composite_score": 0.82, "correlation": 0.91}
  ],
  "signal": {
    "direction": "bullish",
    "confidence": 0.41,
    "historical_edge_pct": 0.891,
    "n_events": 10,
    "win_rate_pct": 70.0,
    "p_value": 0.382,
    "significant_at_10pct": false,
    "baseline": {"win_rate_pct": 63.0, "mean_return_pct": 0.39, "n_windows": 740, "horizon_days": 10},
    "excess_edge_pct": 0.501
  }
}
```

`confidence` is the Wilson-interval lower bound of the directional win rate — it
shrinks with sample size, so 7/10 wins reads ~0.40 while 70/100 reads ~0.60.
`p_value` is a one-sided binomial test of the win count against the ticker's
unconditional N-day win rate (`baseline`), so an event "edge" that just matches
the ticker's normal drift is not reported as significant.

## Data Providers

### Yahoo Finance (default)
No authentication needed. Good for daily OHLCV going back 20+ years.

### Interactive Brokers
Requires TWS or IB Gateway running locally. Install extra:
```bash
pip install -e ".[ibkr]"
qpat analyze SPY -e fomc -p ibkr
```

## Adding Custom Events

```bash
# Via CLI
qpat events add -n "DeepSeek R1 Launch" -d 2025-01-20 -c geopolitical -desc "China AI shock"

# Ticker-specific
qpat events add -n "TSLA FSD v13" -d 2025-03-01 -c custom -t TSLA -desc "Full self driving release"
```

Custom events are stored in `~/.qpat/custom_events.json`.

## Extending for Your Quant Agent

The JSON export is designed for direct consumption:

```python
import json

with open("spy_fomc.json") as f:
    data = json.load(f)

signal = data["signal"]
if (signal["direction"] == "bullish"
        and signal["significant_at_10pct"]
        and (signal["excess_edge_pct"] or 0) > 0):
    # Your trading logic here
    print(f"BUY signal: {signal['excess_edge_pct']}% edge over baseline "
          f"(p={signal['p_value']}, n={signal['n_events']})")
```

## Requirements

- Python 3.10+
- Internet connection (for yfinance data fetching)
- Optional: TWS/IB Gateway for IBKR provider
