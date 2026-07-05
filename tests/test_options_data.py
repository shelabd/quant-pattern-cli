"""Offline tests for the options chain provider layer. No network access."""

import io
import json
import urllib.error
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_patterns.butterfly import (
    CHAIN_COLUMNS,
    DEFAULT_IV_FALLBACK,
    score_pins,
)
from quant_patterns.options_data import (
    CboeAPIError,
    CboeOptionsProvider,
    MassiveAPIError,
    MassiveOptionsProvider,
    OptionsChainProvider,
    YFinanceOptionsProvider,
    _sides_to_frame,
    cboe_chain_rows,
    ChainSourceError,
    fetch_chains,
    get_massive_api_key,
    get_options_provider,
    massive_chain_frame,
)

EXPIRY = date(2026, 6, 17)


def contract(ctype, strike, oi=1000, vol=50, bid=1.0, ask=1.1, last=1.05,
             iv=0.2, gamma=0.05, expiry=EXPIRY):
    return {
        "details": {
            "strike_price": strike,
            "contract_type": ctype,
            "expiration_date": expiry.isoformat(),
        },
        "open_interest": oi,
        "day": {"volume": vol, "close": last},
        "last_quote": {"bid": bid, "ask": ask},
        "last_trade": {"price": last},
        "implied_volatility": iv,
        "greeks": {"gamma": gamma} if gamma is not None else {},
    }


# ── Snapshot → chain frame mapping ───────────────────────────────────────────


class TestMassiveChainFrame:
    def test_merges_call_and_put_per_strike(self):
        frame = massive_chain_frame([
            contract("call", 580, oi=5000, bid=2.0, ask=2.2, iv=0.18, gamma=0.04),
            contract("put", 580, oi=3000, bid=1.8, ask=2.0, iv=0.22, gamma=0.06),
        ])
        assert list(frame.columns) == CHAIN_COLUMNS + ["gamma"]
        row = frame.iloc[0]
        assert row["strike"] == 580.0
        assert row["call_oi"] == 5000 and row["put_oi"] == 3000
        assert row["call_bid"] == 2.0 and row["put_ask"] == 2.0
        assert row["iv"] == pytest.approx(0.20)      # mean of 0.18 / 0.22
        assert row["gamma"] == pytest.approx(0.05)   # mean of 0.04 / 0.06

    def test_missing_quote_and_oi_become_zero(self):
        c = contract("call", 100)
        c["last_quote"] = None
        c["last_trade"] = None
        c["day"] = {}
        c["open_interest"] = None
        frame = massive_chain_frame([c])
        row = frame.iloc[0]
        assert row["call_bid"] == 0.0 and row["call_ask"] == 0.0
        assert row["call_last"] == 0.0
        assert row["call_oi"] == 0 and row["call_vol"] == 0
        # put side never present at all
        assert row["put_oi"] == 0 and row["put_bid"] == 0.0

    def test_last_falls_back_to_day_close(self):
        c = contract("call", 100, last=3.3)
        c["last_trade"] = {}
        frame = massive_chain_frame([c])
        assert frame.iloc[0]["call_last"] == 3.3

    def test_iv_fallback_when_missing(self):
        c = contract("call", 100, iv=None)
        frame = massive_chain_frame([c])
        assert frame.iloc[0]["iv"] == DEFAULT_IV_FALLBACK

    def test_gamma_nan_when_absent(self):
        frame = massive_chain_frame([contract("call", 100, gamma=None)])
        assert np.isnan(frame.iloc[0]["gamma"])

    def test_strikes_sorted_and_non_options_skipped(self):
        frame = massive_chain_frame([
            contract("put", 105),
            contract("call", 95),
            {"details": {"contract_type": "other", "strike_price": 50,
                         "expiration_date": EXPIRY.isoformat()}},
        ])
        assert list(frame["strike"]) == [95.0, 105.0]

    def test_empty_contracts_give_empty_frame_with_columns(self):
        frame = massive_chain_frame([])
        assert frame.empty
        assert list(frame.columns) == CHAIN_COLUMNS + ["gamma"]


# ── CBOE delayed-quotes mapping ──────────────────────────────────────────────


def cboe_option(occ, oi=1000, vol=50, bid=1.0, ask=1.1, last=1.05, iv=0.2, gamma=0.05):
    return {"option": occ, "open_interest": float(oi), "volume": float(vol),
            "bid": bid, "ask": ask, "last_trade_price": last,
            "iv": iv, "gamma": gamma}


class TestCboeChainRows:
    def test_parses_occ_and_merges_sides(self):
        rows = cboe_chain_rows([
            cboe_option("SPY260617C00580000", oi=5000, iv=0.18, gamma=0.04),
            cboe_option("SPY260617P00580000", oi=3000, iv=0.22, gamma=0.06),
        ])
        assert list(rows) == [EXPIRY]
        frame = _sides_to_frame(rows[EXPIRY])
        row = frame.iloc[0]
        assert row["strike"] == 580.0
        assert row["call_oi"] == 5000 and row["put_oi"] == 3000
        assert row["iv"] == pytest.approx(0.20)
        assert row["gamma"] == pytest.approx(0.05)

    def test_weekly_roots_fold_into_same_chain(self):
        rows = cboe_chain_rows([
            cboe_option("SPX260617C04000000"),
            cboe_option("SPXW260617P04000000"),
        ])
        strikes = rows[EXPIRY]
        assert "call_oi" in strikes[4000.0] and "put_oi" in strikes[4000.0]

    def test_groups_multiple_expiries(self):
        rows = cboe_chain_rows([
            cboe_option("SPY260617C00580000"),
            cboe_option("SPY260619C00580000"),
        ])
        assert sorted(rows) == [EXPIRY, date(2026, 6, 19)]

    def test_unparseable_symbols_skipped(self):
        assert cboe_chain_rows([{"option": "garbage"}, {"option": None}, {}]) == {}

    def test_float_oi_becomes_int_and_missing_fields_zero(self):
        rows = cboe_chain_rows([{"option": "SPY260617C00580000",
                                 "open_interest": 2.0, "iv": None}])
        frame = _sides_to_frame(rows[EXPIRY])
        assert frame["call_oi"].dtype == np.int64
        row = frame.iloc[0]
        assert row["call_oi"] == 2
        assert row["call_bid"] == 0.0 and row["iv"] == DEFAULT_IV_FALLBACK


class TestCboeProvider:
    def _payload(self, options):
        return {"data": {"options": options, "close": 580.0}, "symbol": "SPY"}

    def test_window_filters_expiries(self, monkeypatch):
        payload = self._payload([
            cboe_option("SPY260617C00580000"),
            cboe_option("SPY260710C00580000"),  # outside window
        ])
        opener, calls = fake_urlopen([payload])
        monkeypatch.setattr("quant_patterns.options_data.urllib.request.urlopen", opener)

        chains = CboeOptionsProvider().get_chains_window("SPY", EXPIRY, date(2026, 6, 19))
        assert [exp for exp, _ in chains] == [EXPIRY]
        assert calls[0].full_url.endswith("/SPY.json")

    def test_index_retries_with_underscore(self, monkeypatch):
        attempts = []

        def opener(req, timeout=None):
            attempts.append(req.full_url)
            if len(attempts) == 1:
                raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)
            body = json.dumps(self._payload([cboe_option("SPXW260617C04000000")]))
            return io.BytesIO(body.encode())
        monkeypatch.setattr("quant_patterns.options_data.urllib.request.urlopen", opener)

        chains = CboeOptionsProvider().get_chains_window("SPX", EXPIRY, EXPIRY)
        assert attempts[0].endswith("/SPX.json") and attempts[1].endswith("/_SPX.json")
        assert len(chains) == 1

    def test_unknown_symbol_raises(self, monkeypatch):
        def opener(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)
        monkeypatch.setattr("quant_patterns.options_data.urllib.request.urlopen", opener)

        with pytest.raises(CboeAPIError):
            CboeOptionsProvider().get_chains_window("NOPE", EXPIRY, EXPIRY)


# ── score_pins prefers provider gamma ────────────────────────────────────────


class TestProviderGammaInScoring:
    def _chain(self, gammas):
        strikes = [99.0, 100.0, 101.0]
        df = pd.DataFrame({
            "strike": strikes,
            "call_oi": [1000, 1000, 1000], "put_oi": [0, 0, 0],
            "call_vol": [0, 0, 0], "put_vol": [0, 0, 0],
            "call_bid": [1.0] * 3, "call_ask": [1.1] * 3, "call_last": [1.05] * 3,
            "put_bid": [1.0] * 3, "put_ask": [1.1] * 3, "put_last": [1.05] * 3,
            "iv": [0.2] * 3,
        })
        if gammas is not None:
            df["gamma"] = gammas
        return df

    def test_provider_gamma_drives_ranking(self):
        # Equal OI everywhere; server gamma alone makes 99 the pin even
        # though local BS gamma would favor the ATM 100 strike.
        chain = self._chain([0.50, 0.01, 0.01])
        scored = score_pins(chain, spot=100.0, dte=3, band_pct=2.0, drift="neutral")
        assert scored.iloc[0]["strike"] == 99.0

    def test_missing_provider_gamma_falls_back_to_bs(self):
        chain = self._chain([np.nan, np.nan, np.nan])
        scored = score_pins(chain, spot=100.0, dte=3, band_pct=2.0, drift="neutral")
        assert (scored["gamma"] > 0).all()

    def test_no_gamma_column_still_works(self):
        chain = self._chain(None)
        scored = score_pins(chain, spot=100.0, dte=3, band_pct=2.0, drift="neutral")
        assert (scored["gamma"] > 0).all()


# ── Massive HTTP client ──────────────────────────────────────────────────────


def fake_urlopen(pages):
    """Return a urlopen stub serving canned JSON payloads keyed by URL prefix
    match order. Records requested URLs and auth headers."""
    calls = []

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=None):
        calls.append(req)
        if not pages:
            raise AssertionError("unexpected extra request")
        return FakeResponse(json.dumps(pages.pop(0)).encode())

    return opener, calls


class TestMassiveProvider:
    def test_pagination_follows_next_url(self, monkeypatch):
        page1 = {"results": [contract("call", 100)],
                 "next_url": "https://api.massive.com/v3/snapshot/options/SPY?cursor=abc"}
        page2 = {"results": [contract("put", 100)]}
        opener, calls = fake_urlopen([page1, page2])
        monkeypatch.setattr("quant_patterns.options_data.urllib.request.urlopen", opener)

        prov = MassiveOptionsProvider("test-key")
        chains = prov.get_chains_window("SPY", EXPIRY, EXPIRY)

        assert len(calls) == 2
        assert calls[0].headers["Authorization"] == "Bearer test-key"
        assert "expiration_date.gte" in calls[0].full_url
        assert calls[1].full_url.endswith("cursor=abc")
        assert len(chains) == 1
        exp, frame = chains[0]
        assert exp == EXPIRY
        assert frame.iloc[0]["call_oi"] == 1000 and frame.iloc[0]["put_oi"] == 1000

    def test_groups_by_expiry_sorted(self, monkeypatch):
        later = date(2026, 6, 19)
        page = {"results": [contract("call", 100, expiry=later),
                            contract("call", 100, expiry=EXPIRY)]}
        opener, _ = fake_urlopen([page])
        monkeypatch.setattr("quant_patterns.options_data.urllib.request.urlopen", opener)

        chains = MassiveOptionsProvider("k").get_chains_window("SPY", EXPIRY, later)
        assert [exp for exp, _ in chains] == [EXPIRY, later]

    def test_auth_failure_raises_helpful_error(self, monkeypatch):
        def opener(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 401, "unauthorized", {}, None)
        monkeypatch.setattr("quant_patterns.options_data.urllib.request.urlopen", opener)

        with pytest.raises(MassiveAPIError, match="API key"):
            MassiveOptionsProvider("bad").get_chains_window("SPY", EXPIRY, EXPIRY)


# ── Factory + key resolution + fallback ──────────────────────────────────────


class TestFactory:
    def test_auto_without_key_is_cboe(self, monkeypatch):
        monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
        monkeypatch.setattr("quant_patterns.options_data.load_config", lambda: {})
        assert isinstance(get_options_provider("auto"), CboeOptionsProvider)

    def test_explicit_cboe(self, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", "env-key")
        assert isinstance(get_options_provider("cboe"), CboeOptionsProvider)

    def test_auto_with_env_key_is_massive(self, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", "env-key")
        prov = get_options_provider("auto")
        assert isinstance(prov, MassiveOptionsProvider)
        assert prov.api_key == "env-key"

    def test_config_key_used_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
        monkeypatch.setattr("quant_patterns.options_data.load_config",
                            lambda: {"massive_api_key": "cfg-key"})
        assert get_massive_api_key() == "cfg-key"

    def test_explicit_massive_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
        monkeypatch.setattr("quant_patterns.options_data.load_config", lambda: {})
        with pytest.raises(ValueError, match="massive-api-key"):
            get_options_provider("massive")

    def test_explicit_yfinance_ignores_key(self, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", "env-key")
        assert isinstance(get_options_provider("yfinance"), YFinanceOptionsProvider)

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown chain source"):
            get_options_provider("ibkr")


class _StubProvider(OptionsChainProvider):
    def __init__(self, result=None, error=None, label="Stub"):
        self.result, self.error, self.label = result, error, label

    def name(self):
        return self.label

    def get_chains_window(self, ticker, start, end):
        if self.error:
            raise self.error
        return self.result or []


class TestFetchChains:
    def test_success_passes_through(self):
        chains = [(EXPIRY, massive_chain_frame([contract("call", 100)]))]
        prov = _StubProvider(result=chains, label="Massive (OPRA)")
        used, got, warnings = fetch_chains(prov, "SPY", EXPIRY, EXPIRY)
        assert used is prov and got is chains and warnings == []

    def test_provider_failure_raises_no_silent_fallback(self):
        # A CBOE/Massive failure must fail loudly: yfinance chains lack OI
        # on fresh weeklies and would corrupt pin scoring invisibly.
        prov = _StubProvider(error=MassiveAPIError("boom"), label="Massive (OPRA)")
        with pytest.raises(ChainSourceError, match="chain-source yfinance"):
            fetch_chains(prov, "SPY", EXPIRY, EXPIRY)

    def test_explicit_yfinance_carries_degraded_warning(self, monkeypatch):
        chains = [(EXPIRY, massive_chain_frame([contract("put", 100)]))]
        monkeypatch.setattr(YFinanceOptionsProvider, "get_chains_window",
                            lambda self, t, s, e: chains)
        used, got, warnings = fetch_chains(YFinanceOptionsProvider(), "SPY", EXPIRY, EXPIRY)
        assert isinstance(used, YFinanceOptionsProvider)
        assert got is chains
        assert len(warnings) == 1 and "degraded" in warnings[0]

    def test_yfinance_failure_propagates(self, monkeypatch):
        monkeypatch.setattr(
            YFinanceOptionsProvider, "get_chains_window",
            lambda self, t, s, e: (_ for _ in ()).throw(RuntimeError("no net")))
        with pytest.raises(RuntimeError, match="no net"):
            fetch_chains(YFinanceOptionsProvider(), "SPY", EXPIRY, EXPIRY)
