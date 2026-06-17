"""Offline tests for the fly forward-test journal. No network access."""

import json
from datetime import date

import pytest

from quant_patterns.butterfly import FlyRecommendation, Leg
from quant_patterns.journal import (
    load_journal,
    log_recommendation,
    score_entry,
    score_journal,
    summarize,
)

EXPIRY = date(2026, 6, 16)
AS_OF = date(2026, 6, 12)


def make_rec(verdict="PASS", body=733.0, width=3.0, debit=0.20, expiry=EXPIRY):
    rec = FlyRecommendation(
        ticker="SPY", spot=737.76, drift="bearish", right="PUT",
        expiry=expiry, dte=4, body_strike=body,
        selected_width=width if verdict == "PASS" else None,
        width_was_adaptive=True,
    )
    rec.verdict = verdict
    if verdict == "PASS":
        rec.debit = debit
        rec.max_profit = width - debit
        rec.legs = [Leg("BUY", 1, "PUT", body - width, expiry, 0.1),
                    Leg("SELL", 2, "PUT", body, expiry, 0.2),
                    Leg("BUY", 1, "PUT", body + width, expiry, 0.5)]
    return rec


# ── Logging ──────────────────────────────────────────────────────────────────


class TestLogging:
    def test_appends_entry(self, tmp_path):
        path = tmp_path / "journal.jsonl"
        entry, appended = log_recommendation(make_rec(), as_of=AS_OF, path=path)
        assert appended
        assert entry["ticker"] == "SPY" and entry["as_of"] == "2026-06-12"
        assert len(load_journal(path)) == 1

    def test_same_day_duplicate_skipped(self, tmp_path):
        path = tmp_path / "journal.jsonl"
        log_recommendation(make_rec(), as_of=AS_OF, path=path)
        _, appended = log_recommendation(make_rec(), as_of=AS_OF, path=path)
        assert not appended
        assert len(load_journal(path)) == 1

    def test_different_day_same_pin_logs(self, tmp_path):
        path = tmp_path / "journal.jsonl"
        log_recommendation(make_rec(), as_of=AS_OF, path=path)
        _, appended = log_recommendation(make_rec(), as_of=date(2026, 6, 13), path=path)
        assert appended and len(load_journal(path)) == 2

    def test_no_pin_not_logged(self, tmp_path):
        path = tmp_path / "journal.jsonl"
        rec = make_rec(verdict="NO TRADE")
        rec.expiry = None
        rec.body_strike = None
        entry, appended = log_recommendation(rec, as_of=AS_OF, path=path)
        assert entry is None and not appended
        assert load_journal(path) == []

    def test_no_trade_with_pin_is_logged(self, tmp_path):
        path = tmp_path / "journal.jsonl"
        entry, appended = log_recommendation(make_rec(verdict="NO TRADE"), as_of=AS_OF, path=path)
        assert appended and entry["verdict"] == "NO TRADE"

    def test_corrupt_lines_skipped(self, tmp_path):
        path = tmp_path / "journal.jsonl"
        path.write_text('{"ticker": "SPY"}\nnot json\n')
        assert load_journal(path) == [{"ticker": "SPY"}]


# ── Scoring math ─────────────────────────────────────────────────────────────


def entry_dict(**kwargs):
    rec = make_rec(**{k: v for k, v in kwargs.items() if k in ("verdict", "body", "width", "debit")})
    return {"as_of": AS_OF.isoformat(), **rec.to_dict()}


class TestScoreEntry:
    def test_settle_on_body_is_max_profit(self):
        out = score_entry(entry_dict(body=733.0, width=3.0, debit=0.20), settle=733.0)
        assert out["payoff_per_share"] == 3.0
        assert out["pnl_per_share"] == pytest.approx(2.80)
        assert out["pnl_per_fly"] == pytest.approx(280.0)
        assert out["r_multiple"] == pytest.approx(14.0)
        assert out["win"] and out["in_tent"]
        assert out["pin_dist"] == 0.0

    def test_settle_at_wing_loses_debit(self):
        out = score_entry(entry_dict(body=733.0, width=3.0, debit=0.20), settle=736.0)
        assert out["payoff_per_share"] == 0.0
        assert out["pnl_per_fly"] == pytest.approx(-20.0)
        assert not out["win"] and not out["in_tent"]

    def test_settle_beyond_wing_capped_at_debit(self):
        out = score_entry(entry_dict(body=733.0, width=3.0, debit=0.20), settle=750.0)
        assert out["pnl_per_share"] == pytest.approx(-0.20)

    def test_breakeven_band(self):
        # settle inside tent but below breakeven: payoff < debit, in_tent yet a loss
        out = score_entry(entry_dict(body=733.0, width=3.0, debit=0.20), settle=735.9)
        assert out["in_tent"] and not out["win"]

    def test_no_trade_scores_pin_only(self):
        out = score_entry(entry_dict(verdict="NO TRADE", body=733.0), settle=734.0)
        assert "win" not in out and "pnl_per_fly" not in out
        assert out["pin_dist"] == pytest.approx(1.0)
        assert out["pin_dist_pct"] == pytest.approx(1.0 / 737.76 * 100, rel=1e-3)


class TestScoreJournal:
    def test_splits_scored_and_pending(self):
        entries = [entry_dict(), entry_dict()]
        entries[1]["expiry"] = "2026-07-17"
        scored, pending = score_journal(
            entries, get_close=lambda t, d: 733.0, today=date(2026, 6, 18))
        assert len(scored) == 1 and len(pending) == 1
        assert scored[0]["settle"] == 733.0

    def test_expiry_today_stays_pending(self):
        scored, pending = score_journal(
            [entry_dict()], get_close=lambda t, d: 733.0, today=EXPIRY)
        assert not scored and len(pending) == 1

    def test_missing_close_stays_pending(self):
        scored, pending = score_journal(
            [entry_dict()], get_close=lambda t, d: None, today=date(2026, 6, 18))
        assert not scored and len(pending) == 1


class TestSummarize:
    def test_aggregates_trades_and_pins(self):
        scored = [
            score_entry(entry_dict(body=733.0, width=3.0, debit=0.20), settle=733.0),   # +280
            score_entry(entry_dict(body=733.0, width=3.0, debit=0.20), settle=740.0),   # -20
            score_entry(entry_dict(verdict="NO TRADE", body=733.0), settle=733.5),      # pin only
        ]
        stats = summarize(scored)
        assert stats["n_scored"] == 3 and stats["n_trades"] == 2
        assert stats["hit_rate"] == 0.5 and stats["win_rate"] == 0.5
        assert stats["total_pnl_per_fly"] == pytest.approx(260.0)
        assert stats["avg_r_multiple"] == pytest.approx((14.0 - 1.0) / 2)
        assert stats["pin_within_half_pct"] == pytest.approx(2 / 3, rel=1e-3)

    def test_empty(self):
        stats = summarize([])
        assert stats["n_scored"] == 0 and stats["n_trades"] == 0
        assert stats["median_abs_pin_dist_pct"] is None

    def test_json_round_trip(self, tmp_path):
        path = tmp_path / "journal.jsonl"
        log_recommendation(make_rec(), as_of=AS_OF, path=path)
        entries = load_journal(path)
        scored, _ = score_journal(entries, get_close=lambda t, d: 733.0,
                                  today=date(2026, 6, 18))
        assert json.loads(json.dumps(summarize(scored)))["n_trades"] == 1

# ── Expected-move forecast calibration ────────────────────────────────────────


def _scored_forecast(settle, pop, ev, body=733.0, width=3.0, debit=0.20):
    """A scored PASS entry carrying ex-ante POP / EV forecast fields."""
    e = entry_dict(body=body, width=width, debit=debit)
    e["prob_profit"] = pop
    e["expected_value_per_fly"] = ev
    return score_entry(e, settle=settle)


class TestCalibration:
    def test_pop_and_ev_calibration(self):
        scored = [
            _scored_forecast(settle=733.0, pop=0.7, ev=50.0),   # win,  +280/fly
            _scored_forecast(settle=740.0, pop=0.2, ev=-10.0),  # loss,  -20/fly
        ]
        cal = summarize(scored)["calibration"]
        assert cal["n_forecast"] == 2
        assert cal["mean_pred_pop"] == pytest.approx(0.45)
        assert cal["actual_win_rate"] == pytest.approx(0.5)
        # Brier = mean((0.7-1)^2, (0.2-0)^2) = (0.09 + 0.04)/2
        assert cal["pop_brier"] == pytest.approx(0.065)
        assert cal["mean_pred_ev_per_fly"] == pytest.approx(20.0)
        assert cal["mean_actual_pnl_per_fly"] == pytest.approx(130.0)
        assert cal["ev_bias_per_fly"] == pytest.approx(110.0)  # actual − predicted

    def test_absent_without_forecast_fields(self):
        # Legacy entries (pre-feature) carry prob_profit=None → no calibration.
        scored = [score_entry(entry_dict(body=733.0, width=3.0, debit=0.20), settle=733.0)]
        assert "calibration" not in summarize(scored)

    def test_buckets_group_by_predicted_pop(self):
        scored = [
            _scored_forecast(settle=733.0, pop=0.70, ev=50.0),
            _scored_forecast(settle=733.0, pop=0.65, ev=40.0),
            _scored_forecast(settle=740.0, pop=0.10, ev=-10.0),
        ]
        buckets = summarize(scored)["calibration"]["buckets"]
        assert any(b["n"] == 2 and b["range"] == "60%–100%" for b in buckets)
        assert any(b["n"] == 1 and b["range"] == "0%–20%" for b in buckets)

    def test_json_serializable(self):
        scored = [_scored_forecast(settle=733.0, pop=0.6, ev=30.0)]
        assert json.loads(json.dumps(summarize(scored)))["calibration"]["n_forecast"] == 1
