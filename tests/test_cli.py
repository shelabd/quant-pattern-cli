"""Tests for CLI commands using Click's CliRunner."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest
from click.testing import CliRunner

from quant_patterns.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def _mock_ohlcv(start_date="2024-01-02", periods=30, base=100.0):
    """Create a mock OHLCV DataFrame."""
    dates = pd.bdate_range(start_date, periods=periods)
    np.random.seed(42)
    close = base + np.cumsum(np.random.normal(0, 1, periods))
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 5_000_000, periods),
        },
        index=dates,
    )


# ── Version & Help ────────────────────────────────────────────────────────────


class TestCLIBasics:
    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Quant Pattern CLI" in result.output

    def test_analyze_help(self, runner):
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--event-type" in result.output
        assert "--event-ticker" in result.output

    def test_export_help(self, runner):
        result = runner.invoke(cli, ["export", "--help"])
        assert result.exit_code == 0
        assert "--event-ticker" in result.output


# ── Events commands ───────────────────────────────────────────────────────────


class TestEventsCommands:
    def test_events_list(self, runner):
        result = runner.invoke(cli, ["events", "list"])
        assert result.exit_code == 0

    def test_events_list_filter_category(self, runner):
        result = runner.invoke(cli, ["events", "list", "-c", "fomc"])
        assert result.exit_code == 0
        assert "FOMC" in result.output

    def test_events_categories(self, runner):
        result = runner.invoke(cli, ["events", "categories"])
        assert result.exit_code == 0
        assert "cpi" in result.output
        assert "fomc" in result.output

    def test_events_add(self, runner, tmp_path):
        with patch("quant_patterns.events.EventCatalog.save_custom_event") as mock_save:
            result = runner.invoke(cli, [
                "events", "add",
                "-n", "Test Event",
                "-d", "2025-06-01",
                "-c", "custom",
                "-desc", "Test description",
            ])
            assert result.exit_code == 0
            assert "Added custom event" in result.output


# ── Analyze command ───────────────────────────────────────────────────────────


class TestAnalyzeCommand:
    @patch("quant_patterns.cli.get_data_provider")
    def test_analyze_no_events(self, mock_get_dp, runner):
        mock_dp = MagicMock()
        mock_dp.name.return_value = "Mock"
        mock_get_dp.return_value = mock_dp

        result = runner.invoke(cli, ["analyze", "ZZZZZ", "-e", "earnings"])
        assert result.exit_code == 0
        assert "No earnings events found" in result.output

    @patch("quant_patterns.cli.get_data_provider")
    def test_analyze_runs(self, mock_get_dp, runner):
        mock_dp = MagicMock()
        mock_dp.name.return_value = "Mock"
        mock_dp.get_daily_ohlcv.return_value = _mock_ohlcv()
        mock_get_dp.return_value = mock_dp

        result = runner.invoke(cli, ["analyze", "SPY", "-e", "fomc", "-b", "5", "-a", "5"])
        assert result.exit_code == 0
        assert "Analyzing SPY" in result.output

    @patch("quant_patterns.cli.get_data_provider")
    def test_analyze_with_event_ticker(self, mock_get_dp, runner):
        mock_dp = MagicMock()
        mock_dp.name.return_value = "Mock"
        mock_dp.get_daily_ohlcv.return_value = _mock_ohlcv()
        mock_get_dp.return_value = mock_dp

        result = runner.invoke(cli, [
            "analyze", "SPY", "-e", "earnings", "--event-ticker", "NVDA",
            "-b", "5", "-a", "5",
        ])
        assert result.exit_code == 0
        assert "NVDA EARNINGS" in result.output

    @patch("quant_patterns.cli.get_data_provider")
    def test_analyze_event_ticker_short_flag(self, mock_get_dp, runner):
        mock_dp = MagicMock()
        mock_dp.name.return_value = "Mock"
        mock_dp.get_daily_ohlcv.return_value = _mock_ohlcv()
        mock_get_dp.return_value = mock_dp

        result = runner.invoke(cli, [
            "analyze", "SPY", "-e", "earnings", "-et", "NVDA", "-b", "5", "-a", "5",
        ])
        assert result.exit_code == 0
        assert "NVDA EARNINGS" in result.output


# ── SR command ────────────────────────────────────────────────────────────────


class TestSRCommand:
    @patch("quant_patterns.cli.get_data_provider")
    def test_sr_runs(self, mock_get_dp, runner):
        mock_dp = MagicMock()
        mock_dp.name.return_value = "Mock"
        mock_dp.get_daily_ohlcv.return_value = _mock_ohlcv(periods=60)
        mock_get_dp.return_value = mock_dp

        result = runner.invoke(cli, ["sr", "SPY", "--lookback", "90"])
        assert result.exit_code == 0
        assert "S/R Analysis" in result.output


# ── Compare command ───────────────────────────────────────────────────────────


class TestCompareCommand:
    @patch("quant_patterns.cli.get_data_provider")
    def test_compare_runs(self, mock_get_dp, runner):
        mock_dp = MagicMock()
        mock_dp.name.return_value = "Mock"
        mock_dp.get_daily_ohlcv.return_value = _mock_ohlcv()
        mock_get_dp.return_value = mock_dp

        result = runner.invoke(cli, [
            "compare", "SPY",
            "-cs", "2024-01-02", "-ce", "2024-01-31",
            "-hs", "2023-01-02", "-he", "2023-01-31",
        ])
        assert result.exit_code == 0
        assert "Comparing SPY" in result.output


# ── Export command ────────────────────────────────────────────────────────────


class TestExportCommand:
    @patch("quant_patterns.cli.get_data_provider")
    def test_export_runs(self, mock_get_dp, runner, tmp_path):
        mock_dp = MagicMock()
        mock_dp.name.return_value = "Mock"
        mock_dp.get_daily_ohlcv.return_value = _mock_ohlcv()
        mock_get_dp.return_value = mock_dp

        outfile = str(tmp_path / "test_export.json")
        result = runner.invoke(cli, [
            "export", "SPY", "-e", "fomc", "-o", outfile, "-b", "5", "-a", "5",
        ])
        assert result.exit_code == 0

    @patch("quant_patterns.cli.get_data_provider")
    def test_export_with_event_ticker(self, mock_get_dp, runner, tmp_path):
        mock_dp = MagicMock()
        mock_dp.name.return_value = "Mock"
        mock_dp.get_daily_ohlcv.return_value = _mock_ohlcv()
        mock_get_dp.return_value = mock_dp

        outfile = str(tmp_path / "test_export.json")
        result = runner.invoke(cli, [
            "export", "SPY", "-e", "earnings", "-et", "NVDA", "-o", outfile,
        ])
        assert result.exit_code == 0


# ── Scan command ──────────────────────────────────────────────────────────────


class TestScanCommand:
    def test_scan_help(self, runner):
        result = runner.invoke(cli, ["scan", "--help"])
        assert result.exit_code == 0
        assert "--days" in result.output
        assert "--lookback" in result.output
        assert "--step" in result.output

    @patch("quant_patterns.cli.get_data_provider")
    def test_scan_runs(self, mock_get_dp, runner):
        mock_dp = MagicMock()
        mock_dp.name.return_value = "Mock"
        mock_dp.get_daily_ohlcv.return_value = _mock_ohlcv(periods=200)
        mock_get_dp.return_value = mock_dp

        result = runner.invoke(cli, ["scan", "SPY", "-d", "10", "-l", "500", "-n", "3"])
        assert result.exit_code == 0
        assert "Scanning SPY" in result.output

    @patch("quant_patterns.cli.get_data_provider")
    def test_scan_with_export(self, mock_get_dp, runner, tmp_path):
        mock_dp = MagicMock()
        mock_dp.name.return_value = "Mock"
        mock_dp.get_daily_ohlcv.return_value = _mock_ohlcv(periods=200)
        mock_get_dp.return_value = mock_dp

        outfile = str(tmp_path / "scan_export.json")
        result = runner.invoke(cli, ["scan", "SPY", "-d", "10", "-l", "500", "-o", outfile])
        assert result.exit_code == 0
        assert "Exported" in result.output
