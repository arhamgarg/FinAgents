import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.parametrize("name", ["pm_gate", "risk_debate", "reliance"])
def test_report_creates_parent_and_preserves_previous_runs(tmp_path, name):
    script = importlib.import_module(f"scripts.uat_{name}")
    output = tmp_path / "reports" / "uat" / f"{name}.md"

    script._append_to_verification_md(output, "\n### Run one\n\nevidence one\n")
    script._append_to_verification_md(output, "\n### Run two\n\nevidence two\n")

    content = output.read_text()
    assert content.count(script._SECTION_MARKER) == 1
    assert content.index("evidence one") < content.index("evidence two")
    assert script._DEFAULT_OUTPUT.parent == Path("reports/uat")


@pytest.mark.parametrize("name", ["pm_gate", "risk_debate"])
def test_default_ticker_asset_is_independent_of_working_directory(tmp_path, monkeypatch, name):
    script = importlib.import_module(f"scripts.uat_{name}")
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(tickers=None, ticker_file=script._DEFAULT_TICKER_FILE)

    assert script._parse_tickers_arg(args) == ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SUZLON"]


def test_trader_creates_report_without_planning_scaffold(tmp_path, monkeypatch):
    import httpx

    from scripts import uat_trader_hold_rate as script

    output = tmp_path / "reports" / "uat" / "trader-hold-rate.md"
    args = SimpleNamespace(
        dry_run=False,
        tickers="RELIANCE",
        date="2026-04-21",
        output=output,
    )
    monkeypatch.setattr(script, "_parse_args", lambda: args)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            content=b"Symbol,Series,Band\nRELIANCE,EQ,20\n",
            raise_for_status=lambda: None,
        ),
    )
    monkeypatch.setattr(
        script,
        "run_pipeline",
        AsyncMock(
            return_value=script.UatResult(
                "RELIANCE",
                0.8,
                "BUY",
                "BUY",
                "",
                3,
                {"technical": 0.8, "fundamentals": 0.8, "news": 0.8},
            )
        ),
    )

    assert script.main() == 0
    first = output.read_text()
    assert "RELIANCE" in first
    assert "## Hold-Rate Results" in first
    assert script.main() == 0
    assert output.read_text().startswith(first)
