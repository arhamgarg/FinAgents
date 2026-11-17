from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "missing_decision",
        "circuit",
        "cost",
        "bull_rounds",
        "risk_rounds",
        "missing_technical",
        "missing_notes",
        "generic_approval",
        "invalid_decision",
    ],
)
def test_acceptance_controls_exit_and_report(tmp_path, monkeypatch, default_settings, failure):
    from finagents.agents.managers.schemas import PortfolioDecision
    from finagents.agents.trader.schemas import TradingSignal
    from scripts import uat_reliance as script

    state = SimpleNamespace(
        investment_debate_state=SimpleNamespace(
            bull_turns=lambda: ["bull"] * default_settings.max_debate_rounds,
            bear_turns=lambda: ["bear"] * default_settings.max_debate_rounds,
        ),
        risk_debate_state=SimpleNamespace(
            risk_turns=lambda: ["risk"] * default_settings.max_risk_discuss_rounds,
        ),
        technical_report=SimpleNamespace(circuit_band_pct=0.20),
        final_trade_decision=PortfolioDecision(
            signal=TradingSignal(
                action="BUY",
                conviction=0.8,
                entry_price=100,
                stop_loss=95,
                target_1=110,
            ),
            approved=True,
            decision="approve",
            pm_notes=(
                "The portfolio has sufficient cash headroom and acceptable sector concentration."
            ),
        ),
    )
    if failure == "missing_decision":
        state.final_trade_decision = None
    elif failure == "circuit":
        state.final_trade_decision = state.final_trade_decision.model_copy(
            update={
                "signal": state.final_trade_decision.signal.model_copy(update={"stop_loss": 50})
            }
        )
    elif failure == "cost":
        state.final_trade_decision = state.final_trade_decision.model_copy(
            update={
                "signal": state.final_trade_decision.signal.model_copy(update={"target_1": 100.1})
            }
        )
    elif failure == "bull_rounds":
        state.investment_debate_state.bull_turns = lambda: []
    elif failure == "risk_rounds":
        state.risk_debate_state = None
    elif failure == "missing_technical":
        state.technical_report = None
    elif failure == "missing_notes":
        state.final_trade_decision = state.final_trade_decision.model_copy(update={"pm_notes": ""})
    elif failure == "generic_approval":
        state.final_trade_decision = state.final_trade_decision.model_copy(
            update={
                "pm_notes": (
                    "Everything looks good and I approve this otherwise satisfactory proposal."
                )
            }
        )
    elif failure == "invalid_decision":
        state.final_trade_decision = state.final_trade_decision.model_copy(
            update={"decision": "invalid"}
        )

    output = tmp_path / "reports" / "reliance.md"
    monkeypatch.setattr("finagents.config.Settings", lambda: default_settings)
    monkeypatch.setattr(
        script,
        "_parse_args",
        lambda: SimpleNamespace(dry_run=False, output=output, date="2026-04-21"),
    )
    pipeline = AsyncMock(return_value=state)
    monkeypatch.setattr(script, "_run_pipeline", pipeline)

    assert script.main() == (1 if failure else 0)
    report = output.read_text()
    assert f"**Acceptance:** {'FAIL' if failure else 'PASS'}" in report
    assert "LLM:" in report
    pipeline.assert_awaited_once_with("2026-04-21", default_settings)


def test_dry_run_does_not_run_pipeline_or_write(tmp_path, monkeypatch):
    from scripts import uat_reliance as script

    output = tmp_path / "reports" / "reliance.md"
    monkeypatch.setattr(
        script,
        "_parse_args",
        lambda: SimpleNamespace(dry_run=True, output=output, date="2026-04-21"),
    )
    pipeline = AsyncMock()
    monkeypatch.setattr(script, "_run_pipeline", pipeline)

    assert script.main() == 0
    pipeline.assert_not_called()
    assert not output.parent.exists()
