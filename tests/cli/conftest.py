from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def cli_runner() -> Any:
    from typer.testing import CliRunner

    return CliRunner()


@pytest.fixture
def tmp_signals_db(tmp_path: Path) -> Path:
    return tmp_path / "signals.db"


@pytest.fixture
def freeze_ist_today(monkeypatch: pytest.MonkeyPatch) -> str:
    frozen = "2026-04-26"

    monkeypatch.setattr(
        "finagents.clock.ist_today_str",
        lambda: frozen,
        raising=False,
    )
    return frozen


@pytest.fixture
def fake_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key-for-picker-tests")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_signals_db):
    from finagents.agents.managers.schemas import PortfolioDecision
    from finagents.agents.trader.schemas import TradingSignal
    from finagents.agents.utils.agent_states import AgentState

    ts = TradingSignal(
        action="BUY",
        conviction=0.7,
        entry_price=1234.0,
        stop_loss=1180.0,
        target_1=1310.0,
        target_2=1380.0,
        position_size_pct=0.03,
        position_size_shares=10,
        risk_reward_ratio=1.45,
        rationale="t",
        analyst_weight_applied={
            "technical": 0.303,
            "fundamentals": 0.455,
            "news": 0.242,
        },
        sizing_method="atr_based",
        time_horizon="positional",
        t2t_flag=False,
    )
    pd_ = PortfolioDecision(
        decision="approve",
        approved=True,
        pm_notes="ok",
        signal=ts,
    )
    state = AgentState(
        ticker="RELIANCE",
        date="2026-04-26",
        final_trade_decision=pd_,
    )

    fake_graph = MagicMock()
    fake_graph.apropagate = AsyncMock(return_value=state)
    monkeypatch.setattr(
        "finagents.cli.run_command.FinAgentsGraph",
        lambda **kw: fake_graph,
    )
    monkeypatch.setattr(
        "finagents.cli.run_command.signals_db_path",
        lambda settings=None: tmp_signals_db,
    )

    from finagents.observability.events import PipelineEvent

    async def _default_stream(*args, **kwargs):
        yield PipelineEvent(
            kind="node_status",
            node_name="technical_analyst_node",
            status="started",
        )
        yield PipelineEvent(
            kind="node_status",
            node_name="technical_analyst_node",
            status="completed",
        )
        yield PipelineEvent(kind="completed", final_state=state)

    monkeypatch.setattr(
        "finagents.cli.run_command.stream_pipeline_events",
        _default_stream,
    )
    return state
