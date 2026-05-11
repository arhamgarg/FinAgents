from __future__ import annotations

import asyncio
import io
import re
from typing import Any, AsyncIterator

import pytest
from rich.console import Console


def _ev(**kwargs: Any):
    from financialagents.observability.events import PipelineEvent

    defaults = dict(
        kind="node_status",
        node_name=None,
        status=None,
        update_payload=None,
        final_state=None,
        error=None,
        round=None,
        max_rounds=None,
        position_size_pct=None,
    )
    defaults.update(kwargs)
    return PipelineEvent(**defaults)


async def _stream(events: list) -> AsyncIterator:
    for ev in events:
        yield ev


class TestCLILiveRender:

    @pytest.mark.asyncio
    async def test_live_snapshot(self) -> None:
        from financialagents.cli.live import run_with_live_render

        console = Console(record=True, width=120, force_terminal=True)
        events = [
            _ev(
                kind="node_status",
                node_name="technical_analyst_node",
                status="started",
            ),
            _ev(
                kind="node_status",
                node_name="technical_analyst_node",
                status="completed",
            ),
            _ev(kind="completed"),
        ]

        await run_with_live_render(
            _stream(events),
            ticker="RELIANCE",
            date="2026-04-26",
            provider="ollama",
            model="llama3.1:70b",
            console=console,
        )

        rendered = console.export_text()
        assert "Analysts" in rendered
        assert "Research Manager" in rendered
        assert "Investment Debate" in rendered
        assert "Trader" in rendered
        assert "Risk Debate + PM" in rendered
        assert "technical_analyst_node" in rendered
        assert "✓" in rendered

    @pytest.mark.asyncio
    async def test_non_tty_log_lines(self) -> None:
        from financialagents.cli.live import run_with_live_render

        buf = io.StringIO()
        console = Console(file=buf, width=120)
        events = [
            _ev(
                kind="node_status",
                node_name="technical_analyst_node",
                status="started",
            ),
            _ev(
                kind="node_status",
                node_name="technical_analyst_node",
                status="completed",
            ),
            _ev(kind="completed"),
        ]

        await run_with_live_render(
            _stream(events),
            ticker="RELIANCE",
            date="2026-04-26",
            provider="ollama",
            model="llama3.1:70b",
            console=console,
        )

        out = buf.getvalue()
        assert re.search(
            r"\[\d{2}:\d{2}:\d{2}\] technical_analyst_node started", out
        ), f"missing started log line: {out!r}"
        assert re.search(
            r"\[\d{2}:\d{2}:\d{2}\] technical_analyst_node done", out
        ), f"missing done log line: {out!r}"
        assert "\x1b[" not in out

    @pytest.mark.asyncio
    async def test_header_format(self) -> None:
        from financialagents.cli.live import run_with_live_render

        console = Console(record=True, width=120, force_terminal=True)
        events = [_ev(kind="completed")]
        await run_with_live_render(
            _stream(events),
            ticker="RELIANCE",
            date="2026-04-26",
            provider="ollama",
            model="llama3.1:70b",
            console=console,
        )
        rendered = console.export_text()
        first_line = next(
            (line.strip() for line in rendered.splitlines() if line.strip()),
            "",
        )
        assert "RELIANCE" in first_line
        assert "2026-04-26" in first_line
        assert "ollama:llama3.1:70b" in first_line
        assert "·" in first_line

    @pytest.mark.asyncio
    async def test_narrow_terminal(self) -> None:
        from financialagents.cli.live import (
            _build_header,
            _initial_stage_state,
            make_stage_table,
        )

        console = Console(record=True, width=60, force_terminal=True)
        stage_state = _initial_stage_state(3)
        stage_state["Analysts"]["started_at"] = 0.0
        stage_state["Analysts"]["children"]["technical_analyst_node"] = {
            "status": "completed",
            "started_at": 0.0,
            "completed_at": 0.5,
            "size_pct": None,
        }
        header = _build_header(
            "RELIANCE", "2026-04-26", "ollama", "llama3.1:70b", 0.5
        )

        console.print(make_stage_table(stage_state, header, 0.5))
        rendered = console.export_text()

        max_line = max(
            (len(line) for line in rendered.splitlines() if line),
            default=0,
        )
        assert max_line <= 60, (
            f"line exceeded narrow-terminal width: {max_line} > 60"
        )

    @pytest.mark.asyncio
    async def test_round_counter_in_label(self) -> None:
        from financialagents.cli.live import run_with_live_render

        console = Console(record=True, width=120, force_terminal=True)
        events = [
            _ev(
                kind="node_status",
                node_name="bull_researcher_node",
                status="started",
            ),
            _ev(
                kind="round_advanced",
                node_name="bull_researcher_node",
                round=2,
            ),
            _ev(kind="completed"),
        ]
        await run_with_live_render(
            _stream(events),
            ticker="RELIANCE",
            date="2026-04-26",
            provider="ollama",
            model="llama3.1:70b",
            max_debate_rounds=3,
            console=console,
        )
        rendered = console.export_text()
        assert "round 2/3" in rendered, (
            f"missing round counter in render: {rendered!r}"
        )

    @pytest.mark.asyncio
    async def test_persona_size_pct_in_row(self) -> None:
        from financialagents.cli.live import run_with_live_render

        console = Console(record=True, width=120, force_terminal=True)
        events = [
            _ev(
                kind="node_status",
                node_name="aggressive_risk_node",
                status="started",
            ),
            _ev(
                kind="node_status",
                node_name="aggressive_risk_node",
                status="completed",
                position_size_pct=4.5,
            ),
            _ev(kind="completed"),
        ]
        await run_with_live_render(
            _stream(events),
            ticker="RELIANCE",
            date="2026-04-26",
            provider="ollama",
            model="llama3.1:70b",
            console=console,
        )
        rendered = console.export_text()
        assert "4.5%" in rendered, f"persona size_pct missing: {rendered!r}"

    @pytest.mark.asyncio
    async def test_last_values_snapshot_tracking(self) -> None:
        from financialagents.agents.utils.agent_states import AgentState
        from financialagents.cli.live import run_with_live_render

        partial = AgentState(
            ticker="RELIANCE",
            date="2026-04-26",
            trader_investment_thesis="partial technical analysis",
        )
        cancel_flag = asyncio.Event()

        async def mock_events():
            yield _ev(kind="values", final_state=partial)
            cancel_flag.set()
            yield _ev(
                kind="node_status",
                node_name="technical_analyst_node",
                status="completed",
            )

        result = await run_with_live_render(
            mock_events(),
            ticker="RELIANCE",
            date="2026-04-26",
            provider="ollama",
            model="llama3.1:70b",
            console=Console(file=io.StringIO()),
            cancel_flag=cancel_flag,
        )
        assert result is not None, "expected rolling values snapshot, got None"
        assert (
            getattr(result, "trader_investment_thesis", None)
            == "partial technical analysis"
        )
