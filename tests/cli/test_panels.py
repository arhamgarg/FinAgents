from __future__ import annotations

from rich.console import Console
from rich.panel import Panel


class TestRenderSignalCardNoneFinalTradeDecision:

    def test_render_signal_card_handles_none_final_trade_decision(self) -> None:
        from finagents.agents.utils.agent_states import AgentState
        from finagents.cli.panels import render_signal_card

        state = AgentState(ticker="TEST", date="2026-04-28")
        assert state.final_trade_decision is None

        panel = render_signal_card(state)
        assert isinstance(panel, Panel)

        c = Console(record=True, width=80)
        c.print(panel)
        out = c.export_text()

        assert "TEST" in out
        assert "2026-04-28" in out

        lowered = out.lower()
        assert "no" in lowered
        assert "decision" in lowered
