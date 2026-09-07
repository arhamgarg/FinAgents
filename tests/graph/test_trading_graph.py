from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from finagents.agents.utils.agent_states import AgentState, InvestDebateState
from finagents.config import Settings


@pytest.fixture
def default_settings() -> Settings:
    return Settings(openrouter_api_key="fake")


class TestFinAgentsGraphFullPipelineFlag:

    def test_default_uses_build_analysis_graph(self) -> None:
        from finagents.graph.trading_graph import FinAgentsGraph

        with patch("finagents.graph.trading_graph.build_analysis_graph") as mock_build:
            mock_build.return_value = MagicMock()
            FinAgentsGraph()
            mock_build.assert_called_once()

    def test_full_pipeline_false_uses_build_analysis_graph(self) -> None:
        from finagents.graph.trading_graph import FinAgentsGraph

        with patch("finagents.graph.trading_graph.build_analysis_graph") as mock_build:
            mock_build.return_value = MagicMock()
            FinAgentsGraph(full_pipeline=False)
            mock_build.assert_called_once()

    def test_full_pipeline_true_uses_build_full_pipeline(self) -> None:
        from finagents.graph.trading_graph import FinAgentsGraph

        with patch("finagents.graph.setup.build_full_pipeline") as mock_full:
            mock_full.return_value = MagicMock()
            FinAgentsGraph(full_pipeline=True)
            mock_full.assert_called_once()


class TestFinAgentsGraphThreadId:

    def test_thread_id_is_unique_per_propagate_call(self, default_settings: Settings) -> None:
        from finagents.graph.trading_graph import FinAgentsGraph

        captured_configs: list[dict] = []

        async def fake_ainvoke(state, config=None):
            captured_configs.append(config or {})
            return {"ticker": state.ticker, "date": state.date}

        mock_graph = MagicMock()
        mock_graph.ainvoke = fake_ainvoke

        with patch(
            "finagents.graph.trading_graph.build_analysis_graph",
            return_value=mock_graph,
        ):
            fa = FinAgentsGraph()
            fa.propagate("RELIANCE", "2024-01-15", default_settings)
            fa.propagate("RELIANCE", "2024-01-15", default_settings)

        assert len(captured_configs) == 2
        tid1 = captured_configs[0]["configurable"]["thread_id"]
        tid2 = captured_configs[1]["configurable"]["thread_id"]
        assert tid1 != tid2, f"P11 regression: same thread_id {tid1} used twice"
        assert tid1.startswith("RELIANCE_2024-01-15_")
        assert len(tid1.split("_")[-1]) == 8

    def test_settings_flow_through_configurable(self, default_settings: Settings) -> None:
        from finagents.graph.trading_graph import FinAgentsGraph

        captured: list[dict] = []

        async def fake_ainvoke(state, config=None):
            captured.append(config or {})
            return {}

        mock_graph = MagicMock()
        mock_graph.ainvoke = fake_ainvoke

        with patch(
            "finagents.graph.trading_graph.build_analysis_graph",
            return_value=mock_graph,
        ):
            fa = FinAgentsGraph()
            fa.propagate("TCS", "2024-01-15", default_settings)

        assert captured[0]["configurable"]["settings"] is default_settings


class TestFinAgentsGraphModelValidate:

    def test_returns_agent_state_from_dict_with_nested_debate(
        self, default_settings: Settings
    ) -> None:
        from finagents.graph.trading_graph import FinAgentsGraph

        async def fake_ainvoke(state, config=None):
            return {
                "ticker": state.ticker,
                "date": state.date,
                "investment_debate_state": {
                    "bull_history": {1: "bull-1"},
                    "bear_history": {1: "bear-1"},
                    "current_round": 1,
                    "max_rounds": 3,
                    "research_brief_snapshot": "brief",
                    "investment_thesis": "plan text",
                },
            }

        mock_graph = MagicMock()
        mock_graph.ainvoke = fake_ainvoke

        with patch(
            "finagents.graph.trading_graph.build_analysis_graph",
            return_value=mock_graph,
        ):
            fa = FinAgentsGraph()
            result_state = fa.propagate("RELIANCE", "2024-01-15", default_settings)

        assert isinstance(result_state, AgentState)
        assert result_state.investment_debate_state is not None
        ids = result_state.investment_debate_state
        assert isinstance(ids, InvestDebateState)
        assert ids.bull_history == {1: "bull-1"}
        assert ids.investment_thesis == "plan text"
        assert ids.bull_turns() == ["bull-1"]

    def test_returns_agent_state_when_already_typed(self, default_settings: Settings) -> None:
        from finagents.graph.trading_graph import FinAgentsGraph

        agent_state_in = AgentState(ticker="INFY", date="2024-01-15")

        async def fake_ainvoke(state, config=None):
            return agent_state_in

        mock_graph = MagicMock()
        mock_graph.ainvoke = fake_ainvoke

        with patch(
            "finagents.graph.trading_graph.build_analysis_graph",
            return_value=mock_graph,
        ):
            fa = FinAgentsGraph()
            result = fa.propagate("INFY", "2024-01-15", default_settings)

        assert result is agent_state_in
