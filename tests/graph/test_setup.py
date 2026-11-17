from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig


@pytest.fixture
def empty_state():
    from finagents.agents.utils.agent_states import AgentState

    return AgentState(ticker="RELIANCE", date="2024-01-15")


@pytest.fixture
def state_with_research_summary():
    from finagents.agents.managers.schemas import ResearchBrief
    from finagents.agents.utils.agent_states import AgentState

    brief = ResearchBrief(consensus_view="Bullish on RELIANCE due to energy sector tailwinds.")
    return AgentState(ticker="RELIANCE", date="2024-01-15", research_summary=brief)


class TestDebateInitNode:
    def test_seeds_investment_debate_state_with_max_rounds(self, state_with_research_summary):
        from finagents.config import Settings
        from finagents.graph.setup import debate_init_node

        settings = Settings(openrouter_api_key="fake", max_debate_rounds=5)
        config: RunnableConfig = {"configurable": {"settings": settings}}
        result = debate_init_node(state_with_research_summary, config)
        assert "investment_debate_state" in result
        ids = result["investment_debate_state"]
        assert ids.max_rounds == 5
        assert ids.current_round == 0
        assert ids.bull_history == {}
        assert ids.bear_history == {}

    def test_research_brief_snapshot_populated_from_consensus_view(
        self, state_with_research_summary
    ):
        from finagents.config import Settings
        from finagents.graph.setup import debate_init_node

        config: RunnableConfig = {"configurable": {"settings": Settings(openrouter_api_key="fake")}}
        result = debate_init_node(state_with_research_summary, config)
        assert "Bullish on RELIANCE" in result["investment_debate_state"].research_brief_snapshot

    def test_clamps_max_rounds_at_minimum_one(self, empty_state):
        from finagents.config import Settings
        from finagents.graph.setup import debate_init_node

        settings = Settings(openrouter_api_key="fake", max_debate_rounds=0)
        result = debate_init_node(empty_state, {"configurable": {"settings": settings}})
        assert result["investment_debate_state"].max_rounds == 1

    def test_clamps_negative_max_rounds(self, empty_state):
        from finagents.config import Settings
        from finagents.graph.setup import debate_init_node

        settings = Settings(openrouter_api_key="fake", max_debate_rounds=-5)
        result = debate_init_node(empty_state, {"configurable": {"settings": settings}})
        assert result["investment_debate_state"].max_rounds == 1

    def test_none_research_summary_yields_empty_snapshot(self, empty_state):
        from finagents.config import Settings
        from finagents.graph.setup import debate_init_node

        config: RunnableConfig = {"configurable": {"settings": Settings(openrouter_api_key="fake")}}
        result = debate_init_node(empty_state, config)
        assert result["investment_debate_state"].research_brief_snapshot == ""

    def test_none_config_requires_settings(self, empty_state):
        from finagents.graph.setup import debate_init_node

        with pytest.raises(TypeError, match="settings"):
            debate_init_node(empty_state, None)

    def test_empty_config_dict_requires_settings(self, empty_state):
        from finagents.graph.setup import debate_init_node

        with pytest.raises(TypeError, match="settings"):
            debate_init_node(empty_state, {})

    def test_empty_configurable_requires_settings(self, empty_state):
        from finagents.graph.setup import debate_init_node

        with pytest.raises(TypeError, match="settings"):
            debate_init_node(empty_state, {"configurable": {}})


class TestBuildAnalysisGraphByteIdentical:
    def test_build_analysis_graph_still_compiles(self):
        from finagents.graph.setup import build_analysis_graph

        graph = build_analysis_graph()
        assert graph is not None
        nodes = set(graph.get_graph().nodes.keys())
        expected = {
            "__start__",
            "__end__",
            "propagation_node",
            "technical_analyst_node",
            "fundamentals_analyst_node",
            "news_analyst_node",
            "validation_node",
            "research_manager_node",
        }
        assert nodes == expected, (
            f"ROADMAP SC-3 regression: build_analysis_graph() node set drifted. "
            f"Expected {expected}, got {nodes}."
        )


class TestRiskInitNode:
    def test_seeds_risk_debate_state_with_max_rounds(self) -> None:
        from finagents.agents.utils.agent_states import AgentState
        from finagents.config import Settings
        from finagents.graph.setup import risk_init_node

        settings = Settings(max_risk_discuss_rounds=2, openrouter_api_key="fake")
        config: RunnableConfig = {"configurable": {"settings": settings}}
        state = AgentState(ticker="T", date="2026-04-21")
        result = risk_init_node(state, config)

        risk_state = result["risk_debate_state"]
        assert risk_state.max_rounds == 2
        assert risk_state.current_round == 0
        assert risk_state.risk_debate_history == {}

    def test_trader_signal_snapshot_populated_from_model_dump_json(self) -> None:
        from finagents.agents.trader.schemas import TradingSignal
        from finagents.agents.utils.agent_states import AgentState
        from finagents.config import Settings
        from finagents.graph.setup import risk_init_node

        sig = TradingSignal(action="BUY", conviction=0.7, entry_price=100.0)
        state = AgentState(ticker="T", date="2026-04-21", trader_signal=sig)
        config: RunnableConfig = {"configurable": {"settings": Settings(openrouter_api_key="fake")}}
        result = risk_init_node(state, config)
        assert '"action":"BUY"' in result["risk_debate_state"].trader_signal_snapshot

    def test_clamps_max_rounds_at_minimum_one(self) -> None:
        from finagents.agents.utils.agent_states import AgentState
        from finagents.config import Settings
        from finagents.graph.setup import risk_init_node

        settings = Settings(max_risk_discuss_rounds=0, openrouter_api_key="fake")
        config: RunnableConfig = {"configurable": {"settings": settings}}
        result = risk_init_node(AgentState(), config)
        assert result["risk_debate_state"].max_rounds == 1

    def test_clamps_negative_max_rounds(self) -> None:
        from finagents.agents.utils.agent_states import AgentState
        from finagents.config import Settings
        from finagents.graph.setup import risk_init_node

        settings = Settings(max_risk_discuss_rounds=-5, openrouter_api_key="fake")
        result = risk_init_node(AgentState(), {"configurable": {"settings": settings}})
        assert result["risk_debate_state"].max_rounds == 1

    def test_none_trader_signal_yields_empty_snapshot(self) -> None:
        from finagents.agents.utils.agent_states import AgentState
        from finagents.config import Settings
        from finagents.graph.setup import risk_init_node

        state = AgentState(ticker="T", date="2026-04-21", trader_signal=None)
        result = risk_init_node(
            state,
            {"configurable": {"settings": Settings(openrouter_api_key="fake")}},
        )
        assert result["risk_debate_state"].trader_signal_snapshot == ""

    def test_none_config_requires_settings(self) -> None:
        from finagents.agents.utils.agent_states import AgentState
        from finagents.graph.setup import risk_init_node

        with pytest.raises(TypeError, match="settings"):
            risk_init_node(AgentState(), None)


class TestCallRiskDebateSubgraphRecursionLimit:
    def test_applies_recursion_limit_formula(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock, patch

        import finagents.graph.setup as setup_mod
        from finagents.agents.utils.agent_states import AgentState, RiskDebateState

        setup_mod._reset_risk_debate_subgraph_cache()

        captured = {}

        class _FakeCompiledSubgraph:
            def __init__(self):
                async def _ainvoke(state, config=None):
                    captured["config"] = config
                    return state.model_dump()

                self.ainvoke = AsyncMock(side_effect=_ainvoke)

        fake_subgraph = _FakeCompiledSubgraph()

        def _fake_build():
            return fake_subgraph

        with patch(
            "finagents.graph.subgraphs.risk_debate.build_risk_debate_subgraph",
            _fake_build,
        ):
            setup_mod._reset_risk_debate_subgraph_cache()
            state = AgentState(risk_debate_state=RiskDebateState(max_rounds=2))
            asyncio.run(setup_mod.call_risk_debate_subgraph(state, None))

        assert captured["config"]["recursion_limit"] == 2 * 3 + 10
        setup_mod._reset_risk_debate_subgraph_cache()

    def test_assertion_when_risk_debate_state_is_none(self) -> None:
        import asyncio

        import finagents.graph.setup as setup_mod
        from finagents.agents.utils.agent_states import AgentState

        setup_mod._reset_risk_debate_subgraph_cache()
        state = AgentState(risk_debate_state=None)
        with pytest.raises(AssertionError, match="risk_init_node must run"):
            asyncio.run(setup_mod.call_risk_debate_subgraph(state, None))


class TestBuildFullPipelineTopology:
    def test_pipeline_registers_debate_and_risk_nodes(self) -> None:
        from finagents.graph.setup import build_full_pipeline

        graph = build_full_pipeline()
        node_names = set(graph.nodes.keys())
        assert "risk_init_node" in node_names
        assert "risk_debate_node" in node_names
        assert "portfolio_manager_node" in node_names

    def test_pipeline_has_trader_to_risk_init_edge(self) -> None:
        from finagents.graph.setup import build_full_pipeline

        graph = build_full_pipeline()
        edges = [(e.source, e.target) for e in graph.get_graph().edges]
        assert ("trader_node", "risk_init_node") in edges
        assert ("risk_init_node", "risk_debate_node") in edges
        assert ("risk_debate_node", "portfolio_manager_node") in edges

    def test_pipeline_ends_at_portfolio_manager_to_end(self) -> None:
        from finagents.graph.setup import build_full_pipeline

        graph = build_full_pipeline()
        edges = [(e.source, e.target) for e in graph.get_graph().edges]
        assert any(src == "portfolio_manager_node" and tgt in ("__end__",) for src, tgt in edges)

    def test_build_analysis_graph_byte_identical_3_analyst(self) -> None:
        from finagents.graph.setup import build_analysis_graph

        graph = build_analysis_graph()
        node_names = set(graph.nodes.keys())
        assert "technical_analyst_node" in node_names
        assert "fundamentals_analyst_node" in node_names
        assert "news_analyst_node" in node_names
        assert "sentiment_analyst_node" not in node_names
        assert "risk_init_node" not in node_names
        assert "portfolio_manager_node" not in node_names
