from __future__ import annotations

import pytest


@pytest.fixture
def empty_state():
    from financialagents.agents.utils.agent_states import AgentState
    return AgentState(ticker="RELIANCE", date="2024-01-15")


@pytest.fixture
def state_with_research_summary():
    from financialagents.agents.managers.schemas import ResearchBrief
    from financialagents.agents.utils.agent_states import AgentState
    brief = ResearchBrief(consensus_view="Bullish on RELIANCE due to energy sector tailwinds.")
    return AgentState(ticker="RELIANCE", date="2024-01-15", research_summary=brief)


class TestDebateInitNode:
    def test_seeds_investment_debate_state_with_max_rounds(self, state_with_research_summary):
        from financialagents.config import Settings
        from financialagents.graph.setup import debate_init_node
        settings = Settings(_env_file=None)
        settings.max_debate_rounds = 5
        config = {"configurable": {"settings": settings}}
        result = debate_init_node(state_with_research_summary, config)
        assert "investment_debate_state" in result
        ids = result["investment_debate_state"]
        assert ids.max_rounds == 5
        assert ids.current_round == 0
        assert ids.bull_history == {}
        assert ids.bear_history == {}

    def test_research_brief_snapshot_populated_from_consensus_view(self, state_with_research_summary):
        from financialagents.config import Settings
        from financialagents.graph.setup import debate_init_node
        config = {"configurable": {"settings": Settings(_env_file=None)}}
        result = debate_init_node(state_with_research_summary, config)
        assert "Bullish on RELIANCE" in result["investment_debate_state"].research_brief_snapshot

    def test_clamps_max_rounds_at_minimum_one(self, empty_state):
        from financialagents.config import Settings
        from financialagents.graph.setup import debate_init_node
        settings = Settings(_env_file=None)
        settings.max_debate_rounds = 0
        result = debate_init_node(empty_state, {"configurable": {"settings": settings}})
        assert result["investment_debate_state"].max_rounds == 1

    def test_clamps_negative_max_rounds(self, empty_state):
        from financialagents.config import Settings
        from financialagents.graph.setup import debate_init_node
        settings = Settings(_env_file=None)
        settings.max_debate_rounds = -5
        result = debate_init_node(empty_state, {"configurable": {"settings": settings}})
        assert result["investment_debate_state"].max_rounds == 1

    def test_none_research_summary_yields_empty_snapshot(self, empty_state):
        from financialagents.config import Settings
        from financialagents.graph.setup import debate_init_node
        config = {"configurable": {"settings": Settings(_env_file=None)}}
        result = debate_init_node(empty_state, config)
        assert result["investment_debate_state"].research_brief_snapshot == ""

    def test_none_config_uses_default_settings(self, empty_state):
        from financialagents.graph.setup import debate_init_node
        result = debate_init_node(empty_state, None)
        assert result["investment_debate_state"].max_rounds == 3

    def test_empty_config_dict_uses_default_settings(self, empty_state):
        from financialagents.graph.setup import debate_init_node
        result = debate_init_node(empty_state, {})
        assert result["investment_debate_state"].max_rounds == 3

    def test_empty_configurable_uses_default_settings(self, empty_state):
        from financialagents.graph.setup import debate_init_node
        result = debate_init_node(empty_state, {"configurable": {}})
        assert result["investment_debate_state"].max_rounds == 3


class TestBuildAnalysisGraphByteIdentical:

    def test_build_analysis_graph_still_compiles(self):
        from financialagents.graph.setup import build_analysis_graph
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
