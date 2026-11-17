from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finagents.agents.risk_mgmt.schemas import PersonaAssessment, RiskConsolidation
from finagents.agents.utils.agent_states import RiskDebateState
from finagents.config import Settings


def _build_mock_persona_llm(assessment: PersonaAssessment):
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=assessment)
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def _build_mock_consolidation_llm(synthesis: str):
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=RiskConsolidation(synthesis=synthesis))
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


class TestRiskDebateSubgraphCompiles:
    def test_build_returns_compiled_graph(self) -> None:
        from finagents.graph.subgraphs.risk_debate import build_risk_debate_subgraph

        subgraph = build_risk_debate_subgraph()
        node_names = set(subgraph.nodes.keys())
        assert "aggressive_risk_node" in node_names
        assert "neutral_risk_node" in node_names
        assert "conservative_risk_node" in node_names
        assert "risk_debate_consolidation_node" in node_names

    def test_builder_returns_graph_not_builder(self) -> None:
        from finagents.graph.subgraphs.risk_debate import build_risk_debate_subgraph

        subgraph = build_risk_debate_subgraph()
        assert hasattr(subgraph, "ainvoke")


class TestRiskDebateSubgraphRoundCount:
    def test_single_round_debate_exits_cleanly(self) -> None:
        from finagents.graph.subgraphs.risk_debate import build_risk_debate_subgraph

        persona_out = PersonaAssessment(
            main_argument="a",
            recommended_position_size_pct=0.05,
            confidence=0.7,
        )
        with (
            patch(
                "finagents.agents.risk_mgmt.aggressive.create_llm_client",
                return_value=_build_mock_persona_llm(persona_out),
            ),
            patch(
                "finagents.agents.risk_mgmt.neutral.create_llm_client",
                return_value=_build_mock_persona_llm(persona_out),
            ),
            patch(
                "finagents.agents.risk_mgmt.conservative.create_llm_client",
                return_value=_build_mock_persona_llm(persona_out),
            ),
            patch(
                "finagents.agents.risk_mgmt.consolidation.create_llm_client",
                return_value=_build_mock_consolidation_llm("r0"),
            ),
        ):
            subgraph = build_risk_debate_subgraph()
            initial = RiskDebateState(
                max_rounds=1,
                trader_signal_snapshot='{"action":"BUY","t2t_flag":false}',
            )
            result = asyncio.run(
                subgraph.ainvoke(initial, config={"configurable": {"settings": Settings()}})
            )
        hydrated = RiskDebateState.model_validate(result)
        assert len(hydrated.risk_turns()) == 1
        assert hydrated.current_round == 1

    def test_two_round_debate(self) -> None:
        from finagents.graph.subgraphs.risk_debate import build_risk_debate_subgraph

        persona_out = PersonaAssessment(
            main_argument="a",
            recommended_position_size_pct=0.05,
            confidence=0.7,
        )
        synth_idx = {"i": 0}

        def _fresh_consol_llm():
            async def _ainvoke(_):
                r = RiskConsolidation(synthesis=f"round-{synth_idx['i']}")
                synth_idx["i"] += 1
                return r

            structured = MagicMock()
            structured.ainvoke = _ainvoke
            llm = MagicMock()
            llm.with_structured_output.return_value = structured
            return llm

        with (
            patch(
                "finagents.agents.risk_mgmt.aggressive.create_llm_client",
                return_value=_build_mock_persona_llm(persona_out),
            ),
            patch(
                "finagents.agents.risk_mgmt.neutral.create_llm_client",
                return_value=_build_mock_persona_llm(persona_out),
            ),
            patch(
                "finagents.agents.risk_mgmt.conservative.create_llm_client",
                return_value=_build_mock_persona_llm(persona_out),
            ),
            patch(
                "finagents.agents.risk_mgmt.consolidation.create_llm_client",
                return_value=_fresh_consol_llm(),
            ),
        ):
            subgraph = build_risk_debate_subgraph()
            initial = RiskDebateState(
                max_rounds=2,
                trader_signal_snapshot='{"action":"BUY","t2t_flag":false}',
            )
            result = asyncio.run(
                subgraph.ainvoke(initial, config={"configurable": {"settings": Settings()}})
            )
        hydrated = RiskDebateState.model_validate(result)
        assert len(hydrated.risk_turns()) == 2
        assert hydrated.current_round == 2

    def test_synthesized_size_populated_on_exit(self) -> None:
        from finagents.graph.subgraphs.risk_debate import build_risk_debate_subgraph

        persona_out = PersonaAssessment(
            main_argument="a",
            recommended_position_size_pct=0.04,
            confidence=0.6,
        )
        with (
            patch(
                "finagents.agents.risk_mgmt.aggressive.create_llm_client",
                return_value=_build_mock_persona_llm(persona_out),
            ),
            patch(
                "finagents.agents.risk_mgmt.neutral.create_llm_client",
                return_value=_build_mock_persona_llm(persona_out),
            ),
            patch(
                "finagents.agents.risk_mgmt.conservative.create_llm_client",
                return_value=_build_mock_persona_llm(persona_out),
            ),
            patch(
                "finagents.agents.risk_mgmt.consolidation.create_llm_client",
                return_value=_build_mock_consolidation_llm("r0"),
            ),
        ):
            subgraph = build_risk_debate_subgraph()
            initial = RiskDebateState(
                max_rounds=1,
                trader_signal_snapshot='{"action":"BUY","t2t_flag":false}',
            )
            result = asyncio.run(
                subgraph.ainvoke(initial, config={"configurable": {"settings": Settings()}})
            )
        hydrated = RiskDebateState.model_validate(result)
        assert hydrated.synthesized_position_size_pct == pytest.approx(0.04)
