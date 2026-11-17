from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from finagents.agents.researchers.bear_researcher import BearArgument
from finagents.agents.researchers.bull_researcher import BullArgument
from finagents.agents.researchers.investment_thesis import InvestmentThesis
from finagents.agents.utils.agent_states import InvestDebateState
from finagents.config import Settings


def _make_mock_llm(coro_factory):
    structured = MagicMock()
    structured.ainvoke = coro_factory
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


class TestInvestDebateSubgraphCompiles:
    def test_build_returns_compiled_graph(self) -> None:
        from finagents.graph.subgraphs.invest_debate import (
            build_invest_debate_subgraph,
        )

        subgraph = build_invest_debate_subgraph()
        assert subgraph is not None

        node_names = set(subgraph.nodes.keys())
        assert "bull_researcher_node" in node_names
        assert "bear_researcher_node" in node_names
        assert "investment_thesis_node" in node_names


class TestInvestDebateSubgraphRoundCount:
    def test_three_round_debate_produces_three_entries_each_side(self) -> None:

        async def fake_bull_ainvoke(prompt):
            return BullArgument(argument="bull-arg-stub")

        async def fake_bear_ainvoke(prompt):
            return BearArgument(argument="bear-arg-stub")

        async def fake_plan_ainvoke(prompt):
            return InvestmentThesis(synthesis="synthesized plan")

        with (
            patch(
                "finagents.agents.researchers.bull_researcher.create_llm_client",
                return_value=_make_mock_llm(fake_bull_ainvoke),
            ),
            patch(
                "finagents.agents.researchers.bear_researcher.create_llm_client",
                return_value=_make_mock_llm(fake_bear_ainvoke),
            ),
            patch(
                "finagents.agents.researchers.investment_thesis.create_llm_client",
                return_value=_make_mock_llm(fake_plan_ainvoke),
            ),
        ):
            from finagents.graph.subgraphs.invest_debate import (
                build_invest_debate_subgraph,
            )

            subgraph = build_invest_debate_subgraph()
            initial = InvestDebateState(max_rounds=3, research_brief_snapshot="brief")
            result = asyncio.run(
                subgraph.ainvoke(initial, config={"configurable": {"settings": Settings()}})
            )

        rehydrated = InvestDebateState.model_validate(result)
        assert len(rehydrated.bull_turns()) == 3, (
            f"Off-by-one: expected 3 bull turns, got {len(rehydrated.bull_turns())}"
        )
        assert len(rehydrated.bear_turns()) == 3
        assert rehydrated.investment_thesis == "synthesized plan"
        assert rehydrated.current_round == 3

    def test_single_round_debate(self) -> None:

        async def fake_bull_ainvoke(prompt):
            return BullArgument(argument="single-bull")

        async def fake_bear_ainvoke(prompt):
            return BearArgument(argument="single-bear")

        async def fake_plan_ainvoke(prompt):
            return InvestmentThesis(synthesis="single-round synthesis")

        with (
            patch(
                "finagents.agents.researchers.bull_researcher.create_llm_client",
                return_value=_make_mock_llm(fake_bull_ainvoke),
            ),
            patch(
                "finagents.agents.researchers.bear_researcher.create_llm_client",
                return_value=_make_mock_llm(fake_bear_ainvoke),
            ),
            patch(
                "finagents.agents.researchers.investment_thesis.create_llm_client",
                return_value=_make_mock_llm(fake_plan_ainvoke),
            ),
        ):
            from finagents.graph.subgraphs.invest_debate import (
                build_invest_debate_subgraph,
            )

            subgraph = build_invest_debate_subgraph()
            initial = InvestDebateState(max_rounds=1, research_brief_snapshot="brief")
            result = asyncio.run(
                subgraph.ainvoke(initial, config={"configurable": {"settings": Settings()}})
            )

        rehydrated = InvestDebateState.model_validate(result)
        assert len(rehydrated.bull_turns()) == 1
        assert len(rehydrated.bear_turns()) == 1
        assert rehydrated.investment_thesis == "single-round synthesis"
        assert rehydrated.current_round == 1
