from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.runnables import RunnableConfig

from finagents.config import Settings


@pytest.fixture
def default_settings() -> Settings:
    return Settings(llm_provider="openrouter", openrouter_api_key="fake")


class TestDebateRetryIdempotency:
    def test_bull_retry_at_same_round_produces_one_entry(
        self,
        default_settings: Settings,
    ) -> None:
        from finagents.agents.researchers.bull_researcher import (
            BullArgument,
            bull_researcher_node,
        )
        from finagents.agents.utils.agent_states import InvestDebateState
        from finagents.agents.utils.reducers import merge_by_key

        state = InvestDebateState(
            current_round=1,
            max_rounds=3,
            research_brief_snapshot="brief",
        )

        mock_llm_1 = MagicMock()
        s1 = MagicMock()
        s1.ainvoke = AsyncMock(return_value=BullArgument(argument="text-v1"))
        mock_llm_1.with_structured_output.return_value = s1

        mock_llm_2 = MagicMock()
        s2 = MagicMock()
        s2.ainvoke = AsyncMock(return_value=BullArgument(argument="text-v2"))
        mock_llm_2.with_structured_output.return_value = s2

        config: RunnableConfig = {"configurable": {"settings": default_settings}}

        with patch(
            "finagents.agents.researchers.bull_researcher.create_llm_client",
            return_value=mock_llm_1,
        ):
            first_return = asyncio.run(bull_researcher_node(state, config))
        with patch(
            "finagents.agents.researchers.bull_researcher.create_llm_client",
            return_value=mock_llm_2,
        ):
            second_return = asyncio.run(bull_researcher_node(state, config))

        merged: dict[int, str] = {}
        merged = merge_by_key(merged, first_return["bull_history"])
        merged = merge_by_key(merged, second_return["bull_history"])

        assert len(merged) == 1
        assert merged == {1: "text-v2"}

    def test_bear_retry_at_same_round_produces_one_entry(
        self,
        default_settings: Settings,
    ) -> None:
        from finagents.agents.researchers.bear_researcher import (
            BearArgument,
            bear_researcher_node,
        )
        from finagents.agents.utils.agent_states import InvestDebateState
        from finagents.agents.utils.reducers import merge_by_key

        state = InvestDebateState(
            current_round=1,
            max_rounds=3,
            research_brief_snapshot="brief",
            bull_history={1: "bull-arg-1"},
        )

        mock_llm_1 = MagicMock()
        s1 = MagicMock()
        s1.ainvoke = AsyncMock(return_value=BearArgument(argument="bear-v1"))
        mock_llm_1.with_structured_output.return_value = s1

        mock_llm_2 = MagicMock()
        s2 = MagicMock()
        s2.ainvoke = AsyncMock(return_value=BearArgument(argument="bear-v2"))
        mock_llm_2.with_structured_output.return_value = s2

        config: RunnableConfig = {"configurable": {"settings": default_settings}}

        with patch(
            "finagents.agents.researchers.bear_researcher.create_llm_client",
            return_value=mock_llm_1,
        ):
            first_return = asyncio.run(bear_researcher_node(state, config))
        with patch(
            "finagents.agents.researchers.bear_researcher.create_llm_client",
            return_value=mock_llm_2,
        ):
            second_return = asyncio.run(bear_researcher_node(state, config))

        merged: dict[int, str] = {}
        merged = merge_by_key(merged, first_return["bear_history"])
        merged = merge_by_key(merged, second_return["bear_history"])

        assert len(merged) == 1
        assert merged == {1: "bear-v2"}
