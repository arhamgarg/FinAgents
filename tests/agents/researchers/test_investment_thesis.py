from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.runnables import RunnableConfig

from finagents.agents.utils.agent_states import InvestDebateState
from finagents.config import Settings


@pytest.fixture
def default_settings() -> Settings:
    return Settings(llm_provider="openrouter", openrouter_api_key="fake")


@pytest.fixture
def populated_state_with_full_history() -> InvestDebateState:
    return InvestDebateState(
        bull_history={1: "b1-bull-thesis", 2: "b2-bull-thesis"},
        bear_history={1: "br1-bear-counter", 2: "br2-bear-counter"},
        current_round=3,
        max_rounds=3,
        research_brief_snapshot="RELIANCE: mixed signals across four analysts.",
    )


@pytest.fixture
def mock_thesis():
    from finagents.agents.researchers.investment_thesis import InvestmentThesis

    return InvestmentThesis(
        synthesis=(
            "On balance, the bull case outweighs the bear on valuation + momentum; "
            "recommend measured accumulation."
        ),
        bull_strength="strong on FCF and sector tailwinds",
        bear_strength="valid on capex drag but already priced in",
    )


class TestInvestmentThesisNode:
    def test_returns_investment_thesis_key(
        self,
        populated_state_with_full_history: InvestDebateState,
        mock_thesis,
        default_settings: Settings,
    ) -> None:
        from finagents.agents.researchers.investment_thesis import investment_thesis_node

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_thesis)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.researchers.investment_thesis.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(investment_thesis_node(populated_state_with_full_history, config))
        assert "investment_thesis" in result
        assert isinstance(result["investment_thesis"], str)
        assert len(result["investment_thesis"]) > 0

    def test_prompt_includes_full_transcript(
        self,
        populated_state_with_full_history: InvestDebateState,
        mock_thesis,
        default_settings: Settings,
    ) -> None:
        from finagents.agents.researchers.investment_thesis import investment_thesis_node

        captured_prompts: list = []

        async def capture_ainvoke(prompt):
            captured_prompts.append(prompt)
            return mock_thesis

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = capture_ainvoke
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.researchers.investment_thesis.create_llm_client",
            return_value=mock_llm,
        ):
            asyncio.run(investment_thesis_node(populated_state_with_full_history, config))
        assert captured_prompts
        full_text = " ".join(getattr(m, "content", "") for m in captured_prompts[0])
        assert "b1-bull-thesis" in full_text
        assert "b2-bull-thesis" in full_text
        assert "br1-bear-counter" in full_text
        assert "br2-bear-counter" in full_text

    def test_retries_once_on_llm_failure(
        self,
        populated_state_with_full_history: InvestDebateState,
        mock_thesis,
        default_settings: Settings,
    ) -> None:
        from finagents.agents.researchers.investment_thesis import investment_thesis_node

        ainvoke_mock = AsyncMock(side_effect=[RuntimeError("LLM timeout"), mock_thesis])
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = ainvoke_mock
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.researchers.investment_thesis.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(investment_thesis_node(populated_state_with_full_history, config))
        assert ainvoke_mock.call_count == 2
        assert "investment_thesis" in result

    def test_returns_sentinel_when_both_attempts_fail(
        self,
        populated_state_with_full_history: InvestDebateState,
        default_settings: Settings,
    ) -> None:
        from finagents.agents.researchers.investment_thesis import investment_thesis_node

        ainvoke_mock = AsyncMock(side_effect=[RuntimeError("first"), RuntimeError("second")])
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = ainvoke_mock
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.researchers.investment_thesis.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(investment_thesis_node(populated_state_with_full_history, config))
        assert ainvoke_mock.call_count == 2
        assert "investment_thesis" in result
        sentinel = result["investment_thesis"]
        assert "investment thesis unavailable" in sentinel.lower()
        assert "second" in sentinel
        assert "first" in sentinel

    def test_prompt_contains_under_800_words_directive(
        self,
        populated_state_with_full_history: InvestDebateState,
        mock_thesis,
        default_settings: Settings,
    ) -> None:
        from finagents.agents.researchers.investment_thesis import investment_thesis_node

        captured_prompts: list = []

        async def capture_ainvoke(prompt):
            captured_prompts.append(prompt)
            return mock_thesis

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = capture_ainvoke
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        config: RunnableConfig = {"configurable": {"settings": default_settings}}
        with patch(
            "finagents.agents.researchers.investment_thesis.create_llm_client",
            return_value=mock_llm,
        ):
            asyncio.run(investment_thesis_node(populated_state_with_full_history, config))
        assert captured_prompts
        full_text = " ".join(getattr(m, "content", "") for m in captured_prompts[0])
        assert "under 800 words" in full_text


class TestInvestmentThesisSchema:
    def test_required_field_synthesis(self) -> None:
        import pydantic

        from finagents.agents.researchers.investment_thesis import InvestmentThesis

        with pytest.raises(pydantic.ValidationError):
            InvestmentThesis()  # ty: ignore[missing-argument]

    def test_optional_strength_fields_default_to_none(self) -> None:
        from finagents.agents.researchers.investment_thesis import InvestmentThesis

        thesis = InvestmentThesis(synthesis="the synthesis")
        assert thesis.bull_strength is None
        assert thesis.bear_strength is None
