from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from financialagents.agents.utils.agent_states import InvestDebateState
from financialagents.config import Settings


@pytest.fixture
def default_settings() -> Settings:
    return Settings(_env_file=None, llm_provider="ollama", groq_api_key=None)


@pytest.fixture
def populated_state() -> InvestDebateState:
    return InvestDebateState(
        bull_history={1: "Bull argument round 1: earnings momentum."},
        bear_history={},
        current_round=1,
        max_rounds=3,
        research_brief_snapshot="Mixed outlook on RELIANCE: energy tailwinds vs capex drag.",
    )


@pytest.fixture
def mock_bear_argument():
    from financialagents.agents.researchers.bear_researcher import BearArgument
    return BearArgument(
        argument="Margin compression from energy capex; peer P/E discount widening.",
        key_points=["capex", "margin"],
    )


class TestBearResearcherNode:
    def test_returns_keyed_entry_AND_increments_current_round(
        self,
        populated_state: InvestDebateState,
        mock_bear_argument,
        default_settings: Settings,
    ) -> None:
        from financialagents.agents.researchers.bear_researcher import bear_researcher_node
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_bear_argument)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        config = {"configurable": {"settings": default_settings}}
        with patch(
            "financialagents.agents.researchers.bear_researcher.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(bear_researcher_node(populated_state, config))
        assert "bear_history" in result
        assert result["bear_history"] == {populated_state.current_round: mock_bear_argument.argument}
        assert "current_round" in result
        assert result["current_round"] == populated_state.current_round + 1

    def test_bear_key_matches_current_round_before_increment(
        self,
        populated_state: InvestDebateState,
        mock_bear_argument,
        default_settings: Settings,
    ) -> None:
        from financialagents.agents.researchers.bear_researcher import bear_researcher_node
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_bear_argument)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        config = {"configurable": {"settings": default_settings}}
        with patch(
            "financialagents.agents.researchers.bear_researcher.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(bear_researcher_node(populated_state, config))
        assert set(result["bear_history"].keys()) == {1}

    def test_max_output_tokens_backstop(
        self,
        populated_state: InvestDebateState,
        mock_bear_argument,
        default_settings: Settings,
    ) -> None:
        from financialagents.agents.researchers.bear_researcher import bear_researcher_node
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_bear_argument)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        config = {"configurable": {"settings": default_settings}}
        with patch(
            "financialagents.agents.researchers.bear_researcher.create_llm_client",
            return_value=mock_llm,
        ) as mock_create:
            asyncio.run(bear_researcher_node(populated_state, config))
        mock_create.assert_called_once_with("deep_think", default_settings, max_output_tokens=1024)

    def test_retries_once_on_llm_failure(
        self,
        populated_state: InvestDebateState,
        mock_bear_argument,
        default_settings: Settings,
    ) -> None:
        from financialagents.agents.researchers.bear_researcher import bear_researcher_node
        ainvoke_mock = AsyncMock(side_effect=[RuntimeError("LLM timeout"), mock_bear_argument])
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = ainvoke_mock
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        config = {"configurable": {"settings": default_settings}}
        with patch(
            "financialagents.agents.researchers.bear_researcher.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(bear_researcher_node(populated_state, config))
        assert ainvoke_mock.call_count == 2
        assert "bear_history" in result
        assert "current_round" in result

    def test_raises_when_both_attempts_fail(
        self,
        populated_state: InvestDebateState,
        default_settings: Settings,
    ) -> None:
        from financialagents.agents.researchers.bear_researcher import bear_researcher_node
        ainvoke_mock = AsyncMock(side_effect=[RuntimeError("first"), RuntimeError("second")])
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = ainvoke_mock
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        config = {"configurable": {"settings": default_settings}}
        with patch(
            "financialagents.agents.researchers.bear_researcher.create_llm_client",
            return_value=mock_llm,
        ):
            with pytest.raises(RuntimeError, match="second"):
                asyncio.run(bear_researcher_node(populated_state, config))

    def test_prompt_contains_under_800_words_directive(
        self,
        populated_state: InvestDebateState,
        mock_bear_argument,
        default_settings: Settings,
    ) -> None:
        from financialagents.agents.researchers.bear_researcher import bear_researcher_node
        captured_prompts: list = []

        async def capture_ainvoke(prompt):
            captured_prompts.append(prompt)
            return mock_bear_argument

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = capture_ainvoke
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        config = {"configurable": {"settings": default_settings}}
        with patch(
            "financialagents.agents.researchers.bear_researcher.create_llm_client",
            return_value=mock_llm,
        ):
            asyncio.run(bear_researcher_node(populated_state, config))
        assert captured_prompts
        msgs = captured_prompts[0]
        full_text = " ".join(getattr(m, "content", "") for m in msgs)
        assert "under 800 words" in full_text


class TestBearArgumentSchema:
    def test_required_field_argument(self) -> None:
        import pydantic
        from financialagents.agents.researchers.bear_researcher import BearArgument
        with pytest.raises(pydantic.ValidationError):
            BearArgument(key_points=[])

    def test_optional_key_points_defaults_to_empty_list(self) -> None:
        from financialagents.agents.researchers.bear_researcher import BearArgument
        arg = BearArgument(argument="thesis")
        assert arg.key_points == []
