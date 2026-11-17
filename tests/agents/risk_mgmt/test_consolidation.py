from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.runnables import RunnableConfig

from finagents.agents.risk_mgmt.consolidation import (
    _parse_size,
    build_consolidation_prompt,
    risk_debate_consolidation_node,
)
from finagents.agents.risk_mgmt.schemas import RiskConsolidation
from finagents.agents.utils.agent_states import RiskDebateState
from finagents.config import Settings


@pytest.fixture
def default_settings() -> Settings:
    return Settings(llm_provider="openrouter", openrouter_api_key="fake")


def _state_with_persona_sizes(
    agg: float = 0.05, neu: float = 0.03, con: float = 0.01, current_round: int = 0
) -> RiskDebateState:
    return RiskDebateState(
        aggressive_assessment=(
            f'{{"main_argument":"agg","counter_to_prior_persona":"",'
            f'"recommended_position_size_pct":{agg},"confidence":0.7}}'
        ),
        neutral_assessment=(
            f'{{"main_argument":"neu","counter_to_prior_persona":"",'
            f'"recommended_position_size_pct":{neu},"confidence":0.6}}'
        ),
        conservative_assessment=(
            f'{{"main_argument":"con","counter_to_prior_persona":"",'
            f'"recommended_position_size_pct":{con},"confidence":0.5}}'
        ),
        current_round=current_round,
        max_rounds=2,
        trader_signal_snapshot='{"action":"BUY","conviction":0.7,"t2t_flag":false}',
    )


def _mock_consolidation_llm(synthesis_text: str = "weighted synthesis"):
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=RiskConsolidation(synthesis=synthesis_text))
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


class TestParseSize:
    def test_returns_zero_for_none(self) -> None:
        assert _parse_size(None) == 0.0

    def test_returns_zero_for_empty_string(self) -> None:
        assert _parse_size("") == 0.0

    def test_returns_zero_for_malformed_json(self) -> None:
        assert _parse_size("not-json") == 0.0

    def test_returns_zero_for_missing_field(self) -> None:
        assert _parse_size('{"main_argument":"a"}') == 0.0

    def test_returns_value_for_valid_input(self) -> None:
        assert _parse_size(
            '{"recommended_position_size_pct":0.042,"confidence":0.7}'
        ) == pytest.approx(0.042)

    def test_returns_zero_for_bare_string_json(self) -> None:
        assert _parse_size('"a-bare-string"') == 0.0

    def test_returns_zero_for_list_json(self) -> None:
        assert _parse_size("[1, 2, 3]") == 0.0

    def test_returns_zero_for_null_json(self) -> None:
        assert _parse_size("null") == 0.0

    def test_returns_zero_for_number_json(self) -> None:
        assert _parse_size("3.14") == 0.0

    def test_returns_zero_for_field_value_null(self) -> None:
        assert _parse_size('{"recommended_position_size_pct":null,"confidence":0.7}') == 0.0


class TestWeightedSizeSynthesis:
    def test_formula_default_weights(self, default_settings: Settings) -> None:
        state = _state_with_persona_sizes(agg=0.05, neu=0.03, con=0.01)
        mock_llm = _mock_consolidation_llm()
        with patch(
            "finagents.agents.risk_mgmt.consolidation.create_llm_client",
            return_value=mock_llm,
        ):
            config: RunnableConfig = {"configurable": {"settings": default_settings}}
            result = asyncio.run(risk_debate_consolidation_node(state, config))
        expected = 0.25 * 0.05 + 0.50 * 0.03 + 0.25 * 0.01
        assert result["synthesized_position_size_pct"] == pytest.approx(expected)

    def test_all_zero_yields_zero_synthesis(self, default_settings: Settings) -> None:
        state = _state_with_persona_sizes(agg=0.0, neu=0.0, con=0.0)
        mock_llm = _mock_consolidation_llm("all reject")
        with patch(
            "finagents.agents.risk_mgmt.consolidation.create_llm_client",
            return_value=mock_llm,
        ):
            config: RunnableConfig = {"configurable": {"settings": default_settings}}
            result = asyncio.run(risk_debate_consolidation_node(state, config))
        assert result["synthesized_position_size_pct"] == 0.0

    def test_missing_aggressive_defaults_to_zero(self, default_settings: Settings) -> None:
        state = _state_with_persona_sizes(neu=0.03, con=0.01)
        state.aggressive_assessment = None
        mock_llm = _mock_consolidation_llm()
        with patch(
            "finagents.agents.risk_mgmt.consolidation.create_llm_client",
            return_value=mock_llm,
        ):
            config: RunnableConfig = {"configurable": {"settings": default_settings}}
            result = asyncio.run(risk_debate_consolidation_node(state, config))
        expected = 0.25 * 0.0 + 0.50 * 0.03 + 0.25 * 0.01
        assert result["synthesized_position_size_pct"] == pytest.approx(expected)

    def test_malformed_neutral_defaults_to_zero(self, default_settings: Settings) -> None:
        state = _state_with_persona_sizes(agg=0.05, con=0.01)
        state.neutral_assessment = "not-json"
        mock_llm = _mock_consolidation_llm()
        with patch(
            "finagents.agents.risk_mgmt.consolidation.create_llm_client",
            return_value=mock_llm,
        ):
            config: RunnableConfig = {"configurable": {"settings": default_settings}}
            result = asyncio.run(risk_debate_consolidation_node(state, config))
        expected = 0.25 * 0.05 + 0.50 * 0.0 + 0.25 * 0.01
        assert result["synthesized_position_size_pct"] == pytest.approx(expected)

    def test_non_default_weights_override(self) -> None:
        custom = Settings(
            llm_provider="openrouter",
            openrouter_api_key="fake",
            risk_persona_weights={"aggressive": 0.5, "neutral": 0.3, "conservative": 0.2},
        )
        state = _state_with_persona_sizes(agg=0.05, neu=0.03, con=0.01)
        mock_llm = _mock_consolidation_llm()
        with patch(
            "finagents.agents.risk_mgmt.consolidation.create_llm_client",
            return_value=mock_llm,
        ):
            config: RunnableConfig = {"configurable": {"settings": custom}}
            result = asyncio.run(risk_debate_consolidation_node(state, config))
        expected = 0.5 * 0.05 + 0.3 * 0.03 + 0.2 * 0.01
        assert result["synthesized_position_size_pct"] == pytest.approx(expected)


class TestDictKeyedHistoryMigration:
    def test_round_key_is_pre_increment_current_round(self, default_settings: Settings) -> None:
        state = _state_with_persona_sizes(current_round=1)
        mock_llm = _mock_consolidation_llm("r1-synth")
        with patch(
            "finagents.agents.risk_mgmt.consolidation.create_llm_client",
            return_value=mock_llm,
        ):
            config: RunnableConfig = {"configurable": {"settings": default_settings}}
            result = asyncio.run(risk_debate_consolidation_node(state, config))
        assert set(result["risk_debate_history"].keys()) == {1}
        assert result["risk_debate_history"][1] == "r1-synth"
        assert result["current_round"] == 2

    def test_first_round_lands_at_key_zero(self, default_settings: Settings) -> None:
        state = _state_with_persona_sizes(current_round=0)
        mock_llm = _mock_consolidation_llm("r0-synth")
        with patch(
            "finagents.agents.risk_mgmt.consolidation.create_llm_client",
            return_value=mock_llm,
        ):
            config: RunnableConfig = {"configurable": {"settings": default_settings}}
            result = asyncio.run(risk_debate_consolidation_node(state, config))
        assert result["risk_debate_history"] == {0: "r0-synth"}
        assert result["current_round"] == 1

    def test_returns_exactly_three_keys(self, default_settings: Settings) -> None:
        state = _state_with_persona_sizes()
        mock_llm = _mock_consolidation_llm()
        with patch(
            "finagents.agents.risk_mgmt.consolidation.create_llm_client",
            return_value=mock_llm,
        ):
            config: RunnableConfig = {"configurable": {"settings": default_settings}}
            result = asyncio.run(risk_debate_consolidation_node(state, config))
        assert set(result.keys()) == {
            "risk_debate_history",
            "current_round",
            "synthesized_position_size_pct",
        }


class TestConsolidationRetryOnce:
    def test_retries_once_on_llm_failure(self, default_settings: Settings) -> None:
        state = _state_with_persona_sizes()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(
            side_effect=[RuntimeError("flake"), RiskConsolidation(synthesis="ok")]
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        with patch(
            "finagents.agents.risk_mgmt.consolidation.create_llm_client",
            return_value=mock_llm,
        ):
            config: RunnableConfig = {"configurable": {"settings": default_settings}}
            result = asyncio.run(risk_debate_consolidation_node(state, config))
        assert mock_structured.ainvoke.await_count == 2
        assert result["risk_debate_history"][0] == "ok"

    def test_double_failure_returns_sentinel(self, default_settings: Settings) -> None:
        state = _state_with_persona_sizes()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(
            side_effect=[RuntimeError("first"), RuntimeError("second")]
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        with patch(
            "finagents.agents.risk_mgmt.consolidation.create_llm_client",
            return_value=mock_llm,
        ):
            config: RunnableConfig = {"configurable": {"settings": default_settings}}
            result = asyncio.run(risk_debate_consolidation_node(state, config))

        assert mock_structured.ainvoke.await_count == 2
        assert "risk_debate_history" in result
        round_key = state.current_round
        sentinel = result["risk_debate_history"][round_key]
        assert "consolidation unavailable" in sentinel.lower()
        assert "second" in sentinel
        assert "first" in sentinel
        assert result["current_round"] == state.current_round + 1
        assert "synthesized_position_size_pct" in result
        assert isinstance(result["synthesized_position_size_pct"], float)


class TestConsolidationPrompt:
    def test_embeds_all_three_persona_assessments(self) -> None:
        state = _state_with_persona_sizes()
        msgs = build_consolidation_prompt(state)
        human = msgs[1].text
        assert isinstance(human, str)
        assert state.aggressive_assessment is not None
        assert state.neutral_assessment is not None
        assert state.conservative_assessment is not None
        assert state.aggressive_assessment in human
        assert state.neutral_assessment in human
        assert state.conservative_assessment in human

    def test_embeds_trader_signal_snapshot(self) -> None:
        state = _state_with_persona_sizes()
        msgs = build_consolidation_prompt(state)
        human = msgs[1].text
        assert state.trader_signal_snapshot in human

    def test_round_2_includes_prior_consolidation(self) -> None:
        state = _state_with_persona_sizes(current_round=1)
        state.risk_debate_history = {0: "Round-1 synthesis text."}
        msgs = build_consolidation_prompt(state)
        human = msgs[1].text
        assert "Round-1 synthesis text." in human

    def test_round_1_has_placeholder(self) -> None:
        state = _state_with_persona_sizes(current_round=0)
        msgs = build_consolidation_prompt(state)
        human = msgs[1].text
        assert "first round" in human.lower() or "no prior consolidation" in human.lower()
