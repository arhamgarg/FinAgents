from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from financialagents.agents.risk_mgmt.aggressive import (
    AGGRESSIVE_SCENARIO,
    AGGRESSIVE_TEMPERATURE,
    aggressive_risk_node,
    build_aggressive_prompt,
)
from financialagents.agents.risk_mgmt.schemas import PersonaAssessment
from financialagents.agents.utils.agent_states import RiskDebateState
from financialagents.config import Settings


@pytest.fixture
def default_settings() -> Settings:
    return Settings(_env_file=None, llm_provider="ollama", groq_api_key=None)


@pytest.fixture
def fresh_risk_state() -> RiskDebateState:
    return RiskDebateState(
        risk_debate_history={},
        current_round=0,
        max_rounds=2,
        trader_signal_snapshot='{"action":"BUY","conviction":0.7,"t2t_flag":false}',
    )


def _build_mock_assessment() -> PersonaAssessment:
    return PersonaAssessment(
        main_argument="Size up — opportunity cost of HOLD is real.",
        counter_to_prior_persona="",
        recommended_position_size_pct=0.05,
        confidence=0.8,
    )


class TestAggressivePromptStructure:
    def test_system_message_contains_covid_2020_scenario(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        msgs = build_aggressive_prompt(fresh_risk_state)
        system = msgs[0].content
        assert isinstance(system, str)
        assert "COVID-2020" in system or "March 2020" in system
        assert "TCS" in system
        assert AGGRESSIVE_SCENARIO.split("\n")[0] in system

    def test_system_message_references_persona_assessment_and_counter(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        msgs = build_aggressive_prompt(fresh_risk_state)
        system = msgs[0].content
        assert isinstance(system, str)
        assert "PersonaAssessment" in system
        assert "counter_to_prior_persona" in system
        assert "recommended_position_size_pct" in system

    def test_human_message_embeds_trader_signal_snapshot(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        msgs = build_aggressive_prompt(fresh_risk_state)
        human = msgs[1].content
        assert isinstance(human, str)
        assert fresh_risk_state.trader_signal_snapshot in human

    def test_round_1_has_placeholder_for_prior_consolidation(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        msgs = build_aggressive_prompt(fresh_risk_state)
        human = msgs[1].content
        assert isinstance(human, str)
        assert "first round" in human.lower() or "no prior consolidation" in human.lower()

    def test_round_2_includes_prior_consolidation(self) -> None:
        state = RiskDebateState(
            risk_debate_history={0: "Round-1 consolidation text: weighted 3% size."},
            current_round=1,
            trader_signal_snapshot='{"action":"BUY"}',
        )
        msgs = build_aggressive_prompt(state)
        human = msgs[1].content
        assert isinstance(human, str)
        assert "Round-1 consolidation text: weighted 3% size." in human


class TestAggressiveNodeBehavior:
    def test_returns_aggressive_assessment_json(
        self, fresh_risk_state: RiskDebateState, default_settings: Settings
    ) -> None:
        mock_assessment = _build_mock_assessment()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_assessment)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        with patch(
            "financialagents.agents.risk_mgmt.aggressive.create_llm_client",
            return_value=mock_llm,
        ) as mock_create:
            config = {"configurable": {"settings": default_settings}}
            result = asyncio.run(aggressive_risk_node(fresh_risk_state, config))

        assert "aggressive_assessment" in result
        parsed = PersonaAssessment.model_validate_json(result["aggressive_assessment"])
        assert parsed.recommended_position_size_pct == 0.05
        mock_create.assert_called_once_with(
            "quick_think",
            default_settings,
            max_output_tokens=1024,
            temperature=AGGRESSIVE_TEMPERATURE,
        )
        mock_llm.with_structured_output.assert_called_once_with(PersonaAssessment)

    def test_temperature_is_0_9(self) -> None:
        assert AGGRESSIVE_TEMPERATURE == 0.9

    def test_retries_once_on_llm_failure(
        self, fresh_risk_state: RiskDebateState, default_settings: Settings
    ) -> None:
        mock_assessment = _build_mock_assessment()
        ainvoke_mock = AsyncMock(side_effect=[RuntimeError("flake"), mock_assessment])
        mock_structured = MagicMock()
        mock_structured.ainvoke = ainvoke_mock
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        with patch(
            "financialagents.agents.risk_mgmt.aggressive.create_llm_client",
            return_value=mock_llm,
        ):
            config = {"configurable": {"settings": default_settings}}
            result = asyncio.run(aggressive_risk_node(fresh_risk_state, config))

        assert ainvoke_mock.await_count == 2
        assert "aggressive_assessment" in result

    def test_double_failure_reraises(
        self, fresh_risk_state: RiskDebateState, default_settings: Settings
    ) -> None:
        ainvoke_mock = AsyncMock(
            side_effect=[RuntimeError("first"), RuntimeError("second")]
        )
        mock_structured = MagicMock()
        mock_structured.ainvoke = ainvoke_mock
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        with patch(
            "financialagents.agents.risk_mgmt.aggressive.create_llm_client",
            return_value=mock_llm,
        ):
            config = {"configurable": {"settings": default_settings}}
            with pytest.raises(RuntimeError, match="second"):
                asyncio.run(aggressive_risk_node(fresh_risk_state, config))

    def test_none_config_uses_default_settings(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        mock_assessment = _build_mock_assessment()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_assessment)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        with patch(
            "financialagents.agents.risk_mgmt.aggressive.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(aggressive_risk_node(fresh_risk_state, None))
        assert "aggressive_assessment" in result
