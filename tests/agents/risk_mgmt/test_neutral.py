from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from financialagents.agents.risk_mgmt.neutral import (
    NEUTRAL_SCENARIO,
    NEUTRAL_TEMPERATURE,
    build_neutral_prompt,
    neutral_risk_node,
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
        trader_signal_snapshot='{"action":"BUY","conviction":0.6,"t2t_flag":false}',
        aggressive_assessment=(
            '{"main_argument":"size up","counter_to_prior_persona":"",'
            '"recommended_position_size_pct":0.05,"confidence":0.8}'
        ),
    )


def _build_mock_assessment() -> PersonaAssessment:
    return PersonaAssessment(
        main_argument="Median sizing — edge is real but not overwhelming.",
        counter_to_prior_persona="Aggressive overweights opportunity cost.",
        recommended_position_size_pct=0.03,
        confidence=0.6,
    )


class TestNeutralPromptStructure:
    def test_system_message_contains_post_covid_nifty_scenario(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        assert "NIFTY" in NEUTRAL_SCENARIO
        assert "2023" in NEUTRAL_SCENARIO or "post-COVID" in NEUTRAL_SCENARIO
        msgs = build_neutral_prompt(fresh_risk_state)
        system = msgs[0].content
        assert isinstance(system, str)
        assert NEUTRAL_SCENARIO.split("\n")[0] in system
        assert "positional delivery trades on NSE/BSE cash segment" in system

    def test_system_message_references_persona_assessment_and_counter(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        msgs = build_neutral_prompt(fresh_risk_state)
        system = msgs[0].content
        assert isinstance(system, str)
        assert "PersonaAssessment" in system
        assert "counter_to_prior_persona" in system
        assert "recommended_position_size_pct" in system

    def test_human_message_embeds_trader_signal_snapshot(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        msgs = build_neutral_prompt(fresh_risk_state)
        human = msgs[1].content
        assert isinstance(human, str)
        assert fresh_risk_state.trader_signal_snapshot in human

    def test_human_message_embeds_aggressive_assessment(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        msgs = build_neutral_prompt(fresh_risk_state)
        human = msgs[1].content
        assert isinstance(human, str)
        assert fresh_risk_state.aggressive_assessment is not None
        assert fresh_risk_state.aggressive_assessment in human

    def test_aggressive_missing_placeholder(self) -> None:
        state = RiskDebateState(
            risk_debate_history={},
            current_round=0,
            trader_signal_snapshot='{"action":"BUY"}',
            aggressive_assessment=None,
        )
        msgs = build_neutral_prompt(state)
        human = msgs[1].content
        assert isinstance(human, str)
        assert (
            "no aggressive assessment" in human.lower()
            or "(no aggressive assessment yet)" in human.lower()
        )

    def test_round_1_has_placeholder_for_prior_consolidation(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        msgs = build_neutral_prompt(fresh_risk_state)
        human = msgs[1].content
        assert isinstance(human, str)
        assert "first round" in human.lower() or "no prior consolidation" in human.lower()

    def test_round_2_includes_prior_consolidation(self) -> None:
        state = RiskDebateState(
            risk_debate_history={0: "Round-1 consolidation: weighted 3% size."},
            current_round=1,
            trader_signal_snapshot='{"action":"BUY"}',
            aggressive_assessment='{"main_argument":"x","recommended_position_size_pct":0.05,"confidence":0.7}',
        )
        msgs = build_neutral_prompt(state)
        human = msgs[1].content
        assert isinstance(human, str)
        assert "Round-1 consolidation: weighted 3% size." in human


class TestNeutralNodeBehavior:
    def test_returns_neutral_assessment_json(
        self, fresh_risk_state: RiskDebateState, default_settings: Settings
    ) -> None:
        mock_assessment = _build_mock_assessment()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_assessment)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        with patch(
            "financialagents.agents.risk_mgmt.neutral.create_llm_client",
            return_value=mock_llm,
        ) as mock_create:
            config = {"configurable": {"settings": default_settings}}
            result = asyncio.run(neutral_risk_node(fresh_risk_state, config))

        assert "neutral_assessment" in result
        parsed = PersonaAssessment.model_validate_json(result["neutral_assessment"])
        assert parsed.recommended_position_size_pct == 0.03
        mock_create.assert_called_once_with(
            "quick_think",
            default_settings,
            temperature=NEUTRAL_TEMPERATURE,
            call_site="neutral_risk",
        )
        mock_llm.with_structured_output.assert_called_once_with(PersonaAssessment)

    def test_temperature_is_0_6(self) -> None:
        assert NEUTRAL_TEMPERATURE == 0.6

    def test_create_llm_client_called_with_neutral_temperature(
        self, fresh_risk_state: RiskDebateState, default_settings: Settings
    ) -> None:
        mock_assessment = _build_mock_assessment()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_assessment)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        with patch(
            "financialagents.agents.risk_mgmt.neutral.create_llm_client",
            return_value=mock_llm,
        ) as mock_create:
            config = {"configurable": {"settings": default_settings}}
            asyncio.run(neutral_risk_node(fresh_risk_state, config))
        mock_create.assert_called_once_with(
            "quick_think",
            default_settings,
            temperature=0.6,
            call_site="neutral_risk",
        )

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
            "financialagents.agents.risk_mgmt.neutral.create_llm_client",
            return_value=mock_llm,
        ):
            config = {"configurable": {"settings": default_settings}}
            result = asyncio.run(neutral_risk_node(fresh_risk_state, config))

        assert ainvoke_mock.await_count == 2
        assert "neutral_assessment" in result

    def test_double_failure_returns_sentinel(
        self, fresh_risk_state: RiskDebateState, default_settings: Settings
    ) -> None:
        import json

        ainvoke_mock = AsyncMock(
            side_effect=[RuntimeError("first"), RuntimeError("second")]
        )
        mock_structured = MagicMock()
        mock_structured.ainvoke = ainvoke_mock
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        with patch(
            "financialagents.agents.risk_mgmt.neutral.create_llm_client",
            return_value=mock_llm,
        ):
            config = {"configurable": {"settings": default_settings}}
            result = asyncio.run(neutral_risk_node(fresh_risk_state, config))

        assert ainvoke_mock.await_count == 2
        assert "neutral_assessment" in result
        sentinel = json.loads(result["neutral_assessment"])
        assert sentinel["confidence"] == 0.0
        assert sentinel["recommended_position_size_pct"] == 0.0
        assert "neutral persona unavailable" in sentinel["main_argument"].lower()
        assert "second" in sentinel["main_argument"]
        assert "first" in sentinel["main_argument"]

    def test_none_config_uses_default_settings(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        mock_assessment = _build_mock_assessment()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_assessment)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        with patch(
            "financialagents.agents.risk_mgmt.neutral.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(neutral_risk_node(fresh_risk_state, None))
        assert "neutral_assessment" in result
