from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from financialagents.agents.risk_mgmt.conservative import (
    CONSERVATIVE_SCENARIO,
    CONSERVATIVE_TEMPERATURE,
    _enforce_guard_04,
    build_conservative_prompt,
    conservative_risk_node,
)
from financialagents.agents.risk_mgmt.schemas import PersonaAssessment
from financialagents.agents.utils.agent_states import RiskDebateState
from financialagents.config import Settings


@pytest.fixture
def default_settings() -> Settings:
    return Settings(_env_file=None, llm_provider="openrouter", openrouter_api_key="fake")


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
        neutral_assessment=(
            '{"main_argument":"median edge","counter_to_prior_persona":"aggressive overreach",'
            '"recommended_position_size_pct":0.03,"confidence":0.6}'
        ),
    )


def _build_mock_assessment() -> PersonaAssessment:
    return PersonaAssessment(
        main_argument="Preserve optionality — small size with tight stops.",
        counter_to_prior_persona="Neutral underweights tail risk.",
        recommended_position_size_pct=0.02,
        confidence=0.4,
    )


class TestConservativePromptStructure:
    def test_system_message_contains_yes_bank_2020_scenario(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        assert "Yes Bank" in CONSERVATIVE_SCENARIO
        assert "2020" in CONSERVATIVE_SCENARIO
        msgs = build_conservative_prompt(fresh_risk_state)
        system = msgs[0].content
        assert isinstance(system, str)
        assert CONSERVATIVE_SCENARIO.split("\n")[0] in system
        assert "positional delivery trades on NSE/BSE cash segment" in system

    def test_system_message_references_persona_assessment_and_counter(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        msgs = build_conservative_prompt(fresh_risk_state)
        system = msgs[0].content
        assert isinstance(system, str)
        assert "PersonaAssessment" in system
        assert "counter_to_prior_persona" in system
        assert "recommended_position_size_pct" in system

    def test_human_message_embeds_trader_signal_snapshot(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        msgs = build_conservative_prompt(fresh_risk_state)
        human = msgs[1].content
        assert isinstance(human, str)
        assert fresh_risk_state.trader_signal_snapshot in human

    def test_human_message_embeds_aggressive_and_neutral_assessments(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        msgs = build_conservative_prompt(fresh_risk_state)
        human = msgs[1].content
        assert isinstance(human, str)
        assert fresh_risk_state.aggressive_assessment is not None
        assert fresh_risk_state.neutral_assessment is not None
        assert fresh_risk_state.aggressive_assessment in human
        assert fresh_risk_state.neutral_assessment in human

    def test_round_1_has_placeholder_for_prior_consolidation(
        self, fresh_risk_state: RiskDebateState
    ) -> None:
        msgs = build_conservative_prompt(fresh_risk_state)
        human = msgs[1].content
        assert isinstance(human, str)
        assert "first round" in human.lower() or "no prior consolidation" in human.lower()

    def test_round_2_includes_prior_consolidation(self) -> None:
        state = RiskDebateState(
            risk_debate_history={0: "Round-1 consolidation: weighted 3% size."},
            current_round=1,
            trader_signal_snapshot='{"action":"BUY"}',
            aggressive_assessment='{"main_argument":"a","recommended_position_size_pct":0.05,"confidence":0.8}',
            neutral_assessment='{"main_argument":"n","recommended_position_size_pct":0.03,"confidence":0.6}',
        )
        msgs = build_conservative_prompt(state)
        human = msgs[1].content
        assert isinstance(human, str)
        assert "Round-1 consolidation: weighted 3% size." in human


class TestConservativeNodeBehavior:
    def test_returns_conservative_assessment_json(
        self, fresh_risk_state: RiskDebateState, default_settings: Settings
    ) -> None:
        mock_assessment = _build_mock_assessment()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_assessment)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        with patch(
            "financialagents.agents.risk_mgmt.conservative.create_llm_client",
            return_value=mock_llm,
        ) as mock_create:
            config = {"configurable": {"settings": default_settings}}
            result = asyncio.run(conservative_risk_node(fresh_risk_state, config))

        assert "conservative_assessment" in result
        parsed = PersonaAssessment.model_validate_json(result["conservative_assessment"])
        assert parsed.recommended_position_size_pct == 0.02
        mock_create.assert_called_once_with(
            "quick_think",
            default_settings,
            temperature=CONSERVATIVE_TEMPERATURE,
            call_site="conservative_risk",
        )
        mock_llm.with_structured_output.assert_called_once_with(PersonaAssessment)

    def test_temperature_is_0_3(self) -> None:
        assert CONSERVATIVE_TEMPERATURE == 0.3

    def test_create_llm_client_called_with_conservative_temperature(
        self, fresh_risk_state: RiskDebateState, default_settings: Settings
    ) -> None:
        mock_assessment = _build_mock_assessment()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_assessment)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        with patch(
            "financialagents.agents.risk_mgmt.conservative.create_llm_client",
            return_value=mock_llm,
        ) as mock_create:
            config = {"configurable": {"settings": default_settings}}
            asyncio.run(conservative_risk_node(fresh_risk_state, config))
        mock_create.assert_called_once_with(
            "quick_think",
            default_settings,
            temperature=0.3,
            call_site="conservative_risk",
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
            "financialagents.agents.risk_mgmt.conservative.create_llm_client",
            return_value=mock_llm,
        ):
            config = {"configurable": {"settings": default_settings}}
            result = asyncio.run(conservative_risk_node(fresh_risk_state, config))

        assert ainvoke_mock.await_count == 2
        assert "conservative_assessment" in result

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
            "financialagents.agents.risk_mgmt.conservative.create_llm_client",
            return_value=mock_llm,
        ):
            config = {"configurable": {"settings": default_settings}}
            result = asyncio.run(conservative_risk_node(fresh_risk_state, config))

        assert ainvoke_mock.await_count == 2
        assert "conservative_assessment" in result
        sentinel = json.loads(result["conservative_assessment"])
        assert sentinel["confidence"] == 0.0
        assert sentinel["recommended_position_size_pct"] == 0.0
        assert "conservative persona unavailable" in sentinel["main_argument"].lower()
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
            "financialagents.agents.risk_mgmt.conservative.create_llm_client",
            return_value=mock_llm,
        ):
            result = asyncio.run(conservative_risk_node(fresh_risk_state, None))
        assert "conservative_assessment" in result


class TestGuard04AuditPassThrough:

    def test_helper_passes_through_when_no_signal(self) -> None:
        assessment = PersonaAssessment(
            main_argument="base thesis",
            recommended_position_size_pct=0.03,
            confidence=0.5,
        )
        out = _enforce_guard_04(assessment, trader_signal_snapshot="")
        assert out.main_argument == "base thesis"
        assert out.recommended_position_size_pct == 0.03

    def test_helper_passes_through_when_malformed_json(self) -> None:
        assessment = PersonaAssessment(
            main_argument="base",
            recommended_position_size_pct=0.02,
            confidence=0.5,
        )
        out = _enforce_guard_04(assessment, trader_signal_snapshot="not-json")
        assert out.main_argument == "base"
        assert out.recommended_position_size_pct == 0.02

    def test_helper_passes_through_when_t2t_false(self) -> None:
        assessment = PersonaAssessment(
            main_argument="aggressive BUY reasoning",
            recommended_position_size_pct=0.05,
            confidence=0.8,
        )
        out = _enforce_guard_04(assessment, '{"action":"BUY","t2t_flag":false}')
        assert out.main_argument == "aggressive BUY reasoning"
        assert out.recommended_position_size_pct == 0.05

    def test_helper_passes_through_when_action_hold(self) -> None:
        assessment = PersonaAssessment(
            main_argument="HOLD reasoning",
            recommended_position_size_pct=0.0,
            confidence=0.3,
        )
        out = _enforce_guard_04(assessment, '{"action":"HOLD","t2t_flag":true}')
        assert out.main_argument == "HOLD reasoning"

    def test_helper_passes_through_when_buy_and_t2t_true(self) -> None:
        assessment = PersonaAssessment(
            main_argument="Momentum is strong — size up aggressively.",
            recommended_position_size_pct=0.04,
            confidence=0.7,
        )
        out = _enforce_guard_04(assessment, '{"action":"BUY","t2t_flag":true}')
        assert out.main_argument == "Momentum is strong — size up aggressively."
        assert out.recommended_position_size_pct == 0.04

    def test_helper_passes_through_when_sell_and_t2t_true(self) -> None:
        assessment = PersonaAssessment(
            main_argument="Short conviction — size 3%.",
            recommended_position_size_pct=0.03,
            confidence=0.6,
        )
        out = _enforce_guard_04(assessment, '{"action":"SELL","t2t_flag":true}')
        assert out.main_argument == "Short conviction — size 3%."
        assert out.recommended_position_size_pct == 0.03
