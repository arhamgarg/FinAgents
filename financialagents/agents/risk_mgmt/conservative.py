from __future__ import annotations

import json
import logging
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from financialagents.agents.risk_mgmt.schemas import PersonaAssessment
from financialagents.agents.utils.agent_states import RiskDebateState
from financialagents.config import Settings
from financialagents.llm_clients.factory import create_llm_client

CONSERVATIVE_TEMPERATURE = 0.3


log = logging.getLogger(__name__)

CONSERVATIVE_SCENARIO = (
    "All signals under review are positional delivery trades on NSE/BSE cash segment — evaluate risk on multi-day-to-multi-week holding horizons. "
    "You are the conservative risk persona for an Indian equity pipeline. "
    "Ground your reasoning in Yes Bank March 2020: holders who refused to "
    "trim positions saw a 90% drawdown in weeks. Preserved capital is "
    "preserved optionality. Prefer HOLD when structural risk (circuit "
    "tightness, T2T restriction, earnings gap, sector bleed) is visible. "
    "Size small; skip freely."
)


def build_conservative_prompt(state: RiskDebateState) -> list[BaseMessage]:
    prior_consolidation = ""
    if state.current_round >= 1:
        prior_consolidation = state.risk_debate_history.get(state.current_round - 1, "")

    aggressive_view = state.aggressive_assessment or "(no aggressive assessment yet)"
    neutral_view = state.neutral_assessment or "(no neutral assessment yet)"

    system_content = (
        CONSERVATIVE_SCENARIO + "\n\n"
        "Emit a PersonaAssessment: (a) main_argument (<= 400 words, lead with "
        "your sizing decision; explain structural risks); "
        "(b) counter_to_prior_persona — fill with your rebuttal to the aggressive "
        "AND neutral personas' cases; "
        "(c) recommended_position_size_pct (float 0.0-1.0); "
        "(d) confidence (float 0.0-1.0)."
    )
    human_content = (
        "## Trader Signal (JSON)\n\n"
        f"{state.trader_signal_snapshot or '(no signal provided)'}\n\n"
        "## Aggressive Persona Assessment (intra-round)\n\n"
        f"{aggressive_view}\n\n"
        "## Neutral Persona Assessment (intra-round)\n\n"
        f"{neutral_view}\n\n"
        "## Prior Round Consolidation — round 2+ only)\n\n"
        f"{prior_consolidation or '(first round — no prior consolidation)'}\n\n"
        f"## Your Task\n\nAssess risk for round {state.current_round + 1}."
    )
    return [SystemMessage(content=system_content), HumanMessage(content=human_content)]


def _enforce_guard_04(
    assessment: PersonaAssessment, trader_signal_snapshot: str
) -> PersonaAssessment:
    try:
        signal_data = json.loads(trader_signal_snapshot)
        log.debug(
            " audit: t2t_flag=%s, action=%s, time_horizon=%s "
            "(positional-only policy — override retired in 4.2.1)",
            signal_data.get("t2t_flag", False),
            signal_data.get("action", "HOLD"),
            signal_data.get("time_horizon", "positional"),
        )
    except (json.JSONDecodeError, TypeError):
        pass
    return assessment


async def conservative_risk_node(
    state: RiskDebateState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    configurable = config.get("configurable", {}) if config else {}
    settings: Settings = configurable.get("settings") or Settings()

    prompt = build_conservative_prompt(state)

    llm = create_llm_client(
        "quick_think",
        settings,
        max_output_tokens=1024,
        temperature=CONSERVATIVE_TEMPERATURE,
        call_site="conservative_risk",
    )
    structured_llm = llm.with_structured_output(PersonaAssessment)

    result: PersonaAssessment
    try:
        result = await structured_llm.ainvoke(prompt)  # ty: ignore[invalid-assignment]
    except Exception as first_err:
        try:
            result = await structured_llm.ainvoke(prompt)  # ty: ignore[invalid-assignment]
        except Exception as retry_err:
            return {
                "conservative_assessment": PersonaAssessment(
                    main_argument=(
                        f"[conservative persona unavailable: LLM error after "
                        f"retry — {retry_err!r} (initial: {first_err!r})]"
                    ),
                    counter_to_prior_persona="",
                    recommended_position_size_pct=0.0,
                    confidence=0.0,
                ).model_dump_json()
            }

    result = _enforce_guard_04(result, state.trader_signal_snapshot)
    return {"conservative_assessment": result.model_dump_json()}
