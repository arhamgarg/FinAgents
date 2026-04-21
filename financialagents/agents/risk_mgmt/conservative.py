from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from financialagents.agents.risk_mgmt.schemas import PersonaAssessment
from financialagents.agents.utils.agent_states import RiskDebateState
from financialagents.config import Settings
from financialagents.llm_clients.factory import create_llm_client

CONSERVATIVE_TEMPERATURE = 0.3
T2T_REJECTION_TOKEN = "T2T intraday rejection"
CONSERVATIVE_SCENARIO = (
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
        "(d) confidence (float 0.0-1.0).\n\n"
        " (MANDATORY): If `trader_signal.t2t_flag` is True AND the signal "
        f"has BUY or SELL action, you MUST recommend position_size_pct=0.0 and "
        f"cite '{T2T_REJECTION_TOKEN}' verbatim in your main_argument."
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
    except (json.JSONDecodeError, TypeError):
        return assessment

    t2t_flag = bool(signal_data.get("t2t_flag", False))
    action = signal_data.get("action", "HOLD")

    if not (t2t_flag and action in ("BUY", "SELL")):
        return assessment

    if T2T_REJECTION_TOKEN in (assessment.main_argument or ""):
        return assessment

    forced_clause = (
        f"[ override] {T2T_REJECTION_TOKEN} — "
        f"deterministic post-LLM enforcement. "
    )
    assessment.main_argument = forced_clause + (assessment.main_argument or "")
    assessment.recommended_position_size_pct = 0.0
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
    )
    structured_llm = llm.with_structured_output(PersonaAssessment)

    result: PersonaAssessment
    try:
        result = await structured_llm.ainvoke(prompt)  # ty: ignore[invalid-assignment]
    except Exception as first_err:
        try:
            result = await structured_llm.ainvoke(prompt)  # ty: ignore[invalid-assignment]
        except Exception as retry_err:
            raise retry_err from first_err

    result = _enforce_guard_04(result, state.trader_signal_snapshot)
    return {"conservative_assessment": result.model_dump_json()}
