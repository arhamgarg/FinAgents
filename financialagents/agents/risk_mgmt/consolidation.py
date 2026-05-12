from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from financialagents.agents.risk_mgmt.schemas import RiskConsolidation
from financialagents.agents.utils.agent_states import RiskDebateState
from financialagents.config import Settings
from financialagents.llm_clients.factory import create_llm_client


def _parse_size(assessment_json: Optional[str]) -> float:
    if not assessment_json:
        return 0.0
    try:
        data = json.loads(assessment_json)
        if not isinstance(data, dict):
            return 0.0
        value = data.get("recommended_position_size_pct", 0.0)
        return float(value) if value is not None else 0.0
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def build_consolidation_prompt(state: RiskDebateState) -> list[BaseMessage]:
    prior_consolidation = ""
    if state.current_round >= 1:
        prior_consolidation = state.risk_debate_history.get(
            state.current_round - 1, ""
        )

    system_content = (
        "You are the risk consolidation node. Three risk personas (aggressive, "
        "neutral, conservative) have filed their PersonaAssessment views for "
        "this round. Your job is to synthesize their arguments into a single "
        "consolidation paragraph for the Portfolio Manager to read.\n\n"
        "IMPORTANT: do NOT compute a weighted numeric size yourself — the "
        "Settings-weighted position size will be computed deterministically in "
        "Python from each persona's recommended_position_size_pct. You produce "
        "only the textual synthesis: capture where the personas agree, where "
        "they disagree, and what the disagreement hinges on. <= 300 words."
    )
    human_content = (
        "## Trader Signal (JSON)\n\n"
        f"{state.trader_signal_snapshot or '(no signal provided)'}\n\n"
        "## Aggressive Persona Assessment (JSON)\n\n"
        f"{state.aggressive_assessment or '(no aggressive assessment)'}\n\n"
        "## Neutral Persona Assessment (JSON)\n\n"
        f"{state.neutral_assessment or '(no neutral assessment)'}\n\n"
        "## Conservative Persona Assessment (JSON)\n\n"
        f"{state.conservative_assessment or '(no conservative assessment)'}\n\n"
        "## Prior Round Consolidation — round 2+ only)\n\n"
        f"{prior_consolidation or '(first round — no prior consolidation)'}\n\n"
        f"## Your Task\n\nSynthesize for round {state.current_round + 1}."
    )
    return [SystemMessage(content=system_content), HumanMessage(content=human_content)]


async def risk_debate_consolidation_node(
    state: RiskDebateState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    configurable = config.get("configurable", {}) if config else {}
    settings: Settings = configurable.get("settings") or Settings()

    prompt = build_consolidation_prompt(state)

    llm = create_llm_client(
        "deep_think",
        settings,
        max_output_tokens=1024,
        call_site="risk_debate_consolidation",
    )
    structured_llm = llm.with_structured_output(RiskConsolidation)

    weights = settings.risk_persona_weights
    agg_size = _parse_size(state.aggressive_assessment)
    neu_size = _parse_size(state.neutral_assessment)
    con_size = _parse_size(state.conservative_assessment)
    synthesized_size = (
        weights.get("aggressive", 0.0) * agg_size
        + weights.get("neutral", 0.0) * neu_size
        + weights.get("conservative", 0.0) * con_size
    )
    round_key = state.current_round

    result: RiskConsolidation
    try:
        result = await structured_llm.ainvoke(prompt)  # ty: ignore[invalid-assignment]
    except Exception as first_err:
        try:
            result = await structured_llm.ainvoke(prompt)  # ty: ignore[invalid-assignment]
        except Exception as retry_err:
            return {
                "risk_debate_history": {
                    round_key: (
                        f"[consolidation unavailable: LLM error after retry "
                        f"— {retry_err!r} (initial: {first_err!r})]"
                    )
                },
                "current_round": state.current_round + 1,
                "synthesized_position_size_pct": synthesized_size,
            }

    return {
        "risk_debate_history": {round_key: result.synthesis},
        "current_round": state.current_round + 1,
        "synthesized_position_size_pct": synthesized_size,
    }
