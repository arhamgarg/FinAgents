from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from financialagents.agents.risk_mgmt.schemas import PersonaAssessment
from financialagents.agents.utils.agent_states import RiskDebateState
from financialagents.config import Settings
from financialagents.llm_clients.factory import create_llm_client

NEUTRAL_TEMPERATURE = 0.6
NEUTRAL_SCENARIO = (
    "You are the neutral risk persona for an Indian equity pipeline. "
    "Ground your reasoning in the post-COVID 2023 NIFTY-50 range-bound "
    "consolidation (17,000–19,000): disciplined position-sizing outperformed "
    "both over-aggressive buyers and cash-holders. Take the median view — "
    "size with a clear edge, skip on weak edges."
)


def build_neutral_prompt(state: RiskDebateState) -> list[BaseMessage]:
    prior_consolidation = ""
    if state.current_round >= 1:
        prior_consolidation = state.risk_debate_history.get(state.current_round - 1, "")

    aggressive_view = state.aggressive_assessment or "(no aggressive assessment yet)"

    system_content = (
        NEUTRAL_SCENARIO + "\n\n"
        "Emit a PersonaAssessment: (a) main_argument (<= 400 words, lead with "
        "your sizing decision; explain the edge); "
        "(b) counter_to_prior_persona — fill with your rebuttal to the aggressive "
        "persona's case; "
        "(c) recommended_position_size_pct (float 0.0-1.0); "
        "(d) confidence (float 0.0-1.0)."
    )
    human_content = (
        "## Trader Signal (JSON)\n\n"
        f"{state.trader_signal_snapshot or '(no signal provided)'}\n\n"
        "## Aggressive Persona Assessment (intra-round)\n\n"
        f"{aggressive_view}\n\n"
        "## Prior Round Consolidation — round 2+ only)\n\n"
        f"{prior_consolidation or '(first round — no prior consolidation)'}\n\n"
        f"## Your Task\n\nAssess risk for round {state.current_round + 1}."
    )
    return [SystemMessage(content=system_content), HumanMessage(content=human_content)]


async def neutral_risk_node(
    state: RiskDebateState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    configurable = config.get("configurable", {}) if config else {}
    settings: Settings = configurable.get("settings") or Settings()

    prompt = build_neutral_prompt(state)

    llm = create_llm_client(
        "quick_think",
        settings,
        max_output_tokens=1024,
        temperature=NEUTRAL_TEMPERATURE,
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

    return {"neutral_assessment": result.model_dump_json()}
