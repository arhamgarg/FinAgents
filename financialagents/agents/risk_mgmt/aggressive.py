from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from financialagents.agents.risk_mgmt.schemas import PersonaAssessment
from financialagents.agents.utils.agent_states import RiskDebateState
from financialagents.config import Settings
from financialagents.llm_clients.factory import create_llm_client

AGGRESSIVE_TEMPERATURE = 0.9
AGGRESSIVE_SCENARIO = (
    "You are the aggressive risk persona for an Indian equity pipeline. "
    "Ground your reasoning in the COVID-2020 bounce: the trader who bought "
    "TCS at March 2020 lows rode the recovery to 2x returns. The opportunity "
    "cost of HOLD is real — an unheld signal that becomes a trend is a miss. "
    "Lean toward sizing up when conviction is real; hedge only when structural "
    "risk (circuit, T2T, earnings gap) is demonstrated."
)


def build_aggressive_prompt(state: RiskDebateState) -> list[BaseMessage]:
    prior_consolidation = ""
    if state.current_round >= 1:
        prior_consolidation = state.risk_debate_history.get(state.current_round - 1, "")

    system_content = (
        AGGRESSIVE_SCENARIO + "\n\n"
        "Emit a PersonaAssessment: (a) main_argument (<= 400 words, lead with "
        "the size recommendation, then explain; do not hedge); "
        "(b) counter_to_prior_persona (empty string for round-1 aggressive — "
        "you have no prior persona to counter); "
        "(c) recommended_position_size_pct (float 0.0-1.0); "
        "(d) confidence (float 0.0-1.0)."
    )
    human_content = (
        "## Trader Signal (JSON)\n\n"
        f"{state.trader_signal_snapshot or '(no signal provided)'}\n\n"
        "## Prior Round Consolidation — round 2+ only)\n\n"
        f"{prior_consolidation or '(first round — no prior consolidation)'}\n\n"
        f"## Your Task\n\nAssess risk for round {state.current_round + 1}."
    )
    return [SystemMessage(content=system_content), HumanMessage(content=human_content)]


async def aggressive_risk_node(
    state: RiskDebateState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    configurable = config.get("configurable", {}) if config else {}
    settings: Settings = configurable.get("settings") or Settings()

    prompt = build_aggressive_prompt(state)

    llm = create_llm_client(
        "quick_think",
        settings,
        max_output_tokens=1024,
        temperature=AGGRESSIVE_TEMPERATURE,
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

    return {"aggressive_assessment": result.model_dump_json()}
