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
    "All signals under review are positional delivery trades on NSE/BSE cash segment — evaluate risk on multi-day-to-multi-week holding horizons. "
    "You are the aggressive risk persona for an Indian equity pipeline. "
    "Ground your reasoning in the Tata Motors 2020-2024 JLR turnaround: "
    "the trader who bought Tata Motors at the March 2020 low (~Rs 65) and "
    "held through the multi-year JLR recovery and EV re-rating compounded "
    "~10x by 2024 (~Rs 950+). The opportunity cost of HOLD is real — a "
    "multi-year delivery conviction that becomes a sustained trend is a "
    "miss that cannot be recovered. Lean toward sizing up when conviction "
    "is real; hedge only when structural risk (circuit, T2T, earnings gap, "
    "sector bleed) is demonstrated."
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
            return {
                "aggressive_assessment": PersonaAssessment(
                    main_argument=(
                        f"[aggressive persona unavailable: LLM error after "
                        f"retry — {retry_err!r} (initial: {first_err!r})]"
                    ),
                    counter_to_prior_persona="",
                    recommended_position_size_pct=0.0,
                    confidence=0.0,
                ).model_dump_json()
            }

    return {"aggressive_assessment": result.model_dump_json()}
