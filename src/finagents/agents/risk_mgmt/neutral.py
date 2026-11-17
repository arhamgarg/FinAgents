from __future__ import annotations

from typing import Any, Optional, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from finagents.agents.risk_mgmt.schemas import PersonaAssessment
from finagents.agents.utils.agent_states import RiskDebateState
from finagents.agents.utils.llm_retry import StructuredOutputError, ainvoke_with_retry
from finagents.agents.utils.run_settings import run_settings
from finagents.llm_clients.factory import create_llm_client

NEUTRAL_TEMPERATURE = 0.6
NEUTRAL_SCENARIO = (
    "All signals under review are positional delivery trades on NSE/BSE cash segment — evaluate "
    "risk on multi-day-to-multi-week holding horizons. "
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
        "## Prior Round Consolidation (round 2+ only)\n\n"
        f"{prior_consolidation or '(first round — no prior consolidation)'}\n\n"
        f"## Your Task\n\nAssess risk for round {state.current_round + 1}."
    )
    return [SystemMessage(content=system_content), HumanMessage(content=human_content)]


async def neutral_risk_node(
    state: RiskDebateState,
    config: Optional[RunnableConfig] = None,  # noqa: UP045
) -> dict[str, Any]:
    settings = run_settings(config)

    prompt = build_neutral_prompt(state)

    llm = create_llm_client(
        "quick_think",
        settings,
        temperature=NEUTRAL_TEMPERATURE,
        call_site="neutral_risk",
    )
    structured_llm = llm.with_structured_output(PersonaAssessment)

    result: PersonaAssessment
    try:
        result = await ainvoke_with_retry(
            structured_llm, prompt, lambda raw: cast(PersonaAssessment, raw)
        )
    except StructuredOutputError as err:
        return {
            "neutral_assessment": PersonaAssessment(
                main_argument=(
                    f"[neutral persona unavailable: LLM error after "
                    f"retry — {err.retry!r} (initial: {err.first!r})]"
                ),
                counter_to_prior_persona="",
                recommended_position_size_pct=0.0,
                confidence=0.0,
            ).model_dump_json()
        }

    return {"neutral_assessment": result.model_dump_json()}
