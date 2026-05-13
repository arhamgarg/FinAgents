from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from financialagents.agents.utils.agent_states import InvestDebateState
from financialagents.config import Settings
from financialagents.llm_clients.factory import create_llm_client


class BullArgument(BaseModel):

    argument: str
    key_points: list[str] = Field(default_factory=list)


def build_bull_prompt(state: InvestDebateState) -> list[BaseMessage]:
    prior_bear = "\n\n".join(
        f"Round {i}: {t}" for i, t in sorted(state.bear_history.items())
    )

    system_content = (
        "You are the bull researcher in an adversarial investment debate. "
        "Your goal: make the strongest case for BUY on this Indian equity "
        "based on the research brief and any prior bear arguments. "
        "Keep your argument under 800 words; lead with the thesis, support "
        "with 3-5 points, close with a direct rebuttal to the prior bear "
        "argument (if any). Do not hedge — take a clear position."
    )
    human_content = (
        "## Research Brief\n\n"
        f"{state.research_brief_snapshot or '(no brief provided)'}\n\n"
        "## Prior Bear Arguments\n\n"
        f"{prior_bear or '(none — this is the first round)'}\n\n"
        f"## Your Task\n\nMake your bull case for round {state.current_round + 1}."
    )
    return [SystemMessage(content=system_content), HumanMessage(content=human_content)]


async def bull_researcher_node(
    state: InvestDebateState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    configurable = config.get("configurable", {}) if config else {}
    settings: Settings = configurable.get("settings") or Settings()

    round_key = state.current_round

    prompt = build_bull_prompt(state)

    llm = create_llm_client("deep_think", settings, call_site="bull_researcher")
    structured_llm = llm.with_structured_output(BullArgument)

    result: BullArgument
    try:
        result = await structured_llm.ainvoke(prompt)
    except Exception as first_err:
        try:
            result = await structured_llm.ainvoke(prompt)
        except Exception as retry_err:
            return {
                "bull_history": {
                    round_key: (
                        f"[bull argument unavailable: LLM error after retry "
                        f"— {retry_err!r} (initial: {first_err!r})]"
                    )
                }
            }

    return {"bull_history": {round_key: result.argument}}
