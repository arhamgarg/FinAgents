from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from finagents.agents.utils.agent_states import InvestDebateState
from finagents.config import Settings
from finagents.llm_clients.factory import create_llm_client


class BearArgument(BaseModel):

    argument: str
    key_points: list[str] = Field(default_factory=list)


def build_bear_prompt(state: InvestDebateState) -> list[BaseMessage]:
    prior_bull = "\n\n".join(f"Round {i}: {t}" for i, t in sorted(state.bull_history.items()))

    system_content = (
        "You are the bear researcher in an adversarial investment debate. "
        "Your goal: make the strongest adversarial case (SELL or HOLD) for "
        "this Indian equity based on the research brief and any prior bull "
        "arguments. Keep your argument under 800 words; lead with the risk "
        "thesis, support with 3-5 points, close with a direct rebuttal to "
        "the prior bull argument (if any). Do not hedge — take a clear "
        "adversarial position."
    )
    human_content = (
        "## Research Brief\n\n"
        f"{state.research_brief_snapshot or '(no brief provided)'}\n\n"
        "## Prior Bull Arguments\n\n"
        f"{prior_bull or '(none — this is the first round)'}\n\n"
        f"## Your Task\n\nMake your bear case for round {state.current_round + 1}."
    )
    return [SystemMessage(content=system_content), HumanMessage(content=human_content)]


async def bear_researcher_node(
    state: InvestDebateState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    configurable = config.get("configurable", {}) if config else {}
    settings: Settings = configurable.get("settings") or Settings()

    round_key = state.current_round

    prompt = build_bear_prompt(state)

    llm = create_llm_client("deep_think", settings, call_site="bear_researcher")
    structured_llm = llm.with_structured_output(BearArgument)

    result: BearArgument
    try:
        result = BearArgument.model_validate(await structured_llm.ainvoke(prompt))
    except Exception as first_err:
        try:
            result = BearArgument.model_validate(await structured_llm.ainvoke(prompt))
        except Exception as retry_err:
            return {
                "bear_history": {
                    round_key: (
                        f"[bear argument unavailable: LLM error after retry "
                        f"— {retry_err!r} (initial: {first_err!r})]"
                    )
                },
                "current_round": state.current_round + 1,
            }

    return {
        "bear_history": {round_key: result.argument},
        "current_round": state.current_round + 1,
    }
