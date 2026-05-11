from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from financialagents.agents.utils.agent_states import InvestDebateState
from financialagents.config import Settings
from financialagents.llm_clients.factory import create_llm_client


class InvestmentThesis(BaseModel):

    synthesis: str
    bull_strength: Optional[str] = None
    bear_strength: Optional[str] = None


def build_investment_thesis_prompt(state: InvestDebateState) -> list[BaseMessage]:
    bull_turns = state.bull_turns()
    bear_turns = state.bear_turns()

    bull_transcript = "\n\n".join(
        f"Bull round {i + 1}: {t}" for i, t in enumerate(bull_turns)
    )
    bear_transcript = "\n\n".join(
        f"Bear round {i + 1}: {t}" for i, t in enumerate(bear_turns)
    )

    system_content = (
        "You are the chief investment strategist for an Indian equity analysis "
        "pipeline. Two adversarial researchers (bull and bear) have just debated "
        "this ticker across multiple rounds. Your job is to synthesize their "
        "arguments into a single actionable investment thesis that the trader node "
        "will convert into a sized trade signal. "
        "Keep your synthesis under 800 words; lead with a clear directional "
        "recommendation (accumulate / hold / reduce), summarise the strongest "
        "bull and bear points, and close with the decisive factor that tilts "
        "the balance. Do not hedge — take a clear position."
    )
    human_content = (
        "## Research Brief\n\n"
        f"{state.research_brief_snapshot or '(no brief provided)'}\n\n"
        "## Bull Arguments (in order)\n\n"
        f"{bull_transcript or '(no bull arguments recorded)'}\n\n"
        "## Bear Arguments (in order)\n\n"
        f"{bear_transcript or '(no bear arguments recorded)'}\n\n"
        "## Your Task\n\n"
        "Synthesize the debate into an InvestmentThesis. Populate `synthesis` "
        "with the full recommendation paragraph; optionally populate "
        "`bull_strength` and `bear_strength` with one-sentence assessments "
        "of how each side held up."
    )
    return [SystemMessage(content=system_content), HumanMessage(content=human_content)]


async def investment_thesis_node(
    state: InvestDebateState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    configurable = config.get("configurable", {}) if config else {}
    settings: Settings = configurable.get("settings") or Settings()

    prompt = build_investment_thesis_prompt(state)

    llm = create_llm_client("deep_think", settings, max_output_tokens=2048, call_site="investment_thesis")
    structured_llm = llm.with_structured_output(InvestmentThesis)

    result: InvestmentThesis
    try:
        result = await structured_llm.ainvoke(prompt)
    except Exception as first_err:
        try:
            result = await structured_llm.ainvoke(prompt)
        except Exception as retry_err:
            return {
                "investment_thesis": (
                    f"[investment thesis unavailable: LLM error after retry "
                    f"— {retry_err!r} (initial: {first_err!r})]"
                )
            }

    return {"investment_thesis": result.synthesis}
