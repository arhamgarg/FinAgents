from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Optional

from financialagents.agents.utils.agent_states import AgentState

NodeStatus = Literal["started", "completed", "errored", "cancelled"]
EventKind = Literal[
    "node_status", "round_advanced", "completed", "errored", "values"
]


STAGE_FOR_NODE: dict[str, str] = {
    "technical_analyst_node": "Analysts",
    "fundamentals_analyst_node": "Analysts",
    "news_analyst_node": "Analysts",
    "research_manager_node": "Research Manager",
    "debate_init_node": "Investment Debate",
    "invest_debate_node": "Investment Debate",
    "bull_researcher_node": "Investment Debate",
    "bear_researcher_node": "Investment Debate",
    "investment_plan_node": "Investment Debate",
    "trader_node": "Trader",
    "risk_init_node": "Risk Debate + PM",
    "risk_debate_node": "Risk Debate + PM",
    "aggressive_risk_node": "Risk Debate + PM",
    "neutral_risk_node": "Risk Debate + PM",
    "conservative_risk_node": "Risk Debate + PM",
    "risk_debate_consolidation_node": "Risk Debate + PM",
    "portfolio_manager_node": "Risk Debate + PM",
}


@dataclass(frozen=True)
class PipelineEvent:

    kind: EventKind
    node_name: Optional[str] = None
    status: Optional[NodeStatus] = None
    update_payload: Optional[dict[str, Any]] = None
    final_state: Optional[AgentState] = None
    error: Optional[BaseException] = None
    round: Optional[int] = None
    max_rounds: Optional[int] = None
    position_size_pct: Optional[float] = None


def _extract_position_size_pct(node_name: str, delta: Any) -> Optional[float]:
    if node_name not in (
        "aggressive_risk_node",
        "neutral_risk_node",
        "conservative_risk_node",
    ):
        return None
    if not isinstance(delta, dict):
        return None
    assess_key = node_name.replace("_risk_node", "_assessment")
    assess = delta.get(assess_key)
    if assess is None:
        return None
    if isinstance(assess, dict):
        value = assess.get("recommended_position_size_pct")
    elif isinstance(assess, str):
        try:
            parsed = json.loads(assess)
        except (ValueError, TypeError):
            return None
        if not isinstance(parsed, dict):
            return None
        value = parsed.get("recommended_position_size_pct")
    elif hasattr(assess, "recommended_position_size_pct"):
        value = getattr(assess, "recommended_position_size_pct")
    else:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def stream_pipeline_events(
    graph: Any,
    initial_state: Any,
    config: dict[str, Any],
) -> AsyncIterator[PipelineEvent]:
    last_state: Any = initial_state
    last_round: int = 0
    started_nodes: set[str] = set()

    try:
        async for ns, payload in graph.astream(
            initial_state,
            config=config,
            stream_mode=["updates", "values"],
            subgraphs=True,
        ):
            if isinstance(payload, dict) and any(
                k in STAGE_FOR_NODE for k in payload
            ):
                for node_name, delta in payload.items():
                    if node_name not in STAGE_FOR_NODE:
                        continue

                    if node_name not in started_nodes:
                        started_nodes.add(node_name)
                        yield PipelineEvent(
                            kind="node_status",
                            node_name=node_name,
                            status="started",
                        )

                    size_pct = _extract_position_size_pct(node_name, delta)

                    yield PipelineEvent(
                        kind="node_status",
                        node_name=node_name,
                        status="completed",
                        update_payload=delta if isinstance(delta, dict) else None,
                        position_size_pct=size_pct,
                    )

                    if node_name in (
                        "bull_researcher_node",
                        "bear_researcher_node",
                    ):
                        if isinstance(delta, dict):
                            new_round = max(
                                len(delta.get("bull_history") or []),
                                len(delta.get("bear_history") or []),
                            )
                            if new_round > last_round:
                                last_round = new_round
                                yield PipelineEvent(
                                    kind="round_advanced",
                                    node_name=node_name,
                                    round=new_round,
                                )
            else:
                last_state = payload
                try:
                    snapshot = (
                        AgentState.model_validate(payload)
                        if payload is not None
                        else None
                    )
                except Exception:
                    snapshot = None
                if snapshot is not None:
                    yield PipelineEvent(
                        kind="values", final_state=snapshot,
                    )

        try:
            final = (
                AgentState.model_validate(last_state)
                if last_state is not None
                else None
            )
        except Exception:
            final = None
        yield PipelineEvent(kind="completed", final_state=final)

    except asyncio.CancelledError:
        yield PipelineEvent(kind="errored", status="cancelled")
        raise
    except BaseException as exc:
        yield PipelineEvent(kind="errored", status="errored", error=exc)
        raise


__all__ = [
    "PipelineEvent",
    "EventKind",
    "NodeStatus",
    "STAGE_FOR_NODE",
    "stream_pipeline_events",
]
