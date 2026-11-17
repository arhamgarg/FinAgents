from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from finagents.agents.utils.agent_states import AgentState

NodeStatus = Literal["started", "completed", "errored", "cancelled"]
EventKind = Literal["node_status", "round_advanced", "completed", "errored", "values"]


STAGE_FOR_NODE: dict[str, str] = {
    "technical_analyst_node": "Analysts",
    "fundamentals_analyst_node": "Analysts",
    "news_analyst_node": "Analysts",
    "research_manager_node": "Research Manager",
    "debate_init_node": "Investment Debate",
    "invest_debate_node": "Investment Debate",
    "bull_researcher_node": "Investment Debate",
    "bear_researcher_node": "Investment Debate",
    "investment_thesis_node": "Investment Debate",
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
    node_name: str | None = None
    status: NodeStatus | None = None
    update_payload: dict[str, Any] | None = None
    final_state: AgentState | None = None
    error: BaseException | None = None
    round: int | None = None
    max_rounds: int | None = None
    position_size_pct: float | None = None


def _extract_position_size_pct(node_name: str, delta: Any) -> float | None:  # noqa: PLR0911
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
        value = assess.recommended_position_size_pct
    else:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _events_for_step(
    ns: Any,
    mode: Any,
    payload: Any,
    state: dict[str, Any],
) -> list[PipelineEvent]:
    out: list[PipelineEvent] = []

    if mode == "updates" and isinstance(payload, dict):
        for node_name, delta in payload.items():
            if node_name not in STAGE_FOR_NODE:
                continue

            started_nodes: set[str] = state["started_nodes"]
            if node_name not in started_nodes:
                started_nodes.add(node_name)
                out.append(
                    PipelineEvent(
                        kind="node_status",
                        node_name=node_name,
                        status="started",
                    )
                )

            size_pct = _extract_position_size_pct(node_name, delta)
            out.append(
                PipelineEvent(
                    kind="node_status",
                    node_name=node_name,
                    status="completed",
                    update_payload=delta if isinstance(delta, dict) else None,
                    position_size_pct=size_pct,
                )
            )

            if node_name == "bear_researcher_node" and isinstance(delta, dict):
                new_round = delta.get("current_round")
                if (
                    new_round is not None
                    and isinstance(new_round, int)
                    and new_round > state["last_round"]
                ):
                    state["last_round"] = new_round
                    out.append(
                        PipelineEvent(
                            kind="round_advanced",
                            node_name=node_name,
                            round=new_round,
                        )
                    )

    elif mode == "values" and ns == ():
        state["last_state"] = payload
        try:
            snapshot = AgentState.model_validate(payload) if payload is not None else None
        except ValidationError:
            snapshot = None
        if snapshot is not None:
            out.append(PipelineEvent(kind="values", final_state=snapshot))

    return out


async def stream_pipeline_events(
    graph: Any,
    initial_state: Any,
    config: dict[str, Any],
    *,
    cancel_event: Any | None = None,
) -> AsyncIterator[PipelineEvent]:
    state: dict[str, Any] = {
        "last_state": initial_state,
        "last_round": 0,
        "started_nodes": set(),
    }

    agen = graph.astream(
        initial_state,
        config=config,
        stream_mode=["updates", "values"],
        subgraphs=True,
    ).__aiter__()

    final_events: list[PipelineEvent] = []
    while True:
        if cancel_event is not None and cancel_event.is_set():
            with contextlib.suppress(BaseException):
                await agen.aclose()
            final_events.append(PipelineEvent(kind="errored", status="cancelled"))
            break
        try:
            step = await agen.__anext__()
        except StopAsyncIteration:
            try:
                final = (
                    AgentState.model_validate(state["last_state"])
                    if state["last_state"] is not None
                    else None
                )
            except ValidationError:
                final = None
            final_events.append(PipelineEvent(kind="completed", final_state=final))
            break
        except asyncio.CancelledError:
            final_events.append(PipelineEvent(kind="errored", status="cancelled"))
            for ev in final_events:
                yield ev
            raise
        except BaseException as exc:
            final_events.append(PipelineEvent(kind="errored", status="errored", error=exc))
            for ev in final_events:
                yield ev
            raise

        ns, mode, payload = step
        for ev in _events_for_step(ns, mode, payload, state):
            yield ev

    for ev in final_events:
        yield ev


__all__ = [
    "STAGE_FOR_NODE",
    "EventKind",
    "NodeStatus",
    "PipelineEvent",
    "stream_pipeline_events",
]
