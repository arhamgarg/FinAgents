from __future__ import annotations

import asyncio
import re

import streamlit as st

from finagents.agents.utils.agent_states import AgentState
from finagents.config import Settings
from finagents.dashboard.signal_card import render_signal_card
from finagents.storage.paths import signals_db_path
from finagents.storage.signals import load_signal_by_run_id


NO_ID_COPY: str = "No signal selected. Open History and click a row."
NOT_FOUND_TEMPLATE: str = "No signal found with run_id {run_id}. Try the History page."
PARTIAL_COPY: str = "Run cancelled. Partial state saved."

_RUN_ID_RE = re.compile(r"^[0-9a-f]{8}$")


def _render_invest_debate_section(state: object) -> None:
    debate_state = getattr(state, "investment_debate_state", None)
    if debate_state is None:
        return
    bull_turns = debate_state.bull_turns()
    bear_turns = debate_state.bear_turns()
    if not bull_turns and not bear_turns:
        return

    st.markdown("## Investment debate")
    rounds = max(len(bull_turns), len(bear_turns))
    for i in range(rounds):
        st.markdown(f"### Round {i + 1}")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Bull**")
            st.write(bull_turns[i] if i < len(bull_turns) else "—")
        with cols[1]:
            st.markdown("**Bear**")
            st.write(bear_turns[i] if i < len(bear_turns) else "—")


def _render_risk_personas_section(state: object) -> None:
    import json as _json

    from finagents.agents.risk_mgmt.schemas import PersonaAssessment

    risk_state = getattr(state, "risk_debate_state", None)
    if risk_state is None:
        return

    personas: list[tuple[str, PersonaAssessment]] = []
    for persona_label, attr in (
        ("aggressive", "aggressive_assessment"),
        ("neutral", "neutral_assessment"),
        ("conservative", "conservative_assessment"),
    ):
        raw = getattr(risk_state, attr, None)
        if raw is None:
            continue
        try:
            parsed = (
                PersonaAssessment.model_validate_json(raw)
                if isinstance(raw, str)
                else PersonaAssessment.model_validate(raw)
            )
        except (ValueError, _json.JSONDecodeError):
            continue
        personas.append((persona_label, parsed))

    if not personas:
        return

    st.markdown("## Risk personas")
    cols = st.columns(len(personas))
    for i, (label, persona) in enumerate(personas):
        with cols[i]:
            st.markdown(f"**{label}**")
            st.write(persona.main_argument)



run_id = st.query_params.get("id")

if not run_id:
    st.title("Signal")
    st.info(NO_ID_COPY)
    st.stop()

if not _RUN_ID_RE.match(run_id):
    st.title("Signal")
    st.warning(NOT_FOUND_TEMPLATE.format(run_id=run_id))
    st.stop()

row = asyncio.run(load_signal_by_run_id(signals_db_path(Settings()), run_id))
if row is None:
    st.title("Signal")
    st.warning(NOT_FOUND_TEMPLATE.format(run_id=run_id))
    st.stop()

st.title(f"Signal · {row.get('ticker', '?')} · {row.get('date', '?')}")

payload = row.get("payload_json")
if not payload:
    st.warning(PARTIAL_COPY)
    with st.expander("Partial state", expanded=False):
        st.write(f"Status: **{row.get('status', '?')}**")
        if row.get("error_msg"):
            st.code(row["error_msg"])
        st.json({k: v for k, v in row.items() if k != "payload_json"})
    st.stop()

try:
    state = AgentState.model_validate_json(payload)
except (ValueError, TypeError) as e:
    st.error(f"Failed to deserialise payload_json: {e}")
    st.stop()

render_signal_card(state, expanded=True)

_render_invest_debate_section(state)
_render_risk_personas_section(state)
