from __future__ import annotations

from typing import Any

import streamlit as st


VERDICT_COLOR_HEX: dict[str, str] = {
    "BUY": "#2EA043",
    "SELL": "#F85149",
    "HOLD": "#D29922",
}
APPROVAL_COLOR_HEX: dict[str, str] = {
    "APPROVED": "#2EA043",
    "MODIFIED": "#D29922",
    "VETOED": "#F85149",
}

_HEX_TO_STREAMLIT_ALIAS: dict[str, str] = {
    "#2EA043": "green",
    "#F85149": "red",
    "#D29922": "orange",
}


def verdict_color(action: str) -> str:
    return VERDICT_COLOR_HEX[action]


def _streamlit_alias(hex_color: str) -> str:
    return _HEX_TO_STREAMLIT_ALIAS.get(hex_color, "blue")


def _approval_label(decision: str, approved: bool) -> str:
    if decision == "veto":
        return "VETOED"
    if decision == "modify":
        return "MODIFIED"
    return "APPROVED" if approved else "VETOED"


def _format_size_str(position_size_pct: Any) -> str:
    if position_size_pct is None:
        return "—"
    pct = float(position_size_pct)
    if pct <= 1:
        return f"{pct * 100:.1f}%"
    return f"{pct:.1f}%"


def render_signal_card(state: Any, *, expanded: bool = False) -> None:
    pd_ = getattr(state, "final_trade_decision", None)
    if pd_ is None:
        st.warning(
            "No PortfolioDecision available — the run was cancelled or "
            "errored before the portfolio manager produced a verdict."
        )
        return

    sig = pd_.signal
    action = sig.action
    approval = _approval_label(pd_.decision, pd_.approved)
    verdict_alias = _streamlit_alias(VERDICT_COLOR_HEX.get(action, "#888"))
    approval_alias = _streamlit_alias(APPROVAL_COLOR_HEX.get(approval, "#888"))

    with st.container(border=True):
        st.markdown(f"**{state.ticker} · {state.date}**")

        st.markdown(
            f"#:{verdict_alias}[{action}]  "
            f":{approval_alias}[\\[{approval}\\]]"
        )

        cols = st.columns(4)
        cols[0].metric(
            "Entry",
            f"₹{sig.entry_price:,.0f}" if sig.entry_price is not None else "—",
        )
        cols[1].metric(
            "Stop",
            f"₹{sig.stop_loss:,.0f}" if sig.stop_loss is not None else "—",
        )
        cols[2].metric(
            "Target",
            f"₹{sig.target_1:,.0f}" if sig.target_1 is not None else "—",
        )
        cols[3].metric("Size", _format_size_str(sig.position_size_pct))

        if pd_.pm_notes:
            st.write(pd_.pm_notes)

        with st.expander("Bull / Bear debate", expanded=expanded):
            _render_invest_debate(state)
        with st.expander("Risk personas", expanded=expanded):
            _render_risk_personas(state)
        with st.expander("Agent reports", expanded=expanded):
            from financialagents.dashboard.agent_reports import (
                render_agent_reports,
            )
            render_agent_reports(state)


def _render_invest_debate(state: Any) -> None:
    debate_state = getattr(state, "investment_debate_state", None)
    if debate_state is None:
        st.write("No investment debate transcript available.")
        return
    bull_turns = debate_state.bull_turns()
    bear_turns = debate_state.bear_turns()
    if not bull_turns and not bear_turns:
        st.write("No investment debate transcript available.")
        return
    rounds = max(len(bull_turns), len(bear_turns))
    for i in range(rounds):
        st.markdown(f"**Round {i + 1}**")
        cols = st.columns(2)
        cols[0].markdown(
            f"**Bull**\n\n{bull_turns[i] if i < len(bull_turns) else ''}"
        )
        cols[1].markdown(
            f"**Bear**\n\n{bear_turns[i] if i < len(bear_turns) else ''}"
        )


def _render_risk_personas(state: Any) -> None:
    import json as _json

    from financialagents.agents.risk_mgmt.schemas import PersonaAssessment
    from financialagents.dashboard.divergence_chart import (
        render_divergence_chart,
    )

    risk_state = getattr(state, "risk_debate_state", None)
    personas: list[PersonaAssessment] = []
    persona_names: list[str] = []
    if risk_state is not None:
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
            personas.append(parsed)
            persona_names.append(persona_label)

    if not personas:
        st.write("No persona assessments available.")
        return

    synth_pct = (
        getattr(risk_state, "synthesized_position_size_pct", None) or 0.0
    )
    synth_pp = synth_pct * 100 if synth_pct <= 1 else synth_pct

    render_divergence_chart(personas, synth_pp)

    cols = st.columns(len(personas))
    for i, (label, p) in enumerate(zip(persona_names, personas)):
        with cols[i]:
            st.markdown(f"**{label}**")
            st.write(p.main_argument)


__all__ = [
    "APPROVAL_COLOR_HEX",
    "VERDICT_COLOR_HEX",
    "render_signal_card",
    "verdict_color",
]
