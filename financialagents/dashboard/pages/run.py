from __future__ import annotations

from datetime import date as _date
from typing import Any

import streamlit as st

from financialagents.cli.ist_dates import ist_today_str
from financialagents.config import Settings


EMPTY_HEADING: str = "Ready when you are"
EMPTY_BODY: str = (
    "Pick a ticker and run the pipeline to see analyst reports, the "
    "investment debate, risk personas, and the portfolio manager's "
    "verdict — typically 1–3 minutes on Ollama, 30–60s on Groq."
)
QUICK_PICKS: tuple[str, ...] = ("RELIANCE", "TCS", "INFY")


def _load_settings() -> Settings:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return Settings()


def _settings_to_provider_default(settings: Settings) -> str:
    return settings.llm_provider


def render_sidebar_form(settings: Settings) -> dict[str, Any] | None:
    with st.sidebar:
        st.markdown("**Run config**")
        with st.form("run_config", clear_on_submit=False):
            ticker = st.text_input("Ticker", value="RELIANCE")
            today_str = ist_today_str()
            today = _date.fromisoformat(today_str)
            date_in = st.date_input("Date", value=today)
            provider_default = _settings_to_provider_default(settings)
            provider = st.selectbox(
                "Provider",
                options=["ollama", "groq"],
                index=0 if provider_default == "ollama" else 1,
            )
            model = st.text_input("Model", value=settings.deep_think_llm)
            rounds = st.number_input(
                "Debate rounds",
                min_value=1,
                max_value=5,
                value=int(settings.max_debate_rounds),
            )
            submitted = st.form_submit_button(
                "Run pipeline",
                type="primary",
            )
            if submitted:
                return {
                    "ticker": ticker,
                    "date": date_in.strftime("%Y-%m-%d"),
                    "provider": provider,
                    "model": model,
                    "rounds": int(rounds),
                }
    return None


def render_empty_state() -> str | None:
    st.markdown(f"## {EMPTY_HEADING}")
    st.write(EMPTY_BODY)
    cols = st.columns(len(QUICK_PICKS))
    for i, ticker in enumerate(QUICK_PICKS):
        with cols[i]:
            if st.button(ticker, key=f"quick_pick_{ticker}"):
                return ticker
    return None


def _build_pending_run_from_quick_pick(
    ticker: str,
    settings: Settings,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "date": ist_today_str(),
        "provider": _settings_to_provider_default(settings),
        "model": settings.deep_think_llm,
        "rounds": int(settings.max_debate_rounds),
    }



settings = _load_settings()
config = render_sidebar_form(settings)
quick_ticker = render_empty_state()

if config is not None or quick_ticker is not None:
    if config is not None:
        pending_config = config
    else:
        assert quick_ticker is not None
        pending_config = _build_pending_run_from_quick_pick(
            quick_ticker, settings,
        )

    st.session_state["pending_run"] = {"config": pending_config}

    if not st.session_state.get("pipeline_running"):
        import asyncio as _asyncio
        import uuid as _uuid

        from financialagents.agents.utils.agent_states import AgentState
        from financialagents.dashboard.live_fragment import start_pipeline
        from financialagents.graph.trading_graph import FinancialAgentsGraph
        from financialagents.storage.migrations import run_migrations
        from financialagents.storage.paths import signals_db_path
        from financialagents.storage.signals import insert_running

        run_id = _uuid.uuid4().hex[:8]
        run_settings = settings.model_copy(
            update={
                "llm_provider": pending_config["provider"],
                "max_debate_rounds": int(pending_config["rounds"]),
            }
        )
        graph = FinancialAgentsGraph(full_pipeline=True)
        initial = AgentState(
            ticker=pending_config["ticker"],
            date=pending_config["date"],
        )
        config_obj = {
            "configurable": {
                "thread_id": run_id,
                "settings": run_settings,
            }
        }
        db_path = signals_db_path(run_settings)
        _asyncio.run(run_migrations(db_path))
        _asyncio.run(
            insert_running(
                run_id,
                pending_config["ticker"],
                pending_config["date"],
                provider=pending_config["provider"],
                model=pending_config["model"],
                db_path=db_path,
            )
        )
        st.session_state.run_id = run_id
        st.session_state.run_db_path = str(db_path)
        st.session_state.persisted = False
        start_pipeline(graph._graph, initial, config_obj)


if (
    st.session_state.get("pipeline_running")
    or st.session_state.get("last_completed_event")
):
    from financialagents.dashboard.live_fragment import (
        live_fragment,
        render_live_region,
    )

    render_live_region()
    live_fragment()


_last_ev = st.session_state.get("last_completed_event")
if (
    _last_ev is not None
    and _last_ev.kind == "completed"
    and _last_ev.final_state is not None
):
    import asyncio as _asyncio_post
    from pathlib import Path as _Path

    from financialagents.dashboard.signal_card import render_signal_card
    from financialagents.storage.signals import update_success

    render_signal_card(_last_ev.final_state)

    if not st.session_state.get("persisted"):
        _state = _last_ev.final_state
        _payload = _state.model_dump_json(exclude={"messages"})
        _pd = _state.final_trade_decision
        _sig = _pd.signal if _pd is not None else None
        _db_path = _Path(
            st.session_state.get("run_db_path", "")
        )
        if _db_path and st.session_state.get("run_id"):
            _asyncio_post.run(
                update_success(
                    st.session_state.run_id,
                    action=_sig.action if _sig is not None else "HOLD",
                    confidence=(
                        float(getattr(_sig, "conviction", 0.0))
                        if _sig is not None
                        else 0.0
                    ),
                    position_size_pct=(
                        float(getattr(_sig, "position_size_pct", 0.0) or 0.0)
                        if _sig is not None
                        else 0.0
                    ),
                    approved=bool(_pd.approved) if _pd is not None else False,
                    pm_decision=(
                        getattr(_pd, "decision", "veto")
                        if _pd is not None
                        else "veto"
                    ),
                    payload_json=_payload,
                    db_path=_db_path,
                )
            )
            st.session_state.persisted = True
