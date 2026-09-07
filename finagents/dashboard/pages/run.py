from __future__ import annotations

from datetime import date as _date
from typing import Any

import streamlit as st

from finagents._redaction import _SECRET_PATTERN, redact_secrets  # noqa: F401  (re-exported)
from finagents.cli.ist_dates import ist_today_str
from finagents.config import Settings, resolve_model

_redact_secrets = redact_secrets
__all_redaction__ = ("_SECRET_PATTERN", "_redact_secrets")


EMPTY_HEADING: str = "Ready when you are"
EMPTY_BODY: str = (
    "Pick a ticker and run the pipeline to see analyst reports, the "
    "investment debate, risk personas, and the portfolio manager's "
    "verdict — typically 30–60s on Groq, 1–3 minutes on OpenRouter."
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
    result: dict[str, Any] | None = None
    with st.sidebar:
        st.markdown("**Run config**")
        with st.form("run_config", clear_on_submit=False):
            ticker = st.text_input("Ticker", value="RELIANCE")
            today_str = ist_today_str()
            today = _date.fromisoformat(today_str)
            date_in = st.date_input("Date", value=today)
            provider_default = _settings_to_provider_default(settings)
            provider_options = ["openrouter", "cerebras", "groq"]
            provider = st.selectbox(
                "Provider",
                options=provider_options,
                index=(
                    provider_options.index(provider_default)
                    if provider_default in provider_options
                    else 0
                ),
            )
            model = st.text_input(
                "Model",
                value=resolve_model(settings, settings.llm_provider, "deep"),
            )
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
                result = {
                    "ticker": ticker,
                    "date": date_in.strftime("%Y-%m-%d"),
                    "provider": provider,
                    "model": model,
                    "rounds": int(rounds),
                }
    return result


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
    provider = _settings_to_provider_default(settings)
    return {
        "ticker": ticker,
        "date": ist_today_str(),
        "provider": provider,
        "model": resolve_model(settings, provider, "deep"),
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
            quick_ticker,
            settings,
        )

    st.session_state["pending_run"] = {"config": pending_config}

    if not st.session_state.get("pipeline_running"):
        if (
            st.session_state.get("pipeline_thread") is not None
            or st.session_state.get("pipeline_cancel") is not None
        ):
            from finagents.dashboard.live_fragment import (
                cancel_pipeline,
            )

            cancel_pipeline()

        st.session_state.last_completed_event = None
        st.session_state.persisted = False
        st.session_state.stage_state = None
        st.session_state.run_id = None

        import asyncio as _asyncio
        import uuid as _uuid

        from finagents.agents.utils.agent_states import AgentState
        from finagents.dashboard.live_fragment import start_pipeline
        from finagents.graph.trading_graph import FinAgentsGraph
        from finagents.storage.migrations import run_migrations
        from finagents.storage.paths import signals_db_path
        from finagents.storage.signals import insert_running

        run_id = _uuid.uuid4().hex[:8]
        run_settings = settings.model_copy(
            update={
                "llm_provider": pending_config["provider"],
                "deep_think_llm": pending_config["model"],
                "quick_think_llm": pending_config["model"],
                "max_debate_rounds": int(pending_config["rounds"]),
            }
        )
        graph = FinAgentsGraph(full_pipeline=True)
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
        start_pipeline(graph.compiled(), initial, config_obj)


if st.session_state.get("pipeline_running") or st.session_state.get("last_completed_event"):
    from finagents.dashboard.live_fragment import (
        live_fragment,
        render_live_region,
    )

    render_live_region()
    live_fragment()


_last_ev = st.session_state.get("last_completed_event")
if _last_ev is not None and _last_ev.kind == "completed" and _last_ev.final_state is not None:
    import asyncio as _asyncio_post
    from pathlib import Path as _Path

    from finagents.dashboard.signal_card import render_signal_card
    from finagents.storage.signals import (
        update_errored as _update_errored_post,
        update_success,
    )

    _state = _last_ev.final_state
    render_signal_card(_state)

    if not st.session_state.get("persisted"):
        _payload = _state.model_dump_json(exclude={"messages"})
        _pd = _state.final_trade_decision
        _db_raw = st.session_state.get("run_db_path", "")
        if _db_raw and st.session_state.get("run_id"):
            _db_path = _Path(_db_raw)
            if _pd is None:
                _asyncio_post.run(
                    _update_errored_post(
                        st.session_state.run_id,
                        error_msg=("pipeline completed without producing a final trade decision"),
                        payload_json=_payload,
                        db_path=_db_path,
                    )
                )
            else:
                _sig = _pd.signal
                _asyncio_post.run(
                    update_success(
                        st.session_state.run_id,
                        action=_sig.action if _sig is not None else "HOLD",
                        confidence=(
                            float(getattr(_sig, "conviction", 0.0)) if _sig is not None else 0.0
                        ),
                        position_size_pct=(
                            float(getattr(_sig, "position_size_pct", 0.0) or 0.0)
                            if _sig is not None
                            else 0.0
                        ),
                        approved=bool(_pd.approved),
                        pm_decision=getattr(_pd, "decision", "veto"),
                        payload_json=_payload,
                        db_path=_db_path,
                    )
                )
            st.session_state.persisted = True


if _last_ev is not None and _last_ev.kind == "errored":
    import asyncio as _asyncio_err
    from pathlib import Path as _Path_err

    from finagents.storage.signals import update_errored

    err_msg_raw = str(_last_ev.error) if _last_ev.error is not None else "pipeline errored"
    err_msg = _redact_secrets(err_msg_raw)

    if not st.session_state.get("persisted"):
        _db_raw_err = st.session_state.get("run_db_path", "")
        if _db_raw_err and st.session_state.get("run_id"):
            _db_path_err = _Path_err(_db_raw_err)
            _asyncio_err.run(
                update_errored(
                    st.session_state.run_id,
                    error_msg=err_msg,
                    db_path=_db_path_err,
                )
            )
            st.session_state.persisted = True

    st.error(f"Pipeline errored: {err_msg}")
