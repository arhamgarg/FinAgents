from __future__ import annotations

import logging
import re
import sys
import uuid
from typing import Optional

from financialagents.cli.ist_dates import ist_today_str
from financialagents.cli.live import (
    install_sigint_handler,
    run_with_live_render,
    run_with_log_lines,
)
from financialagents.graph.trading_graph import FinancialAgentsGraph
from financialagents.observability.events import stream_pipeline_events
from financialagents.storage.paths import signals_db_path

_SECRET_PATTERN = re.compile(
    r"((?:GROQ_API_KEY|ANGELONE_[A-Z_]+|REDDIT_[A-Z_]+|(?<![A-Z_])api_key))"
    r"[=:]\s*['\"]?\S+['\"]?",
    re.IGNORECASE,
)


def _redact_secrets(text: str) -> str:
    return _SECRET_PATTERN.sub(lambda m: f"{m.group(1)}=***", text)


def _new_run_id() -> str:
    return uuid.uuid4().hex[:8]


async def run_pipeline(
    ticker: str,
    date: Optional[str],
    *,
    json_output: bool,
    provider_override: Optional[str],
    debug: bool,
) -> int:
    from rich.console import Console

    from financialagents.agents.utils.agent_states import AgentState
    from financialagents.cli.panels import (
        render_cancelled_panel,
        render_error_panel,
        render_signal_card,
    )
    from financialagents.cli.picker import (
        list_configured_providers,
        prompt_provider,
        should_skip_picker,
    )
    from financialagents.config import Settings
    from financialagents.storage.migrations import run_migrations
    from financialagents.storage.signals import (
        insert_running,
        update_cancelled,
        update_errored,
        update_success,
    )

    if debug:
        logging.basicConfig(level=logging.DEBUG)

    out_console = Console(file=sys.stderr if json_output else sys.stdout)
    json_console = Console(file=sys.stdout)

    settings = Settings()
    if provider_override is not None:
        settings = settings.model_copy(update={"llm_provider": provider_override})

    if should_skip_picker(settings, provider_flag=provider_override):
        providers = list_configured_providers(settings)
        provider, model = providers[0]
        if provider_override is not None:
            for name, m in providers:
                if name == provider_override:
                    provider, model = name, m
                    break
            else:
                provider, model = provider_override, settings.deep_think_llm
    else:
        provider, model = prompt_provider(settings)

    date_str = date or ist_today_str()
    run_id = _new_run_id()
    db_path = signals_db_path(settings)
    await run_migrations(db_path)
    await insert_running(
        run_id,
        ticker,
        date_str,
        provider=provider,
        model=model,
        db_path=db_path,
    )

    cancel_flag = None
    try:
        graph = FinancialAgentsGraph(full_pipeline=True)
        initial_state = AgentState(ticker=ticker, date=date_str)
        config = {
            "configurable": {
                "settings": settings,
                "thread_id": run_id,
            }
        }

        cancel_flag = install_sigint_handler()

        events = stream_pipeline_events(
            graph._graph, initial_state, config,
        )

        if json_output:
            log_console = Console(file=sys.stderr)
            state = await run_with_log_lines(
                events,
                ticker=ticker,
                date=date_str,
                provider=provider,
                model=model,
                log_console=log_console,
                cancel_flag=cancel_flag,
            )
        else:
            state = await run_with_live_render(
                events,
                ticker=ticker,
                date=date_str,
                provider=provider,
                model=model,
                max_debate_rounds=settings.max_debate_rounds,
                console=out_console,
                cancel_flag=cancel_flag,
            )

        if cancel_flag is not None and cancel_flag.is_set():
            cancel_payload = (
                state.model_dump_json(exclude={"messages"})
                if state is not None
                else None
            )
            await update_cancelled(
                run_id,
                error_msg="user cancelled (SIGINT)",
                payload_json=cancel_payload,
                db_path=db_path,
            )
            out_console.print(render_cancelled_panel("current node", run_id))
            return 130

        if state is None:
            await update_errored(
                run_id,
                error_msg="event stream ended without completion",
                db_path=db_path,
            )
            return 1

        payload_json = state.model_dump_json(exclude={"messages"})
        pd_ = state.final_trade_decision
        ts = pd_.signal if pd_ is not None else None
        await update_success(
            run_id,
            action=ts.action if ts else "HOLD",
            confidence=getattr(ts, "conviction", 0.0) if ts else 0.0,
            position_size_pct=(
                getattr(ts, "position_size_pct", 0.0) or 0.0 if ts else 0.0
            ),
            approved=bool(pd_.approved) if pd_ else False,
            pm_decision=getattr(pd_, "decision", "veto") if pd_ else "veto",
            payload_json=payload_json,
            db_path=db_path,
        )
        if json_output:
            json_console.print_json(payload_json)
        else:
            out_console.print(render_signal_card(state))
        return 0
    except KeyboardInterrupt:
        await update_cancelled(
            run_id, error_msg="user cancelled (SIGINT)", db_path=db_path,
        )
        out_console.print("[yellow]Cancelled[/yellow]")
        return 130
    except BaseException as e:
        redacted = _redact_secrets(str(e))
        await update_errored(run_id, error_msg=redacted, db_path=db_path)
        if debug:
            raise
        out_console.print(render_error_panel(e))
        return 1
