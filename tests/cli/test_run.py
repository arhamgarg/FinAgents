from __future__ import annotations

import asyncio
import json
import re
import sqlite3

import pytest

pytestmark = pytest.mark.usefixtures("fake_providers")


def _read_signals(db_path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM signals").fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


class TestRedactSecrets:

    def test_canonical_groq(self) -> None:
        from financialagents.cli.run_command import _redact_secrets

        s = "connection failed: GROQ_API_KEY=secret123"
        r = _redact_secrets(s)
        assert r == "connection failed: GROQ_API_KEY=***"
        assert "GROQ_***" not in r
        assert r.count("***") == 1

    def test_lowercase_standalone_api_key(self) -> None:
        from financialagents.cli.run_command import _redact_secrets

        assert _redact_secrets("api_key=foo") == "api_key=***"

    def test_negative_case_substring_not_matched(self) -> None:
        from financialagents.cli.run_command import _redact_secrets

        neg = _redact_secrets("my_api_key_lookup=foo")
        assert "***" not in neg

    def test_angelone_prefix(self) -> None:
        from financialagents.cli.run_command import _redact_secrets

        assert (
            _redact_secrets("ANGELONE_TOTP_SECRET=xyz") == "ANGELONE_TOTP_SECRET=***"
        )

    def test_reddit_prefix(self) -> None:
        from financialagents.cli.run_command import _redact_secrets

        assert (
            _redact_secrets("REDDIT_CLIENT_SECRET=abc") == "REDDIT_CLIENT_SECRET=***"
        )


class TestRunIdGeneration:

    def test_run_id_stable_8hex(self) -> None:
        from financialagents.cli.run_command import _new_run_id

        run_id = _new_run_id()
        assert re.match(r"^[0-9a-f]{8}$", run_id), f"unexpected run_id: {run_id!r}"


class TestFaRun:

    def test_happy_path_json(
        self, fake_pipeline, tmp_signals_db, capsys
    ) -> None:
        from financialagents.cli.run_command import run_pipeline

        rc = asyncio.run(
            run_pipeline(
                ticker="RELIANCE",
                date="2026-04-26",
                json_output=True,
                provider_override="ollama",
                debug=False,
            )
        )
        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["ticker"] == "RELIANCE"
        rows = _read_signals(tmp_signals_db)
        assert len(rows) == 1
        assert rows[0]["status"] == "success"
        assert rows[0]["action"] == "BUY"
        assert re.match(r"^[0-9a-f]{8}$", rows[0]["run_id"])

    def test_rich_panel(
        self, fake_pipeline, tmp_signals_db, capsys
    ) -> None:
        from financialagents.cli.run_command import run_pipeline

        rc = asyncio.run(
            run_pipeline(
                ticker="RELIANCE",
                date="2026-04-26",
                json_output=False,
                provider_override="ollama",
                debug=False,
            )
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "BUY" in out
        assert "RELIANCE" in out
        rows = _read_signals(tmp_signals_db)
        assert len(rows) == 1
        assert rows[0]["status"] == "success"

    def test_debug_traceback_reraises(
        self, fake_pipeline, tmp_signals_db, monkeypatch
    ) -> None:
        from financialagents.cli.run_command import run_pipeline

        async def _broken_stream(*args, **kwargs):
            raise RuntimeError("boom: GROQ_API_KEY=topsecret")
            yield  # pragma: no cover — generator marker

        monkeypatch.setattr(
            "financialagents.cli.run_command.stream_pipeline_events",
            _broken_stream,
        )

        with pytest.raises(RuntimeError):
            asyncio.run(
                run_pipeline(
                    ticker="RELIANCE",
                    date="2026-04-26",
                    json_output=False,
                    provider_override="ollama",
                    debug=True,
                )
            )
        rows = _read_signals(tmp_signals_db)
        assert len(rows) == 1
        assert rows[0]["status"] == "errored"
        assert "topsecret" not in rows[0]["error_msg"]
        assert "GROQ_API_KEY=***" in rows[0]["error_msg"]

    def test_no_debug_renders_error_panel(
        self, tmp_signals_db, capsys, monkeypatch
    ) -> None:
        from financialagents.cli.run_command import run_pipeline
        from unittest.mock import MagicMock

        fake_graph = MagicMock()
        monkeypatch.setattr(
            "financialagents.cli.run_command.FinancialAgentsGraph",
            lambda **kw: fake_graph,
        )
        monkeypatch.setattr(
            "financialagents.cli.run_command.signals_db_path",
            lambda settings=None: tmp_signals_db,
        )

        async def _broken_stream(*args, **kwargs):
            raise RuntimeError("kaboom")
            yield  # pragma: no cover — generator marker

        monkeypatch.setattr(
            "financialagents.cli.run_command.stream_pipeline_events",
            _broken_stream,
        )

        rc = asyncio.run(
            run_pipeline(
                ticker="RELIANCE",
                date="2026-04-26",
                json_output=False,
                provider_override="ollama",
                debug=False,
            )
        )
        assert rc == 1
        out = capsys.readouterr().out
        assert "kaboom" in out
        rows = _read_signals(tmp_signals_db)
        assert rows[0]["status"] == "errored"

    def test_default_date_is_ist_today(
        self, fake_pipeline, tmp_signals_db, freeze_ist_today, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "financialagents.cli.run_command.ist_today_str",
            lambda: freeze_ist_today,
            raising=False,
        )
        from financialagents.cli.run_command import run_pipeline

        rc = asyncio.run(
            run_pipeline(
                ticker="RELIANCE",
                date=None,
                json_output=True,
                provider_override="ollama",
                debug=False,
            )
        )
        assert rc == 0
        rows = _read_signals(tmp_signals_db)
        assert rows[0]["date"] == freeze_ist_today

    def test_first_ctrl_c(
        self, tmp_signals_db, capsys, monkeypatch
    ) -> None:
        from financialagents.cli.run_command import run_pipeline
        from unittest.mock import AsyncMock, MagicMock

        broken_graph = MagicMock()
        broken_graph.apropagate = AsyncMock(side_effect=KeyboardInterrupt())
        async def _ki_stream(*args, **kwargs):
            raise KeyboardInterrupt()
            yield  # pragma: no cover — generator marker

        monkeypatch.setattr(
            "financialagents.cli.run_command.FinancialAgentsGraph",
            lambda **kw: broken_graph,
        )
        monkeypatch.setattr(
            "financialagents.cli.run_command.signals_db_path",
            lambda settings=None: tmp_signals_db,
        )
        monkeypatch.setattr(
            "financialagents.cli.run_command.stream_pipeline_events",
            _ki_stream,
        )

        rc = asyncio.run(
            run_pipeline(
                ticker="RELIANCE",
                date="2026-04-26",
                json_output=False,
                provider_override="ollama",
                debug=False,
            )
        )
        assert rc == 130
        rows = _read_signals(tmp_signals_db)
        assert rows[0]["status"] == "cancelled"


def _make_event(**kwargs):
    from financialagents.observability.events import PipelineEvent

    defaults = dict(
        kind="node_status",
        node_name=None,
        status=None,
        update_payload=None,
        final_state=None,
        error=None,
        round=None,
        max_rounds=None,
        position_size_pct=None,
    )
    defaults.update(kwargs)
    return PipelineEvent(**defaults)


class TestLiveRenderIntegration:

    def test_run_command_uses_live_render_in_tty(
        self, tmp_signals_db, fake_pipeline, capsys, monkeypatch
    ) -> None:
        from financialagents.agents.utils.agent_states import AgentState
        from financialagents.cli.run_command import run_pipeline

        final_state: AgentState = fake_pipeline

        async def fake_stream(*args, **kwargs):
            yield _make_event(
                kind="node_status",
                node_name="technical_analyst_node",
                status="started",
            )
            yield _make_event(
                kind="node_status",
                node_name="technical_analyst_node",
                status="completed",
            )
            yield _make_event(kind="completed", final_state=final_state)

        monkeypatch.setattr(
            "financialagents.cli.run_command.stream_pipeline_events",
            fake_stream,
        )

        called = {"live": 0, "log": 0}
        from financialagents.cli import live as live_mod

        original_live = live_mod.run_with_live_render
        original_log = live_mod.run_with_log_lines

        async def spy_live(*args, **kwargs):
            called["live"] += 1
            return await original_live(*args, **kwargs)

        async def spy_log(*args, **kwargs):
            called["log"] += 1
            return await original_log(*args, **kwargs)

        monkeypatch.setattr(
            "financialagents.cli.run_command.run_with_live_render", spy_live
        )
        monkeypatch.setattr(
            "financialagents.cli.run_command.run_with_log_lines", spy_log
        )

        rc = asyncio.run(
            run_pipeline(
                ticker="RELIANCE",
                date="2026-04-26",
                json_output=False,
                provider_override="ollama",
                debug=False,
            )
        )
        assert rc == 0
        assert called["live"] == 1
        rows = _read_signals(tmp_signals_db)
        assert rows[0]["status"] == "success"

    def test_run_command_uses_log_lines_for_json(
        self, tmp_signals_db, fake_pipeline, capsys, monkeypatch
    ) -> None:
        from financialagents.cli.run_command import run_pipeline

        final_state = fake_pipeline

        async def fake_stream(*args, **kwargs):
            yield _make_event(
                kind="node_status",
                node_name="technical_analyst_node",
                status="started",
            )
            yield _make_event(
                kind="node_status",
                node_name="technical_analyst_node",
                status="completed",
            )
            yield _make_event(kind="completed", final_state=final_state)

        monkeypatch.setattr(
            "financialagents.cli.run_command.stream_pipeline_events",
            fake_stream,
        )

        called = {"live": 0, "log": 0}

        async def spy_log(events, **kwargs):
            called["log"] += 1
            last_state = None
            async for ev in events:
                if ev.kind == "completed":
                    last_state = ev.final_state
            return last_state

        async def spy_live(*args, **kwargs):  # pragma: no cover
            called["live"] += 1
            return None

        monkeypatch.setattr(
            "financialagents.cli.run_command.run_with_log_lines", spy_log
        )
        monkeypatch.setattr(
            "financialagents.cli.run_command.run_with_live_render", spy_live
        )

        rc = asyncio.run(
            run_pipeline(
                ticker="RELIANCE",
                date="2026-04-26",
                json_output=True,
                provider_override="ollama",
                debug=False,
            )
        )
        assert rc == 0
        assert called["log"] == 1
        assert called["live"] == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["ticker"] == "RELIANCE"

    def test_cancelled_run_persists_partial_payload(
        self, tmp_signals_db, fake_providers, monkeypatch
    ) -> None:
        from financialagents.agents.utils.agent_states import AgentState
        from financialagents.cli import live as live_mod
        from financialagents.cli import run_command as rc_mod
        from unittest.mock import MagicMock

        fake_graph = MagicMock()
        monkeypatch.setattr(
            "financialagents.cli.run_command.FinancialAgentsGraph",
            lambda **kw: fake_graph,
        )
        monkeypatch.setattr(
            "financialagents.cli.run_command.signals_db_path",
            lambda settings=None: tmp_signals_db,
        )

        partial = AgentState(
            ticker="RELIANCE",
            date="2026-04-26",
            trader_investment_plan="partial-tech-analysis",
        )

        async def fake_stream(*args, **kwargs):
            yield _make_event(kind="values", final_state=partial)
            yield _make_event(
                kind="node_status",
                node_name="technical_analyst_node",
                status="completed",
            )

        original_install = live_mod.install_sigint_handler

        def install_pre_set():
            ev = original_install()
            ev.set()
            return ev

        monkeypatch.setattr(
            "financialagents.cli.run_command.install_sigint_handler",
            install_pre_set,
        )
        monkeypatch.setattr(
            "financialagents.cli.run_command.stream_pipeline_events",
            fake_stream,
        )

        exit_code = asyncio.run(
            rc_mod.run_pipeline(
                ticker="RELIANCE",
                date="2026-04-26",
                json_output=False,
                provider_override="ollama",
                debug=False,
            )
        )
        assert exit_code == 130

        rows = _read_signals(tmp_signals_db)
        assert len(rows) == 1
        row = rows[0]
        assert row["status"] == "cancelled", (
            f"expected cancelled status, got {row['status']!r}"
        )
        assert row["payload_json"] is not None, (
            "P5- cancelled rows MUST persist the partial AgentState"
        )
        payload = json.loads(row["payload_json"])
        assert payload.get("trader_investment_plan") == "partial-tech-analysis"

    def test_install_sigint_handler_first_ctrl_c_sets_flag(self) -> None:
        import os as _os
        import signal as _signal

        from financialagents.cli.live import install_sigint_handler

        async def main():
            flag = install_sigint_handler()
            assert not flag.is_set()
            _os.kill(_os.getpid(), _signal.SIGINT)
            await asyncio.sleep(0.05)
            return flag.is_set()

        was_set = asyncio.run(main())
        assert was_set, "first SIGINT must set the cancel_flag"

    def test_second_ctrl_c_hard_kills(self, monkeypatch) -> None:
        import os as _os

        from financialagents.cli.live import install_sigint_handler

        recorded = []

        def record_exit(code):
            recorded.append(code)
            raise SystemExit(code)

        monkeypatch.setattr(_os, "_exit", record_exit)

        flag = install_sigint_handler()
        import signal as _signal

        _os.kill(_os.getpid(), _signal.SIGINT)
        import time as _time

        _time.sleep(0.05)
        assert flag.is_set()
        with pytest.raises(SystemExit) as exc_info:
            _os.kill(_os.getpid(), _signal.SIGINT)
            _time.sleep(0.05)
        assert exc_info.value.code == 130
        assert recorded == [130]
