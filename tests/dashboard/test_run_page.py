from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

AppTest = pytest.importorskip(
    "streamlit.testing.v1",
    reason="Streamlit AppTest required",
).AppTest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_PAGE_PATH = REPO_ROOT / "finagents" / "dashboard" / "pages" / "run.py"


def _all_widget_labels(at, attr: str) -> list[str]:
    return [getattr(w, "label", "") for w in getattr(at, attr)]


class TestRunPageEmptyState:
    def test_empty_state_heading_and_body(self) -> None:
        at = AppTest.from_file(str(RUN_PAGE_PATH)).run(timeout=10)
        assert not at.exception, f"AppTest raised: {[str(e.value) for e in at.exception]}"
        markdown_blobs = [m.value for m in at.markdown]
        assert any("Ready when you are" in m for m in markdown_blobs), (
            f"Empty-state heading missing; got markdown: {markdown_blobs!r}"
        )
        text_blobs = [t.value for t in at.text] + markdown_blobs
        body_haystack = " ".join(text_blobs)
        assert "Pick a ticker and run the pipeline" in body_haystack, (
            f"Empty-state body copy missing; got: {body_haystack!r}"
        )

    def test_quick_pick_buttons(self) -> None:
        at = AppTest.from_file(str(RUN_PAGE_PATH)).run(timeout=10)
        assert not at.exception
        labels = _all_widget_labels(at, "button")
        for ticker in ("RELIANCE", "TCS", "INFY"):
            assert ticker in labels, (
                f"Quick-pick button {ticker!r} missing; got buttons: {labels!r}"
            )


class TestRunPageSidebarForm:
    def test_sidebar_form(self) -> None:
        at = AppTest.from_file(str(RUN_PAGE_PATH)).run(timeout=10)
        assert not at.exception
        sidebar_text_labels = _all_widget_labels(at.sidebar, "text_input")
        assert "Ticker" in sidebar_text_labels, (
            f"Sidebar Ticker text_input missing; got: {sidebar_text_labels!r}"
        )
        assert "Model" in sidebar_text_labels, (
            f"Sidebar Model text_input missing; got: {sidebar_text_labels!r}"
        )
        sidebar_date_labels = _all_widget_labels(at.sidebar, "date_input")
        assert "Date" in sidebar_date_labels, (
            f"Sidebar Date date_input missing; got: {sidebar_date_labels!r}"
        )
        sidebar_select_labels = _all_widget_labels(at.sidebar, "selectbox")
        assert "Provider" in sidebar_select_labels, (
            f"Sidebar Provider selectbox missing; got: {sidebar_select_labels!r}"
        )
        sidebar_num_labels = _all_widget_labels(at.sidebar, "number_input")
        assert "Debate rounds" in sidebar_num_labels, (
            f"Sidebar Debate rounds number_input missing; got: {sidebar_num_labels!r}"
        )
        sidebar_button_labels = _all_widget_labels(at.sidebar, "button")
        assert "Run pipeline" in sidebar_button_labels, (
            f"Sidebar Run pipeline submit button missing; got: {sidebar_button_labels!r}"
        )

    def test_provider_options(self) -> None:
        at = AppTest.from_file(str(RUN_PAGE_PATH)).run(timeout=10)
        assert not at.exception
        provider = next(
            (s for s in at.sidebar.selectbox if s.label == "Provider"),
            None,
        )
        assert provider is not None, "Provider selectbox not found"
        assert list(provider.options) == ["openrouter", "cerebras", "groq"], (
            f"Provider options mismatch: {provider.options!r}"
        )


class TestRunPageFormSubmission:
    @pytest.fixture
    def isolated_submission(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FA_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-api-key-fake")
        with patch("finagents.dashboard.live_fragment.start_pipeline") as start_pipeline:
            yield tmp_path / "signals.db", start_pipeline

    def test_form_submission_sets_pending_run_marker(self, isolated_submission) -> None:
        at = AppTest.from_file(str(RUN_PAGE_PATH)).run(timeout=10)
        assert not at.exception
        submit = next(
            (b for b in at.sidebar.button if b.label == "Run pipeline"),
            None,
        )
        assert submit is not None, "Run pipeline submit not found"
        submit.click()
        at.run(timeout=10)
        assert not at.exception
        assert "pending_run" in at.session_state, (
            "pending_run session_state marker missing; "
            f"keys: {list(at.session_state.filtered_state.keys())!r}"
        )
        pending = at.session_state["pending_run"]
        assert isinstance(pending, dict), f"pending_run type: {type(pending)}"
        assert "config" in pending, f"pending_run keys: {list(pending.keys())!r}"
        assert pending["config"]["ticker"] == "RELIANCE", (
            f"Default ticker not RELIANCE: {pending['config']!r}"
        )
        db_path, start_pipeline = isolated_submission
        assert db_path.is_file()
        assert Path(at.session_state["run_db_path"]) == db_path
        start_pipeline.assert_called_once()

    def test_quick_pick_button_sets_pending_run_marker(self, isolated_submission) -> None:
        at = AppTest.from_file(str(RUN_PAGE_PATH)).run(timeout=10)
        assert not at.exception
        tcs_button = next((b for b in at.button if b.label == "TCS"), None)
        assert tcs_button is not None, "TCS quick-pick button not found"
        tcs_button.click()
        at.run(timeout=10)
        assert not at.exception
        assert "pending_run" in at.session_state
        assert at.session_state["pending_run"]["config"]["ticker"] == "TCS", (
            f"pending_run after TCS quick-pick: {at.session_state['pending_run']!r}"
        )
        db_path, start_pipeline = isolated_submission
        assert db_path.is_file()
        assert Path(at.session_state["run_db_path"]) == db_path
        start_pipeline.assert_called_once()




class TestModelFieldPropagation:

    def test_user_model_input_flows_into_settings(self) -> None:
        captured: dict[str, object] = {}

        def fake_start_pipeline(graph, initial_state, config):
            captured["settings"] = config["configurable"]["settings"]

        class FakeGraph:
            def __init__(self, full_pipeline: bool = False) -> None:
                self._graph = MagicMock()

            def compiled(self):
                return self._graph

        with (
            patch(
                "finagents.dashboard.live_fragment.start_pipeline",
                new=fake_start_pipeline,
            ),
            patch(
                "finagents.graph.trading_graph.FinAgentsGraph",
                new=FakeGraph,
            ),
            patch(
                "finagents.storage.migrations.run_migrations",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "finagents.storage.signals.insert_running",
                new=AsyncMock(return_value=None),
            ),
        ):
            at = AppTest.from_file(str(RUN_PAGE_PATH)).run(timeout=10)
            assert not at.exception, f"AppTest raised: {[str(e.value) for e in at.exception]}"
            model_input = next(
                (t for t in at.sidebar.text_input if t.label == "Model"),
                None,
            )
            assert model_input is not None, "Model text_input not found"
            model_input.set_value("groq:llama-3.1-405b")
            submit = next(
                (b for b in at.sidebar.button if b.label == "Run pipeline"),
                None,
            )
            assert submit is not None, "Run pipeline submit not found"
            submit.click()
            at.run(timeout=10)
            assert not at.exception, f"After submit: {[str(e.value) for e in at.exception]}"

        assert "settings" in captured, (
            "fake_start_pipeline was not invoked — the kickoff block "
            "did not run; check pipeline_running guard"
        )
        from finagents.config import Settings

        settings = captured["settings"]
        assert isinstance(settings, Settings)
        assert settings.deep_think_llm == "groq:llama-3.1-405b", (
            " settings.deep_think_llm must reflect user-selected "
            f"Model, got {settings.deep_think_llm!r}"
        )
        assert settings.quick_think_llm == "groq:llama-3.1-405b", (
            " settings.quick_think_llm must also reflect the "
            "user-selected Model (analyst nodes route through the "
            f"quick tier per, got {settings.quick_think_llm!r}"
        )




class TestResubmissionStateReset:

    def test_consecutive_runs_reset_session_state(self) -> None:
        from finagents.observability.events import PipelineEvent

        stale_event = PipelineEvent(
            kind="completed",
            status="completed",
            final_state=None,
        )
        captured_run_ids: list[str] = []

        def fake_start_pipeline(graph, initial_state, config):
            captured_run_ids.append(config["configurable"]["thread_id"])

        class FakeGraph:
            def __init__(self, full_pipeline: bool = False) -> None:
                self._graph = MagicMock()

            def compiled(self):
                return self._graph

        with (
            patch(
                "finagents.dashboard.live_fragment.start_pipeline",
                new=fake_start_pipeline,
            ),
            patch(
                "finagents.graph.trading_graph.FinAgentsGraph",
                new=FakeGraph,
            ),
            patch(
                "finagents.storage.migrations.run_migrations",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "finagents.storage.signals.insert_running",
                new=AsyncMock(return_value=None),
            ),
        ):
            at = AppTest.from_file(str(RUN_PAGE_PATH))
            at.session_state["last_completed_event"] = stale_event
            at.session_state["persisted"] = True
            at.session_state["stage_state"] = None
            at.session_state["run_id"] = "deadbeef"
            at.session_state["pipeline_running"] = False
            at.run(timeout=10)
            assert not at.exception, f"Initial mount raised: {[str(e.value) for e in at.exception]}"

            ticker_input = next(
                (t for t in at.sidebar.text_input if t.label == "Ticker"),
                None,
            )
            assert ticker_input is not None, "Ticker text_input not found"
            ticker_input.set_value("TCS")
            submit = next(
                (b for b in at.sidebar.button if b.label == "Run pipeline"),
                None,
            )
            assert submit is not None, "Run pipeline submit not found"
            submit.click()
            at.run(timeout=10)
            assert not at.exception, f"After submit: {[str(e.value) for e in at.exception]}"

        assert at.session_state["last_completed_event"] is None, (
            " last_completed_event must be reset to None before "
            f"kickoff; got {at.session_state['last_completed_event']!r}"
        )
        assert at.session_state["persisted"] is False, (
            " persisted must be reset to False before kickoff; "
            f"got {at.session_state['persisted']!r}"
        )
        assert len(captured_run_ids) == 1, f"start_pipeline call count: {len(captured_run_ids)}"
        assert captured_run_ids[0] != "deadbeef", (
            " kickoff must generate a fresh run_id, not reuse the "
            f"stale one; got {captured_run_ids[0]!r}"
        )
        assert at.session_state["run_id"] == captured_run_ids[0], (
            "session_state.run_id must equal the new thread_id"
        )




class TestErroredEventPersists:

    def test_errored_event_calls_update_errored(self) -> None:
        from finagents.observability.events import PipelineEvent

        errored_event = PipelineEvent(
            kind="errored",
            status="errored",
            error=Exception("groq api_key=sk-secret-401-unauthorized"),
        )
        captured_calls: list[dict] = []

        async def fake_update_errored(
            run_id: str,
            *,
            error_msg: str,
            payload_json=None,
            db_path,
        ) -> None:
            captured_calls.append(
                {
                    "run_id": run_id,
                    "error_msg": error_msg,
                    "db_path": str(db_path),
                }
            )

        with patch(
            "finagents.storage.signals.update_errored",
            new=fake_update_errored,
        ):
            at = AppTest.from_file(str(RUN_PAGE_PATH))
            at.session_state["last_completed_event"] = errored_event
            at.session_state["run_id"] = "12345678"
            at.session_state["run_db_path"] = "/tmp/test_signals.db"
            at.session_state["persisted"] = False
            at.session_state["pipeline_running"] = False
            at.run(timeout=10)
            assert not at.exception, f"AppTest raised: {[str(e.value) for e in at.exception]}"

        assert len(captured_calls) == 1, (
            f"update_errored call count: {len(captured_calls)}; calls: {captured_calls!r}"
        )
        call = captured_calls[0]
        assert call["run_id"] == "12345678", f"update_errored run_id mismatch: {call['run_id']!r}"
        assert "api_key=***" in call["error_msg"], (
            f"_redact_secrets did not redact api_key in error_msg: {call['error_msg']!r}"
        )
        assert "sk-secret-401-unauthorized" not in call["error_msg"], (
            f"raw secret leaked through to error_msg: {call['error_msg']!r}"
        )
        assert at.session_state["persisted"] is True, (
            " persisted must flip to True after update_errored; "
            f"got {at.session_state['persisted']!r}"
        )
        error_blobs = [e.value for e in at.error]
        assert any("Pipeline errored" in blob for blob in error_blobs), (
            f"st.error panel missing; got: {error_blobs!r}"
        )

    def test_errored_event_carries_real_exception_to_dashboard(self) -> None:
        from finagents.observability.events import PipelineEvent

        errored_event = PipelineEvent(
            kind="errored",
            status="errored",
            error=RuntimeError("boom: 260526-no0-marker"),
        )
        captured_calls: list[dict] = []

        async def fake_update_errored(
            run_id: str,
            *,
            error_msg: str,
            payload_json=None,
            db_path,
        ) -> None:
            captured_calls.append({"error_msg": error_msg})

        with patch(
            "finagents.storage.signals.update_errored",
            new=fake_update_errored,
        ):
            at = AppTest.from_file(str(RUN_PAGE_PATH))
            at.session_state["last_completed_event"] = errored_event
            at.session_state["run_id"] = "aabbccdd"
            at.session_state["run_db_path"] = "/tmp/test_signals.db"
            at.session_state["persisted"] = False
            at.session_state["pipeline_running"] = False
            at.run(timeout=10)
            assert not at.exception, f"AppTest raised: {[str(e.value) for e in at.exception]}"

        assert len(captured_calls) == 1, f"update_errored call count: {len(captured_calls)}"
        error_msg = captured_calls[0]["error_msg"]
        assert "boom" in error_msg, f"'boom' missing from error_msg: {error_msg!r}"
        assert "260526-no0-marker" in error_msg, f"marker missing from error_msg: {error_msg!r}"
        assert "event stream ended without completion" not in error_msg, (
            f"generic fallback string leaked into error_msg: {error_msg!r}"
        )
        error_blobs = [e.value for e in at.error]
        assert any("boom" in blob and "260526-no0-marker" in blob for blob in error_blobs), (
            f"st.error panel missing marker; got: {error_blobs!r}"
        )




class TestCompletedWithNoDecisionPersistsErrored:

    def test_completed_event_with_none_decision_calls_update_errored(
        self,
    ) -> None:
        from finagents.agents.utils.agent_states import AgentState
        from finagents.observability.events import PipelineEvent

        partial_state = AgentState(ticker="RELIANCE", date="2026-04-26")
        assert partial_state.final_trade_decision is None

        completed_event = PipelineEvent(
            kind="completed",
            status="completed",
            final_state=partial_state,
        )

        errored_calls: list[dict] = []
        success_calls: list[dict] = []

        async def fake_update_errored(
            run_id: str,
            *,
            error_msg: str,
            payload_json=None,
            db_path,
        ) -> None:
            errored_calls.append(
                {
                    "run_id": run_id,
                    "error_msg": error_msg,
                    "payload_json": payload_json,
                    "db_path": str(db_path),
                }
            )

        async def fake_update_success(
            run_id: str,
            *,
            action,
            confidence,
            position_size_pct,
            approved,
            pm_decision,
            payload_json,
            db_path,
        ) -> None:
            success_calls.append(
                {
                    "run_id": run_id,
                    "action": action,
                    "approved": approved,
                    "pm_decision": pm_decision,
                }
            )

        with (
            patch(
                "finagents.storage.signals.update_errored",
                new=fake_update_errored,
            ),
            patch(
                "finagents.storage.signals.update_success",
                new=fake_update_success,
            ),
        ):
            at = AppTest.from_file(str(RUN_PAGE_PATH))
            at.session_state["last_completed_event"] = completed_event
            at.session_state["run_id"] = "deadbeef"
            at.session_state["run_db_path"] = "/tmp/test_signals.db"
            at.session_state["persisted"] = False
            at.session_state["pipeline_running"] = False
            at.run(timeout=10)
            assert not at.exception, f"AppTest raised: {[str(e.value) for e in at.exception]}"

        assert success_calls == [], (
            f"update_success was called on a degenerate completion; calls: {success_calls!r}"
        )
        assert len(errored_calls) == 1, (
            f"update_errored call count: {len(errored_calls)}; calls: {errored_calls!r}"
        )
        call = errored_calls[0]
        assert call["run_id"] == "deadbeef", f"run_id mismatch: {call['run_id']!r}"
        assert (
            "pipeline completed without producing" in call["error_msg"].lower()
            or "no final trade decision" in call["error_msg"].lower()
        ), f"unexpected error_msg: {call['error_msg']!r}"
        assert call["payload_json"] is not None, (
            "payload_json must be passed through so partial state is "
            "preserved (update_errored's COALESCE pattern)"
        )
        assert at.session_state["persisted"] is True, (
            "persisted must flip to True after update_errored; "
            f"got {at.session_state['persisted']!r}"
        )

    def test_completed_event_with_decision_still_calls_update_success(
        self,
    ) -> None:
        from finagents.agents.managers.schemas import PortfolioDecision
        from finagents.agents.trader.schemas import TradingSignal
        from finagents.agents.utils.agent_states import AgentState
        from finagents.observability.events import PipelineEvent

        ts = TradingSignal(
            action="BUY",
            conviction=0.7,
            entry_price=1234.0,
            stop_loss=1180.0,
            target_1=1310.0,
            target_2=1380.0,
            position_size_pct=0.03,
            position_size_shares=10,
            risk_reward_ratio=1.45,
            rationale="t",
            analyst_weight_applied={
                "technical": 0.303,
                "fundamentals": 0.455,
                "news": 0.242,
            },
            sizing_method="atr_based",
            time_horizon="positional",
            t2t_flag=False,
        )
        pd_ = PortfolioDecision(
            decision="approve",
            approved=True,
            pm_notes="ok",
            signal=ts,
        )
        success_state = AgentState(
            ticker="RELIANCE",
            date="2026-04-26",
            final_trade_decision=pd_,
        )
        completed_event = PipelineEvent(
            kind="completed",
            status="completed",
            final_state=success_state,
        )

        errored_calls: list[dict] = []
        success_calls: list[dict] = []

        async def fake_update_errored(
            run_id: str,
            *,
            error_msg: str,
            payload_json=None,
            db_path,
        ) -> None:
            errored_calls.append({"error_msg": error_msg})

        async def fake_update_success(
            run_id: str,
            *,
            action,
            confidence,
            position_size_pct,
            approved,
            pm_decision,
            payload_json,
            db_path,
        ) -> None:
            success_calls.append(
                {
                    "run_id": run_id,
                    "action": action,
                    "approved": approved,
                    "pm_decision": pm_decision,
                }
            )

        with (
            patch(
                "finagents.storage.signals.update_errored",
                new=fake_update_errored,
            ),
            patch(
                "finagents.storage.signals.update_success",
                new=fake_update_success,
            ),
        ):
            at = AppTest.from_file(str(RUN_PAGE_PATH))
            at.session_state["last_completed_event"] = completed_event
            at.session_state["run_id"] = "cafebabe"
            at.session_state["run_db_path"] = "/tmp/test_signals.db"
            at.session_state["persisted"] = False
            at.session_state["pipeline_running"] = False
            at.run(timeout=10)
            assert not at.exception, f"AppTest raised: {[str(e.value) for e in at.exception]}"

        assert errored_calls == [], (
            f"update_errored was called on a valid success completion; calls: {errored_calls!r}"
        )
        assert len(success_calls) == 1, (
            f"update_success call count: {len(success_calls)}; calls: {success_calls!r}"
        )
        call = success_calls[0]
        assert call["run_id"] == "cafebabe"
        assert call["action"] == "BUY"
        assert call["approved"] is True
        assert call["pm_decision"] == "approve"




class TestEmptyRunDbPathFailsClosed:

    def test_completed_event_with_empty_db_path_skips_write(self) -> None:
        from finagents.observability.events import PipelineEvent
        from finagents.agents.utils.agent_states import AgentState

        partial_state = AgentState(ticker="RELIANCE", date="2026-04-26")
        completed_event = PipelineEvent(
            kind="completed",
            status="completed",
            final_state=partial_state,
        )

        success_calls: list[dict] = []
        errored_calls: list[dict] = []

        async def fake_update_success(**kwargs) -> None:
            success_calls.append(kwargs)

        async def fake_update_errored(**kwargs) -> None:
            errored_calls.append(kwargs)

        with (
            patch(
                "finagents.storage.signals.update_success",
                new=fake_update_success,
            ),
            patch(
                "finagents.storage.signals.update_errored",
                new=fake_update_errored,
            ),
        ):
            at = AppTest.from_file(str(RUN_PAGE_PATH))
            at.session_state["last_completed_event"] = completed_event
            at.session_state["run_id"] = "deadbeef"
            at.session_state["run_db_path"] = ""
            at.session_state["persisted"] = False
            at.session_state["pipeline_running"] = False
            at.run(timeout=10)
            assert not at.exception, f"AppTest raised: {[str(e.value) for e in at.exception]}"

        assert success_calls == [], (
            f"update_success called with empty run_db_path: {success_calls!r}"
        )
        assert errored_calls == [], (
            f"update_errored called with empty run_db_path: {errored_calls!r}"
        )
        assert at.session_state["persisted"] is False, (
            "persisted must remain False when no write happened"
        )

    def test_errored_event_with_empty_db_path_skips_write(self) -> None:
        from finagents.observability.events import PipelineEvent

        errored_event = PipelineEvent(
            kind="errored",
            status="errored",
            error=RuntimeError("boom"),
        )

        errored_calls: list[dict] = []

        async def fake_update_errored(**kwargs) -> None:
            errored_calls.append(kwargs)

        with patch(
            "finagents.storage.signals.update_errored",
            new=fake_update_errored,
        ):
            at = AppTest.from_file(str(RUN_PAGE_PATH))
            at.session_state["last_completed_event"] = errored_event
            at.session_state["run_id"] = "deadbeef"
            at.session_state["run_db_path"] = ""
            at.session_state["persisted"] = False
            at.session_state["pipeline_running"] = False
            at.run(timeout=10)
            assert not at.exception, f"AppTest raised: {[str(e.value) for e in at.exception]}"

        assert errored_calls == [], (
            f"update_errored called with empty run_db_path: {errored_calls!r}"
        )
        assert at.session_state["persisted"] is False, (
            "persisted must remain False when no write happened"
        )




class TestSessionResetCancelsStaleProducer:

    def test_session_reset_cancels_prior_producer_before_new_kickoff(
        self,
    ) -> None:
        import threading

        call_order: list[str] = []
        cancel_event_observed: dict[str, object] = {}

        def fake_cancel_pipeline() -> None:
            call_order.append("cancel_pipeline")

        def fake_start_pipeline(graph, initial_state, config):
            call_order.append("start_pipeline")
            cancel_event_observed["thread_id"] = config["configurable"]["thread_id"]

        class FakeGraph:
            def __init__(self, full_pipeline: bool = False) -> None:
                self._graph = MagicMock()

            def compiled(self):
                return self._graph

        stale_cancel_event = threading.Event()
        stale_thread = MagicMock(spec=threading.Thread)
        stale_thread.is_alive.return_value = True

        with (
            patch(
                "finagents.dashboard.live_fragment.start_pipeline",
                new=fake_start_pipeline,
            ),
            patch(
                "finagents.dashboard.live_fragment.cancel_pipeline",
                new=fake_cancel_pipeline,
            ),
            patch(
                "finagents.graph.trading_graph.FinAgentsGraph",
                new=FakeGraph,
            ),
            patch(
                "finagents.storage.migrations.run_migrations",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "finagents.storage.signals.insert_running",
                new=AsyncMock(return_value=None),
            ),
        ):
            at = AppTest.from_file(str(RUN_PAGE_PATH))
            at.session_state["pipeline_running"] = False
            at.session_state["pipeline_cancel"] = stale_cancel_event
            at.session_state["pipeline_thread"] = stale_thread
            at.session_state["run_id"] = "stale_run"
            at.session_state["stage_state"] = None
            at.session_state["persisted"] = True
            at.session_state["last_completed_event"] = None
            at.run(timeout=10)
            assert not at.exception, f"Initial mount raised: {[str(e.value) for e in at.exception]}"

            ticker_input = next(
                (t for t in at.sidebar.text_input if t.label == "Ticker"),
                None,
            )
            assert ticker_input is not None, "Ticker text_input not found"
            ticker_input.set_value("TCS")
            submit = next(
                (b for b in at.sidebar.button if b.label == "Run pipeline"),
                None,
            )
            assert submit is not None, "Run pipeline submit not found"
            submit.click()
            at.run(timeout=10)
            assert not at.exception, f"After submit: {[str(e.value) for e in at.exception]}"

        assert "cancel_pipeline" in call_order, (
            "cancel_pipeline was not invoked despite a stale producer "
            f"handle being present; call_order: {call_order!r}"
        )
        assert "start_pipeline" in call_order, (
            f"start_pipeline was not invoked; call_order: {call_order!r}"
        )
        cancel_idx = call_order.index("cancel_pipeline")
        start_idx = call_order.index("start_pipeline")
        assert cancel_idx < start_idx, (
            "cancel_pipeline must run BEFORE start_pipeline in the "
            f"kickoff branch; got call_order: {call_order!r}"
        )

    def test_no_stale_producer_means_no_cancel_call(self) -> None:
        call_order: list[str] = []

        def fake_cancel_pipeline() -> None:
            call_order.append("cancel_pipeline")

        def fake_start_pipeline(graph, initial_state, config):
            call_order.append("start_pipeline")

        class FakeGraph:
            def __init__(self, full_pipeline: bool = False) -> None:
                self._graph = MagicMock()

            def compiled(self):
                return self._graph

        with (
            patch(
                "finagents.dashboard.live_fragment.start_pipeline",
                new=fake_start_pipeline,
            ),
            patch(
                "finagents.dashboard.live_fragment.cancel_pipeline",
                new=fake_cancel_pipeline,
            ),
            patch(
                "finagents.graph.trading_graph.FinAgentsGraph",
                new=FakeGraph,
            ),
            patch(
                "finagents.storage.migrations.run_migrations",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "finagents.storage.signals.insert_running",
                new=AsyncMock(return_value=None),
            ),
        ):
            at = AppTest.from_file(str(RUN_PAGE_PATH))
            at.session_state["pipeline_running"] = False
            at.run(timeout=10)
            assert not at.exception, f"Initial mount raised: {[str(e.value) for e in at.exception]}"

            submit = next(
                (b for b in at.sidebar.button if b.label == "Run pipeline"),
                None,
            )
            assert submit is not None, "Run pipeline submit not found"
            submit.click()
            at.run(timeout=10)
            assert not at.exception, f"After submit: {[str(e.value) for e in at.exception]}"

        assert "cancel_pipeline" not in call_order, (
            f"cancel_pipeline was invoked on a clean first run; call_order: {call_order!r}"
        )
        assert "start_pipeline" in call_order, (
            f"start_pipeline missing from call_order: {call_order!r}"
        )
