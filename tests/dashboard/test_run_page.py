from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="Streamlit AppTest required",
).AppTest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_PAGE_PATH = REPO_ROOT / "financialagents" / "dashboard" / "pages" / "run.py"


def _all_widget_labels(at, attr: str) -> list[str]:
    return [getattr(w, "label", "") for w in getattr(at, attr)]


class TestRunPageEmptyState:
    def test_empty_state_heading_and_body(self) -> None:
        at = AppTest.from_file(str(RUN_PAGE_PATH)).run(timeout=10)
        assert not at.exception, (
            f"AppTest raised: {[str(e.value) for e in at.exception]}"
        )
        markdown_blobs = [m.value for m in at.markdown]
        assert any(
            "Ready when you are" in m for m in markdown_blobs
        ), f"Empty-state heading missing; got markdown: {markdown_blobs!r}"
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
            f"Sidebar Debate rounds number_input missing; "
            f"got: {sidebar_num_labels!r}"
        )
        sidebar_button_labels = _all_widget_labels(at.sidebar, "button")
        assert "Run pipeline" in sidebar_button_labels, (
            f"Sidebar Run pipeline submit button missing; "
            f"got: {sidebar_button_labels!r}"
        )

    def test_provider_options(self) -> None:
        at = AppTest.from_file(str(RUN_PAGE_PATH)).run(timeout=10)
        assert not at.exception
        provider = next(
            (s for s in at.sidebar.selectbox if s.label == "Provider"),
            None,
        )
        assert provider is not None, "Provider selectbox not found"
        assert list(provider.options) == ["openrouter", "cerebras", "groq", "ollama"], (
            f"Provider options mismatch: {provider.options!r}"
        )


class TestRunPageFormSubmission:
    def test_form_submission_sets_pending_run_marker(self) -> None:
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

    def test_quick_pick_button_sets_pending_run_marker(self) -> None:
        at = AppTest.from_file(str(RUN_PAGE_PATH)).run(timeout=10)
        assert not at.exception
        tcs_button = next((b for b in at.button if b.label == "TCS"), None)
        assert tcs_button is not None, "TCS quick-pick button not found"
        tcs_button.click()
        at.run(timeout=10)
        assert not at.exception
        assert "pending_run" in at.session_state
        assert at.session_state["pending_run"]["config"]["ticker"] == "TCS", (
            f"pending_run after TCS quick-pick: "
            f"{at.session_state['pending_run']!r}"
        )




class TestModelFieldPropagation:

    def test_user_model_input_flows_into_settings(self) -> None:
        captured: dict[str, object] = {}

        def fake_start_pipeline(graph, initial_state, config):
            captured["settings"] = config["configurable"]["settings"]

        class FakeGraph:
            def __init__(self, full_pipeline: bool = False) -> None:
                self._graph = MagicMock()

        with patch(
            "financialagents.dashboard.live_fragment.start_pipeline",
            new=fake_start_pipeline,
        ), patch(
            "financialagents.graph.trading_graph.FinancialAgentsGraph",
            new=FakeGraph,
        ), patch(
            "financialagents.storage.migrations.run_migrations",
            new=AsyncMock(return_value=None),
        ), patch(
            "financialagents.storage.signals.insert_running",
            new=AsyncMock(return_value=None),
        ):
            at = AppTest.from_file(str(RUN_PAGE_PATH)).run(timeout=10)
            assert not at.exception, (
                f"AppTest raised: {[str(e.value) for e in at.exception]}"
            )
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
            assert not at.exception, (
                f"After submit: {[str(e.value) for e in at.exception]}"
            )

        assert "settings" in captured, (
            "fake_start_pipeline was not invoked — the kickoff block "
            "did not run; check pipeline_running guard"
        )
        settings = captured["settings"]
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
        from financialagents.observability.events import PipelineEvent

        stale_event = PipelineEvent(
            kind="completed", status="completed", final_state=None,
        )
        captured_run_ids: list[str] = []

        def fake_start_pipeline(graph, initial_state, config):
            captured_run_ids.append(config["configurable"]["thread_id"])

        class FakeGraph:
            def __init__(self, full_pipeline: bool = False) -> None:
                self._graph = MagicMock()

        with patch(
            "financialagents.dashboard.live_fragment.start_pipeline",
            new=fake_start_pipeline,
        ), patch(
            "financialagents.graph.trading_graph.FinancialAgentsGraph",
            new=FakeGraph,
        ), patch(
            "financialagents.storage.migrations.run_migrations",
            new=AsyncMock(return_value=None),
        ), patch(
            "financialagents.storage.signals.insert_running",
            new=AsyncMock(return_value=None),
        ):
            at = AppTest.from_file(str(RUN_PAGE_PATH))
            at.session_state["last_completed_event"] = stale_event
            at.session_state["persisted"] = True
            at.session_state["stage_state"] = None
            at.session_state["run_id"] = "deadbeef"
            at.session_state["pipeline_running"] = False
            at.run(timeout=10)
            assert not at.exception, (
                f"Initial mount raised: {[str(e.value) for e in at.exception]}"
            )

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
            assert not at.exception, (
                f"After submit: {[str(e.value) for e in at.exception]}"
            )

        assert at.session_state["last_completed_event"] is None, (
            " last_completed_event must be reset to None before "
            f"kickoff; got {at.session_state['last_completed_event']!r}"
        )
        assert at.session_state["persisted"] is False, (
            " persisted must be reset to False before kickoff; "
            f"got {at.session_state['persisted']!r}"
        )
        assert len(captured_run_ids) == 1, (
            f"start_pipeline call count: {len(captured_run_ids)}"
        )
        assert captured_run_ids[0] != "deadbeef", (
            " kickoff must generate a fresh run_id, not reuse the "
            f"stale one; got {captured_run_ids[0]!r}"
        )
        assert at.session_state["run_id"] == captured_run_ids[0], (
            "session_state.run_id must equal the new thread_id"
        )




class TestErroredEventPersists:

    def test_errored_event_calls_update_errored(self) -> None:
        from financialagents.observability.events import PipelineEvent

        errored_event = PipelineEvent(
            kind="errored",
            status="errored",
            error=Exception("groq api_key=sk-secret-401-unauthorized"),
        )
        captured_calls: list[dict] = []

        async def fake_update_errored(
            run_id: str, *, error_msg: str,
            payload_json=None, db_path,
        ) -> None:
            captured_calls.append({
                "run_id": run_id,
                "error_msg": error_msg,
                "db_path": str(db_path),
            })

        with patch(
            "financialagents.storage.signals.update_errored",
            new=fake_update_errored,
        ):
            at = AppTest.from_file(str(RUN_PAGE_PATH))
            at.session_state["last_completed_event"] = errored_event
            at.session_state["run_id"] = "12345678"
            at.session_state["run_db_path"] = "/tmp/test_signals.db"
            at.session_state["persisted"] = False
            at.session_state["pipeline_running"] = False
            at.run(timeout=10)
            assert not at.exception, (
                f"AppTest raised: {[str(e.value) for e in at.exception]}"
            )

        assert len(captured_calls) == 1, (
            f"update_errored call count: {len(captured_calls)}; "
            f"calls: {captured_calls!r}"
        )
        call = captured_calls[0]
        assert call["run_id"] == "12345678", (
            f"update_errored run_id mismatch: {call['run_id']!r}"
        )
        assert "api_key=***" in call["error_msg"], (
            f"_redact_secrets did not redact api_key in error_msg: "
            f"{call['error_msg']!r}"
        )
        assert "sk-secret-401-unauthorized" not in call["error_msg"], (
            "raw secret leaked through to error_msg: "
            f"{call['error_msg']!r}"
        )
        assert at.session_state["persisted"] is True, (
            " persisted must flip to True after update_errored; "
            f"got {at.session_state['persisted']!r}"
        )
        error_blobs = [e.value for e in at.error]
        assert any("Pipeline errored" in blob for blob in error_blobs), (
            f"st.error panel missing; got: {error_blobs!r}"
        )
