from __future__ import annotations

from pathlib import Path

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
        assert set(provider.options) == {"ollama", "groq"}, (
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
