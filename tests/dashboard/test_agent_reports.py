from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def agent_reports_app(tmp_path):
    def _factory(populate_reports: bool = True) -> Path:
        if populate_reports:
            state_block = (
                "from financialagents.agents.analysts.schemas import (\n"
                " TechnicalReport, FundamentalsAnalystReport, NewsReport,\n"
                ")\n"
                "tech = TechnicalReport(\n"
                " rsi=58.0, current_price=2500.0, confidence=0.7,\n"
                " notes='Tech notes.',\n"
                ")\n"
                "fund = FundamentalsAnalystReport(\n"
                " pe_ratio=22.5, confidence=0.6,\n"
                " fundamental_assessment='Fundamentals OK.',\n"
                ")\n"
                "news = NewsReport(\n"
                " article_count=5, sentiment_direction='positive',\n"
                " confidence=0.8,\n"
                " news_assessment='News positive.',\n"
                ")\n"
                "state = AgentState(\n"
                " ticker='RELIANCE', date='2026-04-27',\n"
                " technical_report=tech,\n"
                " fundamentals_report=fund,\n"
                " news_report=news,\n"
                ")\n"
            )
        else:
            state_block = (
                "state = AgentState(ticker='RELIANCE', date='2026-04-27')\n"
            )
        app_src = (
            "import streamlit as st\n"
            "from financialagents.agents.utils.agent_states import AgentState\n"
            "from financialagents.dashboard.agent_reports import "
            "render_agent_reports\n"
            f"{state_block}"
            "render_agent_reports(state)\n"
        )
        path = tmp_path / f"agent_reports_{'full' if populate_reports else 'empty'}.py"
        path.write_text(app_src)
        return path
    return _factory


class TestAgentReports:
    def test_three_columns(self, agent_reports_app) -> None:
        AppTest = pytest.importorskip(
            "streamlit.testing.v1",
            reason="Streamlit AppTest required",
        ).AppTest
        at = AppTest.from_file(
            str(agent_reports_app(populate_reports=True))
        ).run(timeout=10)
        assert not at.exception, (
            f"AppTest raised: {[str(e.value) for e in at.exception]}"
        )
        markdown_blobs = [m.value for m in at.markdown]
        haystack = " ".join(markdown_blobs)
        for required in ("Technical", "Fundamentals", "News"):
            assert required in haystack, (
                f"Missing analyst label {required!r}; got: {haystack!r}"
            )

    def test_missing_reports_render_dash(self, agent_reports_app) -> None:
        AppTest = pytest.importorskip(
            "streamlit.testing.v1",
            reason="Streamlit AppTest required",
        ).AppTest
        at = AppTest.from_file(
            str(agent_reports_app(populate_reports=False))
        ).run(timeout=10)
        assert not at.exception, (
            f"AppTest raised: {[str(e.value) for e in at.exception]}"
        )
        all_text = " ".join(m.value for m in at.markdown) + " ".join(
            t.value for t in at.text
        )
        assert "—" in all_text or "-" in all_text, (
            f"Expected em-dash placeholder; got: {all_text!r}"
        )
