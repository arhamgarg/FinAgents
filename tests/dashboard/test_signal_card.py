from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_signal_state(
    action: Literal["BUY", "SELL", "HOLD"] = "BUY",
    decision: Literal["approve", "modify", "veto"] = "approve",
) -> object:
    from finagents.agents.managers.schemas import PortfolioDecision
    from finagents.agents.trader.schemas import TradingSignal
    from finagents.agents.utils.agent_states import AgentState

    sig = TradingSignal(
        action=action,
        conviction=0.72,
        entry_price=2500.0,
        stop_loss=2400.0,
        target_1=2700.0,
        position_size_pct=0.045,
        rationale="Test rationale.",
    )
    pd_ = PortfolioDecision(
        signal=sig,
        approved=(decision == "approve"),
        decision=decision,
        pm_notes="PM rationale here.",
    )
    return AgentState(
        ticker="RELIANCE",
        date="2026-04-27",
        final_trade_decision=pd_,
    )


@pytest.fixture
def signal_card_app_factory(tmp_path):

    def _factory(action: str = "BUY", decision: str = "approve") -> Path:
        app_src = (
            "import streamlit as st\n"
            "from finagents.agents.managers.schemas import "
            "PortfolioDecision\n"
            "from finagents.agents.trader.schemas import TradingSignal\n"
            "from finagents.agents.utils.agent_states import AgentState\n"
            "from finagents.dashboard.signal_card import "
            "render_signal_card\n"
            "sig = TradingSignal(\n"
            f"    action={action!r},\n"
            "    conviction=0.72,\n"
            "    entry_price=2500.0,\n"
            "    stop_loss=2400.0,\n"
            "    target_1=2700.0,\n"
            "    position_size_pct=0.045,\n"
            "    rationale='Test rationale.',\n"
            ")\n"
            "pd_ = PortfolioDecision(\n"
            "    signal=sig,\n"
            f"    approved={decision == 'approve'},\n"
            f"    decision={decision!r},\n"
            "    pm_notes='PM rationale here.',\n"
            ")\n"
            "state = AgentState(\n"
            "    ticker='RELIANCE',\n"
            "    date='2026-04-27',\n"
            "    final_trade_decision=pd_,\n"
            ")\n"
            "render_signal_card(state)\n"
        )
        path = tmp_path / f"signal_card_{action.lower()}_{decision}.py"
        path.write_text(app_src)
        return path

    return _factory


class TestVerdictColorHelper:
    def test_verdict_color_mapping_table(self) -> None:
        from finagents.dashboard.signal_card import verdict_color

        assert verdict_color("BUY") == "#2EA043"
        assert verdict_color("SELL") == "#F85149"
        assert verdict_color("HOLD") == "#D29922"


class TestSignalCardVerdictColors:
    def _render_and_collect_markdown(self, app_path: Path) -> str:
        app_test = pytest.importorskip(
            "streamlit.testing.v1",
            reason="Streamlit AppTest required",
        ).AppTest
        at = app_test.from_file(str(app_path)).run(timeout=10)
        assert not at.exception, f"AppTest raised: {[str(e.value) for e in at.exception]}"
        return " ".join(m.value for m in at.markdown)

    def test_verdict_color_mapping_buy(self, signal_card_app_factory) -> None:
        text = self._render_and_collect_markdown(
            signal_card_app_factory(action="BUY", decision="approve")
        )
        assert ":green[BUY]" in text or "green[BUY]" in text, (
            f"BUY verdict missing :green[] colour token; markdown: {text!r}"
        )

    def test_verdict_color_mapping_sell(self, signal_card_app_factory) -> None:
        text = self._render_and_collect_markdown(
            signal_card_app_factory(action="SELL", decision="veto")
        )
        assert ":red[SELL]" in text or "red[SELL]" in text, (
            f"SELL verdict missing :red[] colour token; markdown: {text!r}"
        )

    def test_verdict_color_mapping_hold(self, signal_card_app_factory) -> None:
        text = self._render_and_collect_markdown(
            signal_card_app_factory(action="HOLD", decision="modify")
        )
        assert ":orange[HOLD]" in text or "orange[HOLD]" in text, (
            f"HOLD verdict missing :orange[] colour token; markdown: {text!r}"
        )


class TestSignalCardNoUnsafeHtml:
    def test_no_unsafe_html(self) -> None:
        import ast

        dashboard_dir = REPO_ROOT / "src" / "finagents" / "dashboard"
        offenders: list[tuple[Path, int]] = []

        class _UnsafeHtmlVisitor(ast.NodeVisitor):
            def __init__(self, py_path: Path) -> None:
                self.py_path = py_path
                self.hits: list[int] = []

            def visit_Call(self, node: ast.Call) -> None:
                for kw in node.keywords:
                    if (
                        kw.arg == "unsafe_allow_html"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
                        self.hits.append(node.lineno)
                self.generic_visit(node)

        for py in dashboard_dir.rglob("*.py"):
            tree = ast.parse(py.read_text())
            visitor = _UnsafeHtmlVisitor(py)
            visitor.visit(tree)
            for lineno in visitor.hits:
                offenders.append((py, lineno))

        assert not offenders, f"unsafe_allow_html=True call found in dashboard/: {offenders!r}"


class TestSignalCardLayout:
    def test_4_metric_tiles(self, signal_card_app_factory) -> None:
        app_test = pytest.importorskip(
            "streamlit.testing.v1",
            reason="Streamlit AppTest required",
        ).AppTest
        at = app_test.from_file(str(signal_card_app_factory(action="BUY", decision="approve"))).run(
            timeout=10
        )
        assert not at.exception
        labels = [m.label for m in at.metric]
        for required in ("Entry", "Stop", "Target", "Size"):
            assert required in labels, f"Missing st.metric label {required!r}; got: {labels!r}"

    def test_3_collapsed_expanders(self, signal_card_app_factory) -> None:
        app_test = pytest.importorskip(
            "streamlit.testing.v1",
            reason="Streamlit AppTest required",
        ).AppTest
        at = app_test.from_file(str(signal_card_app_factory(action="BUY", decision="approve"))).run(
            timeout=10
        )
        assert not at.exception
        labels = [exp.label for exp in at.expander]
        for required in ("Bull / Bear debate", "Risk personas", "Agent reports"):
            assert required in labels, f"Missing expander {required!r}; got: {labels!r}"


class TestSignalCardMissingDecision:
    def test_no_decision_warning(self, tmp_path) -> None:
        app_test = pytest.importorskip(
            "streamlit.testing.v1",
            reason="Streamlit AppTest required",
        ).AppTest
        app_src = (
            "import streamlit as st\n"
            "from finagents.agents.utils.agent_states import AgentState\n"
            "from finagents.dashboard.signal_card import "
            "render_signal_card\n"
            "state = AgentState(ticker='RELIANCE', date='2026-04-27')\n"
            "render_signal_card(state)\n"
        )
        path = tmp_path / "signal_card_none.py"
        path.write_text(app_src)
        at = app_test.from_file(str(path)).run(timeout=10)
        assert not at.exception, f"AppTest raised: {[str(e.value) for e in at.exception]}"
        warning_text = " ".join(w.value for w in at.warning)
        assert "PortfolioDecision" in warning_text or "no" in warning_text.lower(), (
            f"Expected no-decision warning; got: {warning_text!r}"
        )
