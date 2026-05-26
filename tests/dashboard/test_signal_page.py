from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="Streamlit AppTest required",
).AppTest


REPO_ROOT = Path(__file__).resolve().parents[2]
SIGNAL_PAGE_PATH = (
    REPO_ROOT / "financialagents" / "dashboard" / "pages" / "signal.py"
)




def _build_payload_json(
    *,
    ticker: str = "RELIANCE",
    date: str = "2026-04-26",
    action: str = "BUY",
    decision: str = "approve",
    bulls: list[str] | None = None,
    bears: list[str] | None = None,
    persona_assessments: dict[str, str] | None = None,
) -> str:
    from financialagents.agents.managers.schemas import PortfolioDecision
    from financialagents.agents.risk_mgmt.schemas import PersonaAssessment
    from financialagents.agents.trader.schemas import TradingSignal
    from financialagents.agents.utils.agent_states import (
        AgentState,
        InvestDebateState,
        RiskDebateState,
    )

    sig = TradingSignal(
        action=action,
        conviction=0.78,
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
    invest_debate = None
    if bulls or bears:
        invest_debate = InvestDebateState(
            bull_history={i: t for i, t in enumerate(bulls or [])},
            bear_history={i: t for i, t in enumerate(bears or [])},
            current_round=max(len(bulls or []), len(bears or [])),
            max_rounds=3,
        )
    risk_debate = None
    if persona_assessments:
        serialised: dict[str, str] = {}
        for persona_label, main_argument in persona_assessments.items():
            pa = PersonaAssessment(
                main_argument=main_argument,
                counter_to_prior_persona="",
                recommended_position_size_pct=0.02,
                confidence=0.7,
            )
            serialised[f"{persona_label}_assessment"] = pa.model_dump_json()
        risk_debate = RiskDebateState(
            **serialised,
            current_round=1,
            synthesized_position_size_pct=0.02,
        )
    state = AgentState(
        ticker=ticker,
        date=date,
        final_trade_decision=pd_,
        investment_debate_state=invest_debate,
        risk_debate_state=risk_debate,
    )
    return state.model_dump_json(exclude={"messages"})


async def _seed_success_row(db_path: Path, run_id: str, payload_json: str) -> None:
    from financialagents.storage.migrations import run_migrations
    from financialagents.storage.signals import insert_running, update_success

    await run_migrations(db_path)
    await insert_running(
        run_id, "RELIANCE", "2026-04-26",
        provider="openrouter", model="openrouter/auto", db_path=db_path,
    )
    await update_success(
        run_id, action="BUY", confidence=0.78,
        position_size_pct=4.5, approved=True, pm_decision="approve",
        payload_json=payload_json, db_path=db_path,
    )


async def _seed_cancelled_row(db_path: Path, run_id: str) -> None:
    from financialagents.storage.migrations import run_migrations
    from financialagents.storage.signals import insert_running, update_cancelled

    await run_migrations(db_path)
    await insert_running(
        run_id, "TCS", "2026-04-26",
        provider="openrouter", model="openrouter/auto", db_path=db_path,
    )
    await update_cancelled(
        run_id, error_msg="User aborted (Ctrl+C)",
        payload_json=None, db_path=db_path,
    )


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    import streamlit as st
    st.cache_data.clear()
    yield
    st.cache_data.clear()




class TestSignalPagePermalink:
    def test_permalink(self, monkeypatch, tmp_path) -> None:
        run_id = "12345678"
        payload_json = _build_payload_json(action="BUY", decision="approve")
        monkeypatch.setenv("FA_DATA_DIR", str(tmp_path))
        asyncio.run(_seed_success_row(tmp_path / "signals.db", run_id, payload_json))

        at = AppTest.from_file(str(SIGNAL_PAGE_PATH))
        at.query_params["id"] = run_id
        at.run(timeout=10)
        assert not at.exception, (
            f"AppTest raised: {[str(e.value) for e in at.exception]}"
        )

        markdown_blobs = " ".join(m.value for m in at.markdown)
        assert ":green[BUY]" in markdown_blobs or "BUY" in markdown_blobs, (
            f"Verdict word missing from rendered markdown: {markdown_blobs!r}"
        )
        title_blobs = " ".join(t.value for t in at.title)
        assert "RELIANCE" in title_blobs, (
            f"Ticker missing from title: {title_blobs!r}"
        )


class TestSignalPageDebateColumns:
    def test_debate_columns(self, monkeypatch, tmp_path) -> None:
        run_id = "abcdef01"
        payload_json = _build_payload_json(
            bulls=["Bull says go up", "Bull again"],
            bears=["Bear says caution", "Bear again"],
            persona_assessments={
                "aggressive": "Aggressive view: high conviction.",
                "neutral": "Neutral view: balanced.",
                "conservative": "Conservative view: hedge.",
            },
        )
        monkeypatch.setenv("FA_DATA_DIR", str(tmp_path))
        asyncio.run(_seed_success_row(tmp_path / "signals.db", run_id, payload_json))

        at = AppTest.from_file(str(SIGNAL_PAGE_PATH))
        at.query_params["id"] = run_id
        at.run(timeout=10)
        assert not at.exception, (
            f"AppTest raised: {[str(e.value) for e in at.exception]}"
        )

        markdown_blobs = " ".join(m.value for m in at.markdown)
        for label in ("Bull", "Bear", "aggressive", "neutral", "conservative"):
            assert label in markdown_blobs, (
                f"Debate label {label!r} missing from rendered markdown; "
                f"got: {markdown_blobs!r}"
            )

        assert "Round 1" in markdown_blobs, (
            f"Round 1 heading missing from rendered markdown; got: {markdown_blobs!r}"
        )

        text_blobs = " ".join(
            (m.value for m in at.markdown)
        ) + " " + " ".join(t.value for t in at.text)
        for snippet in (
            "Bull says go up",
            "Bear says caution",
            "Aggressive view",
            "Neutral view",
            "Conservative view",
        ):
            assert snippet in text_blobs, (
                f"Debate snippet {snippet!r} missing; got: {text_blobs!r}"
            )


class TestSignalPageInvalidRunId:
    def test_invalid_run_id(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FA_DATA_DIR", str(tmp_path))
        at = AppTest.from_file(str(SIGNAL_PAGE_PATH))
        at.query_params["id"] = "notHEX!!"
        at.run(timeout=10)
        assert not at.exception, (
            f"AppTest raised: {[str(e.value) for e in at.exception]}"
        )

        warning_text = " ".join(w.value for w in at.warning)
        assert "No signal found with run_id" in warning_text, (
            f"Not-found copy missing; got warnings: {warning_text!r}"
        )
        assert "notHEX!!" in warning_text, (
            f"User-supplied run_id missing from not-found copy: {warning_text!r}"
        )
        assert "Try the History page." in warning_text, (
            f"Not-found tail copy missing; got: {warning_text!r}"
        )


class TestSignalPagePartialState:
    def test_no_payload_json_partial(self, monkeypatch, tmp_path) -> None:
        run_id = "cafebabe"
        monkeypatch.setenv("FA_DATA_DIR", str(tmp_path))
        asyncio.run(_seed_cancelled_row(tmp_path / "signals.db", run_id))

        at = AppTest.from_file(str(SIGNAL_PAGE_PATH))
        at.query_params["id"] = run_id
        at.run(timeout=10)
        assert not at.exception, (
            f"AppTest raised: {[str(e.value) for e in at.exception]}"
        )

        warning_text = " ".join(w.value for w in at.warning)
        assert "Run cancelled. Partial state saved." in warning_text, (
            f"Partial-state warning copy missing; got: {warning_text!r}"
        )
        expander_labels = [exp.label for exp in at.expander]
        assert "Partial state" in expander_labels, (
            f"Partial state expander missing; got expanders: {expander_labels!r}"
        )


class TestSignalPageNoIdQueryParam:
    def test_no_id_query_param(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FA_DATA_DIR", str(tmp_path))
        at = AppTest.from_file(str(SIGNAL_PAGE_PATH)).run(timeout=10)
        assert not at.exception, (
            f"AppTest raised: {[str(e.value) for e in at.exception]}"
        )

        info_text = " ".join(i.value for i in at.info)
        assert "No signal selected. Open History and click a row." in info_text, (
            f"No-id-param copy missing; got infos: {info_text!r}"
        )
