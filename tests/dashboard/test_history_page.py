from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

AppTest = pytest.importorskip(
    "streamlit.testing.v1",
    reason="Streamlit AppTest required",
).AppTest


REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY_PAGE_PATH = REPO_ROOT / "finagents" / "dashboard" / "pages" / "history.py"




async def _seed_three_rows(db_path: Path) -> list[str]:
    from finagents.storage.migrations import run_migrations
    from finagents.storage.signals import (
        insert_running,
        update_cancelled,
        update_errored,
        update_success,
    )

    await run_migrations(db_path)

    await insert_running(
        "aaaaaaaa",
        "RELIANCE",
        "2026-04-26",
        provider="openrouter",
        model="openrouter/auto",
        db_path=db_path,
    )
    await update_success(
        "aaaaaaaa",
        action="BUY",
        confidence=0.78,
        position_size_pct=3.0,
        approved=True,
        pm_decision="approve",
        payload_json='{"ticker":"RELIANCE","date":"2026-04-26"}',
        db_path=db_path,
    )

    await insert_running(
        "bbbbbbbb",
        "TCS",
        "2026-04-26",
        provider="openrouter",
        model="openrouter/auto",
        db_path=db_path,
    )
    await update_cancelled(
        "bbbbbbbb",
        error_msg="User aborted (Ctrl+C)",
        payload_json=None,
        db_path=db_path,
    )

    await insert_running(
        "cccccccc",
        "INFY",
        "2026-04-26",
        provider="openrouter",
        model="openrouter/auto",
        db_path=db_path,
    )
    await update_errored(
        "cccccccc",
        error_msg="LLM connection refused",
        payload_json=None,
        db_path=db_path,
    )

    return ["aaaaaaaa", "bbbbbbbb", "cccccccc"]


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    import streamlit as st

    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def history_app(monkeypatch, tmp_path) -> AppTest:
    monkeypatch.setenv("FA_DATA_DIR", str(tmp_path))
    return AppTest.from_file(str(HISTORY_PAGE_PATH))




class TestHistoryPageDefaultTable:
    def test_default_table(self, history_app, tmp_path) -> None:
        asyncio.run(_seed_three_rows(tmp_path / "signals.db"))
        at = history_app.run(timeout=10)
        assert not at.exception, f"AppTest raised: {[str(e.value) for e in at.exception]}"

        dataframes = list(at.dataframe)
        assert len(dataframes) == 1, f"Expected exactly 1 dataframe; got {len(dataframes)}"
        df_proxy = dataframes[0]
        df = df_proxy.value
        assert len(df) == 3, f"Expected 3 rows; got {len(df)} rows"

        import json

        cfg_str = df_proxy.proto.columns
        assert cfg_str, "Dataframe proto.columns is empty (no column_config)"
        cfg = json.loads(cfg_str)
        for hidden in ("run_id", "payload_json", "provider", "model"):
            assert cfg.get(hidden, {}).get("hidden") is True, (
                f"Column {hidden!r} should be hidden; got: {cfg.get(hidden)!r}"
            )
        expected_visible = {
            "created_at": ("Created", "medium", "datetime"),
            "ticker": ("Ticker", "small", "text"),
            "confidence": ("Confidence", "small", "number"),
            "position_size_pct": ("Size %", "small", "number"),
            "approved": ("Approved", "small", "checkbox"),
            "status": ("Status", "small", "text"),
        }
        for col, (label, width, type_) in expected_visible.items():
            spec = cfg.get(col)
            assert spec, f"Visible column {col!r} missing from column_config"
            assert spec.get("label") == label, (
                f"{col!r} label mismatch: {spec.get('label')!r} vs {label!r}"
            )
            assert spec.get("width") == width, (
                f"{col!r} width mismatch: {spec.get('width')!r} vs {width!r}"
            )
            assert spec.get("type_config", {}).get("type") == type_, (
                f"{col!r} type mismatch: {spec.get('type_config')!r}"
            )
        assert cfg["confidence"]["type_config"].get("format") == "%.2f"
        assert cfg["position_size_pct"]["type_config"].get("format") == "%.1f%%"


class TestHistoryPageRefresh:
    def test_refresh(self, history_app, tmp_path, monkeypatch) -> None:
        asyncio.run(_seed_three_rows(tmp_path / "signals.db"))
        at = history_app.run(timeout=10)
        assert not at.exception

        cleared = {"count": 0}
        import streamlit as st

        original_clear = st.cache_data.clear

        def _tracking_clear() -> None:
            cleared["count"] += 1
            original_clear()

        monkeypatch.setattr(st.cache_data, "clear", _tracking_clear)

        refresh = next(
            (b for b in at.button if b.label == "Refresh history"),
            None,
        )
        assert refresh is not None, (
            f"Refresh history button missing; got: {[b.label for b in at.button]!r}"
        )
        refresh.click()
        at.run(timeout=10)
        assert not at.exception
        assert cleared["count"] >= 1, (
            "st.cache_data.clear() was not called after Refresh history click"
        )


class TestHistoryPageStatusChip:
    def test_status_chip(self, history_app, tmp_path) -> None:
        asyncio.run(_seed_three_rows(tmp_path / "signals.db"))
        at = history_app.run(timeout=10)
        assert not at.exception

        df = at.dataframe[0].value
        statuses = list(df["status"])
        assert any("success" in str(s) for s in statuses), (
            f"Missing 'success' chip in status column: {statuses!r}"
        )
        assert any("cancelled" in str(s) for s in statuses), (
            f"Missing 'cancelled' chip in status column: {statuses!r}"
        )
        assert any("errored" in str(s) for s in statuses), (
            f"Missing 'errored' chip in status column: {statuses!r}"
        )
        glyphs = ("🟢", "🟡", "🔴")
        for s in statuses:
            s_str = str(s)
            if any(token in s_str for token in ("success", "cancelled", "errored")):
                assert any(g in s_str for g in glyphs), (
                    f"Status string {s_str!r} missing color glyph; expected one of {glyphs!r}"
                )


class TestHistoryPageEmptyState:
    def test_empty_state(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FA_DATA_DIR", str(tmp_path))
        at = AppTest.from_file(str(HISTORY_PAGE_PATH)).run(timeout=10)
        assert not at.exception, f"AppTest raised: {[str(e.value) for e in at.exception]}"

        markdown_blobs = [m.value for m in at.markdown]
        text_blobs = [t.value for t in at.text]
        haystack = " ".join(markdown_blobs + text_blobs)
        assert "No signals yet" in haystack, (
            f"Empty-state heading missing; markdown: {markdown_blobs!r}; text: {text_blobs!r}"
        )
        assert (
            "Run your first pipeline from the Run page. Signals will appear here, newest first."
        ) in haystack, f"Empty-state body missing; got: {haystack!r}"
