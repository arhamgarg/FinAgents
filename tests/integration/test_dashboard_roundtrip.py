import asyncio
from pathlib import Path

from streamlit.testing.v1 import AppTest

from finagents.config import Settings
from finagents.storage.paths import signals_db_path
from finagents.storage.signals import load_history
from tests.pipeline import pipeline_scenario


def test_dashboard_consecutive_runs():
    pages = Path(__file__).resolve().parents[2] / "finagents/dashboard/pages"
    at = AppTest.from_file(str(pages / "run.py")).run()
    with pipeline_scenario() as h:
        ids = []
        for ticker in ("RELIANCE", "TCS"):
            next(w for w in at.sidebar.text_input if w.label == "Ticker").set_value(ticker)
            next(w for w in at.sidebar.button if w.label == "Run pipeline").click()
            at.run(timeout=10)
            assert not at.exception
            thread = at.session_state["pipeline_thread"]
            thread.join(timeout=10)
            assert not thread.is_alive()
            at.run(timeout=10)
            assert not at.exception
            assert at.session_state["persisted"]
            ids.append(at.session_state["run_id"])
    assert ids[0] != ids[1]
    assert len(h.calls) == 13
    rows = asyncio.run(load_history(signals_db_path(Settings())))
    assert len(rows) == 2
    assert {row["ticker"] for row in rows} == {"RELIANCE", "TCS"}
    assert all(row["status"] == "success" for row in rows)
    history = AppTest.from_file(str(pages / "history.py")).run()
    assert not history.exception
    assert len(history.dataframe[0].value) == 2
    signal = AppTest.from_file(str(pages / "signal.py"))
    signal.query_params["id"] = ids[-1]
    signal.run()
    assert not signal.exception
