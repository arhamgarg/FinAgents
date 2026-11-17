from __future__ import annotations

import streamlit as st

run_page = st.Page(
    "pages/run.py",
    title="Run",
    icon=":material/play_arrow:",
)
history_page = st.Page(
    "pages/history.py",
    title="History",
    icon=":material/history:",
)
signal_page = st.Page(
    "pages/signal.py",
    title="Signal",
    icon=":material/insights:",
)

pg = st.navigation([run_page, history_page, signal_page])
pg.run()
