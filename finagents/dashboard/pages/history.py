from __future__ import annotations

import asyncio

import pandas as pd
import streamlit as st

from finagents.config import Settings
from finagents.storage.paths import signals_db_path
from finagents.storage.signals import load_history


EMPTY_HEADING: str = "No signals yet"
EMPTY_BODY: str = (
    "Run your first pipeline from the Run page. Signals will appear here, newest first."
)


_STATUS_CHIP: dict[str, str] = {
    "success": "🟢 success",
    "cancelled": "🟡 cancelled",
    "errored": "🔴 errored",
    "running": "⏳ running",
}


def _status_chip(s: str | None) -> str:
    if not s:
        return "—"
    return _STATUS_CHIP.get(s, s)


@st.cache_data(ttl=60)
def _cached_history() -> pd.DataFrame:
    rows = asyncio.run(load_history(signals_db_path(Settings()), limit=1000, include_payload=False))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "status" in df.columns:
        df["status"] = df["status"].map(_status_chip)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df



st.title("History")

top_cols = st.columns([1, 6])
with top_cols[0]:
    if st.button("Refresh history"):
        st.cache_data.clear()
        st.rerun()
with top_cols[1]:
    st.caption("Click a row to open the signal.")

df = _cached_history()

if df.empty:
    st.markdown(f"## {EMPTY_HEADING}")
    st.write(EMPTY_BODY)
else:
    event = st.dataframe(
        df,
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        width="stretch",
        column_config={
            "run_id": None,
            "payload_json": None,
            "provider": None,
            "model": None,
            "error_msg": None,
            "created_at": st.column_config.DatetimeColumn(
                "Created",
                format="YYYY-MM-DD HH:mm",
                width="medium",
            ),
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "date": st.column_config.TextColumn("Date", width="small"),
            "action": st.column_config.TextColumn("Action", width="small"),
            "confidence": st.column_config.NumberColumn(
                "Confidence",
                format="%.2f",
                width="small",
            ),
            "position_size_pct": st.column_config.NumberColumn(
                "Size %",
                format="%.1f%%",
                width="small",
            ),
            "approved": st.column_config.CheckboxColumn(
                "Approved",
                width="small",
            ),
            "pm_decision": st.column_config.TextColumn("PM", width="small"),
            "status": st.column_config.TextColumn("Status", width="small"),
        },
    )
    selection = getattr(event, "selection", None)
    rows = getattr(selection, "rows", None) if selection is not None else None
    if rows:
        idx = rows[0]
        st.query_params["id"] = str(df.iloc[idx]["run_id"])
        st.switch_page("pages/signal.py")
