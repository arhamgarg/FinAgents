from __future__ import annotations

from typing import Any

import streamlit as st


_REPORTS: tuple[tuple[str, str], ...] = (
    ("Technical", "technical_report"),
    ("Fundamentals", "fundamentals_report"),
    ("News", "news_report"),
)


def _render_technical(report: Any) -> None:
    rsi = getattr(report, "rsi", None)
    price = getattr(report, "current_price", None)
    confidence = getattr(report, "confidence", 0.0)
    if rsi is not None:
        st.metric(
            "RSI",
            f"{rsi:.1f}",
            delta=f"conf {confidence:.2f}",
            delta_color="off",
        )
    elif price is not None:
        st.metric(
            "Price",
            f"₹{price:,.0f}",
            delta=f"conf {confidence:.2f}",
            delta_color="off",
        )
    else:
        st.metric("Confidence", f"{confidence:.2f}")
    notes = getattr(report, "notes", "") or ""
    if notes:
        st.caption(notes)


def _render_fundamentals(report: Any) -> None:
    pe = getattr(report, "pe_ratio", None)
    confidence = getattr(report, "confidence", 0.0)
    if pe is not None:
        st.metric(
            "P/E",
            f"{pe:.1f}",
            delta=f"conf {confidence:.2f}",
            delta_color="off",
        )
    else:
        st.metric("Confidence", f"{confidence:.2f}")
    assessment = getattr(report, "fundamental_assessment", "") or ""
    if assessment:
        st.caption(assessment)


def _render_news(report: Any) -> None:
    direction = getattr(report, "sentiment_direction", "") or "—"
    article_count = getattr(report, "article_count", 0)
    confidence = getattr(report, "confidence", 0.0)
    st.metric(
        "Sentiment",
        direction,
        delta=f"{article_count} articles · conf {confidence:.2f}",
        delta_color="off",
    )
    assessment = getattr(report, "news_assessment", "") or ""
    if assessment:
        st.caption(assessment)


_RENDERERS: dict[str, Any] = {
    "technical_report": _render_technical,
    "fundamentals_report": _render_fundamentals,
    "news_report": _render_news,
}


def render_agent_reports(state: Any) -> None:
    cols = st.columns(3)
    for i, (label, attr) in enumerate(_REPORTS):
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                report = getattr(state, attr, None)
                if report is None:
                    st.write("—")
                    continue
                renderer = _RENDERERS.get(attr)
                if renderer is not None:
                    renderer(report)
                with st.expander("Full report", expanded=False):
                    text = (
                        getattr(report, "report_markdown", None)
                        or _format_report_text(report)
                    )
                    st.markdown(text)


def _format_report_text(report: Any) -> str:
    if hasattr(report, "model_dump"):
        data = report.model_dump()
    else:
        data = {
            k: getattr(report, k)
            for k in dir(report)
            if not k.startswith("_") and not callable(getattr(report, k))
        }
    lines = []
    for key, value in data.items():
        if value is None or value == "" or value == [] or value == {}:
            continue
        lines.append(f"- **{key}:** {value}")
    return "\n".join(lines) if lines else "—"


__all__ = ["render_agent_reports"]
