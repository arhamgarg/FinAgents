from __future__ import annotations

from typing import Any, Sequence

import streamlit as st


def _color_for_divergence_pp(pp: float) -> str:
    if pp <= 5.0:
        return "green"
    if pp <= 10.0:
        return "orange"
    return "red"


def _normalize_size_pct(value: Any) -> float:
    if value is None:
        return 0.0
    pct = float(value)
    return pct * 100 if pct <= 1 else pct


def _compute_divergence_pp(
    personas: Sequence[Any],
    _synth_pct: float,
) -> float:
    if not personas:
        return 0.0
    sizes = [
        _normalize_size_pct(getattr(p, "recommended_position_size_pct", 0.0)) for p in personas
    ]
    return max(sizes) - min(sizes)


def render_divergence_chart(
    personas: Sequence[Any],
    synth_pct: float,
) -> None:
    if not personas:
        st.write("No persona assessments available.")
        return

    pp = _compute_divergence_pp(personas, synth_pct)
    color_alias = _color_for_divergence_pp(pp)
    st.markdown(f":{color_alias}[divergence: {pp:.1f}pp]")

    persona_labels = []
    sizes = []
    for i, p in enumerate(personas):
        label = getattr(p, "persona_name", None) or f"persona_{i + 1}"
        persona_labels.append(label)
        sizes.append(_normalize_size_pct(getattr(p, "recommended_position_size_pct", 0.0)))

    try:
        import pandas as pd

        df = pd.DataFrame({"persona": persona_labels, "size_pct": sizes})
        st.bar_chart(df, x="size_pct", y="persona", horizontal=True)
    except (ValueError, TypeError, ImportError) as exc:
        st.warning(f"Chart render fallback: {exc}")
        for label, size in zip(persona_labels, sizes):
            st.write(f"- {label}: {size:.1f}%")

    if synth_pct:
        st.caption(f"Synthesised size: {synth_pct:.1f}%")


__all__ = [
    "render_divergence_chart",
]
