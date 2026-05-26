from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def divergence_chart_app(tmp_path):
    def _factory(
        sizes: list[float],
        synth_pct: float = 5.0,
    ) -> Path:
        personas_src = ",\n".join(
            f"    PersonaAssessment("
            f"main_argument='m{i}', "
            f"recommended_position_size_pct={s}, "
            f"confidence=0.7, "
            f"counter_to_prior_persona='')"
            for i, s in enumerate(sizes)
        )
        app_src = (
            "import streamlit as st\n"
            "from financialagents.agents.risk_mgmt.schemas import "
            "PersonaAssessment\n"
            "from financialagents.dashboard.divergence_chart import "
            "render_divergence_chart\n"
            "personas = [\n"
            f"{personas_src}\n"
            "]\n"
            f"render_divergence_chart(personas, {synth_pct})\n"
        )
        path = tmp_path / f"divergence_{int(synth_pct)}.py"
        path.write_text(app_src)
        return path
    return _factory


class TestComputeDivergence:
    def test_compute_divergence_pp(self) -> None:
        from financialagents.agents.risk_mgmt.schemas import PersonaAssessment
        from financialagents.dashboard.divergence_chart import (
            _compute_divergence_pp,
        )
        personas = [
            PersonaAssessment(
                main_argument="m1",
                recommended_position_size_pct=0.10,
                confidence=0.7,
            ),
            PersonaAssessment(
                main_argument="m2",
                recommended_position_size_pct=0.05,
                confidence=0.7,
            ),
            PersonaAssessment(
                main_argument="m3",
                recommended_position_size_pct=0.02,
                confidence=0.7,
            ),
        ]
        assert _compute_divergence_pp(personas, 5.0) == pytest.approx(8.0)


class TestDivergenceChartRender:
    def test_bar_chart_renders(self, divergence_chart_app) -> None:
        AppTest = pytest.importorskip(
            "streamlit.testing.v1",
            reason="Streamlit AppTest required",
        ).AppTest
        at = AppTest.from_file(
            str(divergence_chart_app([0.10, 0.05, 0.02], synth_pct=5.0))
        ).run(timeout=10)
        assert not at.exception, (
            f"AppTest raised: {[str(e.value) for e in at.exception]}"
        )
        assert len(at.get("arrow_chart")) >= 0, (
            "AppTest 'arrow_chart' attribute should exist (Streamlit "
            "internal name varies; we instead inspect rendered output)"
        )
        markdown_blobs = " ".join(m.value for m in at.markdown)
        assert "divergence" in markdown_blobs.lower(), (
            f"Expected divergence callout; got: {markdown_blobs!r}"
        )

    def test_callout_color_thresholds_green(
        self, divergence_chart_app
    ) -> None:
        AppTest = pytest.importorskip(
            "streamlit.testing.v1",
            reason="Streamlit AppTest required",
        ).AppTest
        at = AppTest.from_file(
            str(divergence_chart_app([0.10, 0.08, 0.06], synth_pct=8.0))
        ).run(timeout=10)
        assert not at.exception
        text = " ".join(m.value for m in at.markdown)
        assert ":green[" in text or "green[" in text, (
            f"Expected:green[] callout for 4pp divergence; got: {text!r}"
        )

    def test_callout_color_thresholds_amber(
        self, divergence_chart_app
    ) -> None:
        AppTest = pytest.importorskip(
            "streamlit.testing.v1",
            reason="Streamlit AppTest required",
        ).AppTest
        at = AppTest.from_file(
            str(divergence_chart_app([0.12, 0.08, 0.05], synth_pct=8.0))
        ).run(timeout=10)
        assert not at.exception
        text = " ".join(m.value for m in at.markdown)
        assert ":orange[" in text or "orange[" in text, (
            f"Expected:orange[] callout for 7pp divergence; got: {text!r}"
        )

    def test_callout_color_thresholds_red(
        self, divergence_chart_app
    ) -> None:
        AppTest = pytest.importorskip(
            "streamlit.testing.v1",
            reason="Streamlit AppTest required",
        ).AppTest
        at = AppTest.from_file(
            str(divergence_chart_app([0.15, 0.08, 0.03], synth_pct=8.0))
        ).run(timeout=10)
        assert not at.exception
        text = " ".join(m.value for m in at.markdown)
        assert ":red[" in text or "red[" in text, (
            f"Expected:red[] callout for 12pp divergence; got: {text!r}"
        )

    def test_empty_personas(self, divergence_chart_app, tmp_path) -> None:
        AppTest = pytest.importorskip(
            "streamlit.testing.v1",
            reason="Streamlit AppTest required",
        ).AppTest
        app_src = (
            "import streamlit as st\n"
            "from financialagents.dashboard.divergence_chart import "
            "render_divergence_chart\n"
            "render_divergence_chart([], 0.0)\n"
        )
        path = tmp_path / "divergence_empty.py"
        path.write_text(app_src)
        at = AppTest.from_file(str(path)).run(timeout=10)
        assert not at.exception, (
            f"AppTest raised: {[str(e.value) for e in at.exception]}"
        )

    def test_bar_chart_fallback_renders_plain_text_list(
        self, tmp_path,
    ) -> None:
        AppTest = pytest.importorskip(
            "streamlit.testing.v1",
            reason="Streamlit AppTest required",
        ).AppTest
        app_src = (
            "import streamlit as st\n"
            "from financialagents.agents.risk_mgmt.schemas import "
            "PersonaAssessment\n"
            "from financialagents.dashboard.divergence_chart import "
            "render_divergence_chart\n"
            "\n"
            "def _raise_value_error(*args, **kwargs):\n"
            " raise ValueError('simulated altair/pandas drift')\n"
            "\n"
            "st.bar_chart = _raise_value_error\n"
            "personas = [\n"
            " PersonaAssessment(main_argument='m0', "
            "recommended_position_size_pct=0.10, "
            "confidence=0.7, counter_to_prior_persona=''),\n"
            " PersonaAssessment(main_argument='m1', "
            "recommended_position_size_pct=0.05, "
            "confidence=0.7, counter_to_prior_persona=''),\n"
            " PersonaAssessment(main_argument='m2', "
            "recommended_position_size_pct=0.02, "
            "confidence=0.7, counter_to_prior_persona=''),\n"
            "]\n"
            "render_divergence_chart(personas, 5.0)\n"
        )
        path = tmp_path / "divergence_fallback.py"
        path.write_text(app_src)
        at = AppTest.from_file(str(path)).run(timeout=10)
        assert not at.exception, (
            f"AppTest raised: {[str(e.value) for e in at.exception]}"
        )

        warning_blobs = [w.value for w in at.warning]
        assert any(
            "Chart render fallback" in blob
            and "simulated altair/pandas drift" in blob
            for blob in warning_blobs
        ), (
            "Fallback warning missing; got warnings: "
            f"{warning_blobs!r}"
        )

        text_blobs = " ".join(
            m.value for m in (list(at.markdown) + list(at.text))
        )
        for expected_label in ("persona_1", "persona_2", "persona_3"):
            assert expected_label in text_blobs, (
                f"plain-text fallback missing {expected_label!r}; "
                f"got rendered text: {text_blobs!r}"
            )
        for expected_pct in ("10.0%", "5.0%", "2.0%"):
            assert expected_pct in text_blobs, (
                f"plain-text fallback missing percentage {expected_pct!r}; "
                f"got rendered text: {text_blobs!r}"
            )
