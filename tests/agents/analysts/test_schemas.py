from __future__ import annotations

import pytest

from financialagents.agents.analysts.schemas import TechnicalReport


class TestTechnicalReportPhase41Fields:
    def test_defaults_atr_14_is_none(self):
        r = TechnicalReport()
        assert r.atr_14 is None

    def test_defaults_circuit_band_pct_is_none(self):
        r = TechnicalReport()
        assert r.circuit_band_pct is None

    def test_defaults_t2t_flag_is_false(self):
        r = TechnicalReport()
        assert r.t2t_flag is False

    def test_explicit_atr_14_stored(self):
        r = TechnicalReport(atr_14=12.5)
        assert r.atr_14 == pytest.approx(12.5)

    def test_explicit_circuit_band_pct_stored(self):
        r = TechnicalReport(circuit_band_pct=0.20)
        assert r.circuit_band_pct == pytest.approx(0.20)

    def test_explicit_t2t_flag_stored(self):
        r = TechnicalReport(t2t_flag=True)
        assert r.t2t_flag is True

    def test_existing_fields_unaffected(self):
        r = TechnicalReport(rsi=55.0, current_price=2450.75)
        assert r.rsi == pytest.approx(55.0)
        assert r.current_price == pytest.approx(2450.75)
        assert r.atr_14 is None
        assert r.circuit_band_pct is None
        assert r.t2t_flag is False

    def test_all_four_pipe12_fields_settable_together(self):
        r = TechnicalReport(
            current_price=2450.75,
            atr_14=12.5,
            circuit_band_pct=0.20,
            t2t_flag=False,
        )
        assert r.current_price == pytest.approx(2450.75)
        assert r.atr_14 == pytest.approx(12.5)
        assert r.circuit_band_pct == pytest.approx(0.20)
        assert r.t2t_flag is False
