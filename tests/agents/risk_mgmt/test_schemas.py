from __future__ import annotations

import pydantic
import pytest


class TestPersonaAssessment:
    def test_minimal_valid_construction(self) -> None:
        from finagents.agents.risk_mgmt.schemas import PersonaAssessment

        pa = PersonaAssessment(
            main_argument="thesis",
            recommended_position_size_pct=0.03,
            confidence=0.7,
        )
        assert pa.counter_to_prior_persona == ""

    @pytest.mark.parametrize("bad_pct", [-0.1, 1.1, 2.0])
    def test_rejects_out_of_range_size(self, bad_pct: float) -> None:
        from finagents.agents.risk_mgmt.schemas import PersonaAssessment

        with pytest.raises(pydantic.ValidationError):
            PersonaAssessment(
                main_argument="t",
                recommended_position_size_pct=bad_pct,
                confidence=0.5,
            )

    @pytest.mark.parametrize("bad_conf", [-0.01, 1.01, -1.0])
    def test_rejects_out_of_range_confidence(self, bad_conf: float) -> None:
        from finagents.agents.risk_mgmt.schemas import PersonaAssessment

        with pytest.raises(pydantic.ValidationError):
            PersonaAssessment(
                main_argument="t",
                recommended_position_size_pct=0.03,
                confidence=bad_conf,
            )

    def test_empty_counter_accepted_for_round_1_aggressive(self) -> None:
        from finagents.agents.risk_mgmt.schemas import PersonaAssessment

        pa = PersonaAssessment(
            main_argument="t",
            counter_to_prior_persona="",
            recommended_position_size_pct=0.03,
            confidence=0.7,
        )
        assert pa.counter_to_prior_persona == ""

    def test_serialization_roundtrip(self) -> None:
        from finagents.agents.risk_mgmt.schemas import PersonaAssessment

        pa = PersonaAssessment(
            main_argument="thesis",
            recommended_position_size_pct=0.05,
            confidence=0.8,
        )
        payload = pa.model_dump_json()
        assert PersonaAssessment.model_validate_json(payload) == pa


class TestRiskConsolidation:
    def test_requires_synthesis(self) -> None:
        from finagents.agents.risk_mgmt.schemas import RiskConsolidation

        with pytest.raises(pydantic.ValidationError):
            RiskConsolidation()  # ty: ignore[missing-argument]

    def test_valid_construction(self) -> None:
        from finagents.agents.risk_mgmt.schemas import RiskConsolidation

        rc = RiskConsolidation(synthesis="weighted view: 3%")
        assert rc.synthesis == "weighted view: 3%"

    def test_has_no_numeric_field(self) -> None:
        from finagents.agents.risk_mgmt.schemas import RiskConsolidation

        assert "synthesized_position_size_pct" not in RiskConsolidation.model_fields
