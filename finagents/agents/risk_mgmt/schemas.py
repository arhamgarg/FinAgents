from __future__ import annotations

from pydantic import BaseModel, Field


class PersonaAssessment(BaseModel):

    main_argument: str
    counter_to_prior_persona: str = ""
    recommended_position_size_pct: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class RiskConsolidation(BaseModel):

    synthesis: str
