from __future__ import annotations

from pydantic import BaseModel, Field


class Finding(BaseModel):

    analyst: str
    key_point: str
    supporting_data: str = ""
    weight: float = 1.0


class ResearchBrief(BaseModel):

    key_findings: list[Finding] = Field(default_factory=list)
    consensus_view: str = ""
    conflicts: list[str] = Field(default_factory=list)
    data_quality_summary: str = ""
    analyst_confidences: dict[str, float] = Field(default_factory=dict)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str = ""
