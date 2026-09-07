from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Literal

import pydantic
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


from finagents.agents.trader.schemas import TradingSignal  # noqa: E402


class PortfolioDecision(BaseModel):

    signal: TradingSignal
    approved: bool
    decision: Literal["approve", "modify", "veto"]
    pm_notes: str = ""


class PortfolioContext(BaseModel):

    holdings: dict[str, int] = Field(default_factory=dict)
    sector_exposure: dict[str, float] = Field(default_factory=dict)
    total_value_inr: float = 0.0

    @classmethod
    def from_seed_file(cls, path: Path) -> "PortfolioContext":
        try:
            data = json.loads(Path(path).read_text())
            return cls.model_validate(data)
        except FileNotFoundError:
            return cls()
        except (json.JSONDecodeError, pydantic.ValidationError) as err:
            warnings.warn(
                f"PortfolioContext seed at {path} is malformed "
                f"({type(err).__name__}: {err}); falling back to empty context.",
                UserWarning,
                stacklevel=2,
            )
            return cls()
