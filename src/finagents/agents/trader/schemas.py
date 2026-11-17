from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TradingSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal["BUY", "HOLD", "SELL"]
    conviction: float = Field(ge=0.0, le=1.0)

    entry_price: float | None = None
    stop_loss: float | None = None
    target_1: float | None = None
    target_2: float | None = None

    position_size_pct: float | None = None
    position_size_shares: int | None = None
    risk_reward_ratio: float | None = None

    rationale: str = ""
    analyst_weight_applied: dict[str, float] = Field(default_factory=dict)

    circuit_band_note: str | None = None
    t2t_flag: bool = False

    sizing_method: Literal["atr_based", "kelly", "fixed_fractional"] = "atr_based"

    time_horizon: Literal["positional"] = "positional"

    @field_validator("time_horizon", mode="before")
    @classmethod
    def coerce_time_horizon(cls, v: object) -> str:
        return "positional"
