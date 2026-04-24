from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class TradingSignal(BaseModel):

    action: Literal["BUY", "HOLD", "SELL"]
    conviction: float = Field(ge=0.0, le=1.0)

    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None

    position_size_pct: Optional[float] = None
    position_size_shares: Optional[int] = None
    risk_reward_ratio: Optional[float] = None

    rationale: str = ""
    analyst_weight_applied: dict[str, float] = Field(default_factory=dict)

    circuit_band_note: Optional[str] = None
    t2t_flag: bool = False

    sizing_method: Literal["atr_based", "kelly", "fixed_fractional"] = "atr_based"
    time_horizon: Literal["positional"] = "positional"
