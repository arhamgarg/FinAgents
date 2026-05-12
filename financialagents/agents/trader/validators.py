from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from financialagents.agents.trader.schemas import TradingSignal
from financialagents.config import Settings

CIRCUIT_BAND_UNAVAILABLE_NOTE = (
    "circuit band unavailable; refusing to size stop without it"
)
STT_COST_FLOOR = 0.003
CIRCUIT_BAND_BREACH_RATIO = 0.8
POSITION_SIZE_HARD_CAP = 0.05
NSE_DEFAULT_CIRCUIT_BAND_PCT = 0.20


class ValidatorContext(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_price: Optional[float] = None
    circuit_band_pct: Optional[float] = None
    settings: Settings


class ValidatorResult(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal: TradingSignal
    note: Optional[str] = None
    halted: bool = False


def price_grounding_validator(
    signal: TradingSignal, ctx: ValidatorContext
) -> ValidatorResult:
    if ctx.current_price is None:
        signal.action = "HOLD"
        signal.entry_price = None
        signal.position_size_pct = None
        signal.position_size_shares = None
        suffix = " [HOLD: current_price unavailable for price grounding]"
        if suffix not in (signal.rationale or ""):
            signal.rationale = (signal.rationale or "") + suffix
        return ValidatorResult(
            signal=signal,
            note="current_price unavailable",
            halted=True,
        )
    signal.entry_price = ctx.current_price
    return ValidatorResult(signal=signal)


def circuit_band_validator(
    signal: TradingSignal, ctx: ValidatorContext
) -> ValidatorResult:
    effective_band = ctx.circuit_band_pct
    if effective_band is None:
        effective_band = NSE_DEFAULT_CIRCUIT_BAND_PCT
        signal.circuit_band_note = CIRCUIT_BAND_UNAVAILABLE_NOTE

    if signal.action == "HOLD":
        return ValidatorResult(signal=signal)
    if signal.entry_price is None or signal.stop_loss is None:
        return ValidatorResult(signal=signal)
    if signal.entry_price <= 0:
        return ValidatorResult(signal=signal)
    band_used = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
    if band_used > effective_band * CIRCUIT_BAND_BREACH_RATIO:
        note = (
            f"stop-loss uses {band_used:.1%} of entry; exceeds "
            f"{int(CIRCUIT_BAND_BREACH_RATIO * 100)}% of "
            f"{effective_band:.0%} circuit band"
        )
        signal.action = "HOLD"
        signal.position_size_pct = None
        signal.position_size_shares = None
        signal.circuit_band_note = note
        return ValidatorResult(signal=signal, note=note, halted=True)
    return ValidatorResult(signal=signal)


def stt_cost_floor_validator(
    signal: TradingSignal, ctx: ValidatorContext
) -> ValidatorResult:
    if signal.action == "HOLD":
        return ValidatorResult(signal=signal)
    if signal.entry_price is None or signal.target_1 is None:
        return ValidatorResult(signal=signal)
    if signal.entry_price <= 0:
        return ValidatorResult(signal=signal)
    target_pct = (signal.target_1 / signal.entry_price) - 1
    if target_pct < STT_COST_FLOOR:
        note = f"target yields {target_pct:.3%}; below 0.3% STT/GST net floor"
        signal.action = "HOLD"
        signal.position_size_pct = None
        signal.position_size_shares = None
        suffix = f" [HOLD: {note}]"
        if suffix not in (signal.rationale or ""):
            signal.rationale = (signal.rationale or "") + suffix
        return ValidatorResult(signal=signal, note=note, halted=True)
    return ValidatorResult(signal=signal)


def kelly_cap_validator(
    signal: TradingSignal, ctx: ValidatorContext
) -> ValidatorResult:
    if (
        signal.position_size_pct is not None
        and signal.position_size_pct > POSITION_SIZE_HARD_CAP
    ):
        signal.position_size_pct = POSITION_SIZE_HARD_CAP
    return ValidatorResult(signal=signal)


def run_validators(
    signal: TradingSignal, ctx: ValidatorContext
) -> TradingSignal:
    for validator in (
        price_grounding_validator,
        circuit_band_validator,
        stt_cost_floor_validator,
        kelly_cap_validator,
    ):
        result = validator(signal, ctx)
        signal = result.signal
    return signal


ValidatorContext.model_rebuild()
ValidatorResult.model_rebuild()
