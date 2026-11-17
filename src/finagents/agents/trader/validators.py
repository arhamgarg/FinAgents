from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from finagents.agents.trader.schemas import TradingSignal
from finagents.config import Settings

CIRCUIT_BAND_DEFAULT_NOTE = (
    "circuit band absent from NSE surveillance; using standard 20% daily limit"
)
CIRCUIT_BAND_UNAVAILABLE_NOTE = CIRCUIT_BAND_DEFAULT_NOTE
STT_COST_FLOOR = 0.003
CIRCUIT_BAND_BREACH_RATIO = 0.8
POSITION_SIZE_HARD_CAP = 0.05
NSE_DEFAULT_CIRCUIT_BAND_PCT = 0.20


class ValidatorContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    current_price: float | None = None
    circuit_band_pct: float | None = None
    settings: Settings


def with_suffix(rationale: str | None, suffix: str) -> str:
    text = rationale or ""
    return text if suffix in text else text + suffix


class ValidatorResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    signal: TradingSignal
    note: str | None = None
    halted: bool = False


def price_grounding_validator(signal: TradingSignal, ctx: ValidatorContext) -> ValidatorResult:
    if ctx.current_price is None:
        held = signal.model_copy(
            update={
                "action": "HOLD",
                "entry_price": None,
                "position_size_pct": None,
                "position_size_shares": None,
                "sizing_method": "atr_based",
                "rationale": with_suffix(
                    signal.rationale, " [HOLD: current_price unavailable for price grounding]"
                ),
            }
        )
        return ValidatorResult(
            signal=held,
            note="current_price unavailable",
            halted=True,
        )
    return ValidatorResult(signal=signal.model_copy(update={"entry_price": ctx.current_price}))


def circuit_band_validator(signal: TradingSignal, ctx: ValidatorContext) -> ValidatorResult:
    effective_band = ctx.circuit_band_pct
    if effective_band is None:
        effective_band = NSE_DEFAULT_CIRCUIT_BAND_PCT
        signal = signal.model_copy(update={"circuit_band_note": CIRCUIT_BAND_DEFAULT_NOTE})

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
        held = signal.model_copy(
            update={
                "action": "HOLD",
                "position_size_pct": None,
                "position_size_shares": None,
                "sizing_method": "atr_based",
                "circuit_band_note": note,
            }
        )
        return ValidatorResult(signal=held, note=note, halted=True)
    return ValidatorResult(signal=signal)


def stt_cost_floor_validator(signal: TradingSignal, ctx: ValidatorContext) -> ValidatorResult:
    if signal.action == "HOLD":
        return ValidatorResult(signal=signal)
    if signal.entry_price is None or signal.target_1 is None:
        return ValidatorResult(signal=signal)
    if signal.entry_price <= 0:
        return ValidatorResult(signal=signal)
    target_pct = (signal.target_1 / signal.entry_price) - 1
    if target_pct < STT_COST_FLOOR:
        note = f"target yields {target_pct:.3%}; below 0.3% STT/GST net floor"
        held = signal.model_copy(
            update={
                "action": "HOLD",
                "position_size_pct": None,
                "position_size_shares": None,
                "sizing_method": "atr_based",
                "rationale": with_suffix(signal.rationale, f" [HOLD: {note}]"),
            }
        )
        return ValidatorResult(signal=held, note=note, halted=True)
    return ValidatorResult(signal=signal)


def kelly_cap_validator(signal: TradingSignal, ctx: ValidatorContext) -> ValidatorResult:
    if signal.position_size_pct is not None and signal.position_size_pct > POSITION_SIZE_HARD_CAP:
        original_pct = signal.position_size_pct
        update: dict[str, float | int] = {"position_size_pct": POSITION_SIZE_HARD_CAP}
        if (
            signal.position_size_shares is not None
            and signal.position_size_shares > 0
            and original_pct > 0
        ):
            scale = POSITION_SIZE_HARD_CAP / original_pct
            update["position_size_shares"] = int(signal.position_size_shares * scale)
        signal = signal.model_copy(update=update)
    return ValidatorResult(signal=signal)


def run_validators(signal: TradingSignal, ctx: ValidatorContext) -> TradingSignal:
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
