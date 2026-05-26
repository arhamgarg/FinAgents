from __future__ import annotations

import pytest

from financialagents.agents.trader.schemas import TradingSignal
from financialagents.config import Settings


@pytest.fixture
def default_settings() -> Settings:
    return Settings(_env_file=None, openrouter_api_key="fake")


def _make_signal(
    *,
    action: str = "BUY",
    conviction: float = 0.8,
    entry_price: float | None = 1500.0,
    stop_loss: float | None = 1450.0,
    target_1: float | None = 1600.0,
    rationale: str = "",
    position_size_pct: float | None = None,
) -> TradingSignal:
    return TradingSignal(
        action=action,  # type: ignore[arg-type]
        conviction=conviction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_1=target_1,
        rationale=rationale,
        position_size_pct=position_size_pct,
    )


class TestPriceGrounding:
    def test_price_grounding_overwrites_entry_price(
        self, default_settings: Settings
    ) -> None:
        from financialagents.agents.trader.validators import (
            ValidatorContext,
            price_grounding_validator,
        )

        signal = _make_signal(entry_price=1500.0)
        ctx = ValidatorContext(
            current_price=2450.75,
            circuit_band_pct=0.20,
            settings=default_settings,
        )
        result = price_grounding_validator(signal, ctx)
        assert result.signal.entry_price == pytest.approx(2450.75)
        assert result.halted is False
        assert result.signal.action == "BUY"

    def test_price_grounding_none_forces_hold(
        self, default_settings: Settings
    ) -> None:
        from financialagents.agents.trader.validators import (
            ValidatorContext,
            price_grounding_validator,
        )

        signal = _make_signal(entry_price=1500.0)
        ctx = ValidatorContext(
            current_price=None,
            circuit_band_pct=0.20,
            settings=default_settings,
        )
        result = price_grounding_validator(signal, ctx)
        assert result.signal.action == "HOLD"
        assert result.halted is True
        assert "current_price unavailable" in (result.signal.rationale or "")
        assert result.signal.position_size_pct is None
        assert result.signal.position_size_shares is None

    def test_price_grounding_preserves_conviction(
        self, default_settings: Settings
    ) -> None:
        from financialagents.agents.trader.validators import (
            ValidatorContext,
            price_grounding_validator,
        )

        signal = _make_signal(conviction=0.8, entry_price=1500.0)
        ctx = ValidatorContext(
            current_price=None,
            circuit_band_pct=0.20,
            settings=default_settings,
        )
        result = price_grounding_validator(signal, ctx)
        assert result.signal.conviction == pytest.approx(0.8)


class TestCircuitBand:
    def test_circuit_band_80pct_breach(self, default_settings: Settings) -> None:
        from financialagents.agents.trader.validators import (
            ValidatorContext,
            circuit_band_validator,
        )

        signal = _make_signal(entry_price=100.0, stop_loss=91.0)
        ctx = ValidatorContext(
            current_price=100.0,
            circuit_band_pct=0.10,
            settings=default_settings,
        )
        result = circuit_band_validator(signal, ctx)
        assert result.signal.action == "HOLD"
        assert result.halted is True
        assert result.signal.circuit_band_note is not None
        assert "circuit band" in result.signal.circuit_band_note.lower()

    def test_circuit_band_unavailable_annotates_but_allows_signal(
        self, default_settings: Settings
    ) -> None:
        from financialagents.agents.trader.validators import (
            CIRCUIT_BAND_UNAVAILABLE_NOTE,
            ValidatorContext,
            circuit_band_validator,
        )

        signal = _make_signal(entry_price=100.0, stop_loss=95.0)
        ctx = ValidatorContext(
            current_price=100.0,
            circuit_band_pct=None,
            settings=default_settings,
        )
        result = circuit_band_validator(signal, ctx)
        assert result.signal.action == "BUY"
        assert result.halted is False
        assert result.signal.circuit_band_note == CIRCUIT_BAND_UNAVAILABLE_NOTE

    def test_circuit_band_unavailable_still_blocks_extreme_stop(
        self, default_settings: Settings
    ) -> None:
        from financialagents.agents.trader.validators import (
            ValidatorContext,
            circuit_band_validator,
        )

        signal = _make_signal(entry_price=100.0, stop_loss=83.0)
        ctx = ValidatorContext(
            current_price=100.0,
            circuit_band_pct=None,
            settings=default_settings,
        )
        result = circuit_band_validator(signal, ctx)
        assert result.signal.action == "HOLD"
        assert result.halted is True
        assert result.signal.circuit_band_note is not None
        assert "circuit band" in result.signal.circuit_band_note.lower()

    def test_circuit_band_within_threshold_allows_signal(
        self, default_settings: Settings
    ) -> None:
        from financialagents.agents.trader.validators import (
            ValidatorContext,
            circuit_band_validator,
        )

        signal = _make_signal(entry_price=100.0, stop_loss=95.0)
        ctx = ValidatorContext(
            current_price=100.0,
            circuit_band_pct=0.20,
            settings=default_settings,
        )
        result = circuit_band_validator(signal, ctx)
        assert result.signal.action == "BUY"
        assert result.halted is False


class TestSttFloor:
    def test_stt_floor(self, default_settings: Settings) -> None:
        from financialagents.agents.trader.validators import (
            ValidatorContext,
            stt_cost_floor_validator,
        )

        signal = _make_signal(
            entry_price=100.0,
            target_1=100.2,
            rationale="bullish breakout",
        )
        ctx = ValidatorContext(
            current_price=100.0,
            circuit_band_pct=0.20,
            settings=default_settings,
        )
        result = stt_cost_floor_validator(signal, ctx)
        assert result.signal.action == "HOLD"
        assert result.halted is True
        assert "0.3%" in (result.signal.rationale or "")

    def test_stt_floor_above_threshold_allows_signal(
        self, default_settings: Settings
    ) -> None:
        from financialagents.agents.trader.validators import (
            ValidatorContext,
            stt_cost_floor_validator,
        )

        signal = _make_signal(entry_price=100.0, target_1=100.5)
        ctx = ValidatorContext(
            current_price=100.0,
            circuit_band_pct=0.20,
            settings=default_settings,
        )
        result = stt_cost_floor_validator(signal, ctx)
        assert result.signal.action == "BUY"
        assert result.halted is False

    def test_stt_floor_skips_when_already_hold(
        self, default_settings: Settings
    ) -> None:
        from financialagents.agents.trader.validators import (
            ValidatorContext,
            stt_cost_floor_validator,
        )

        signal = _make_signal(action="HOLD", entry_price=100.0, target_1=100.1)
        ctx = ValidatorContext(
            current_price=100.0,
            circuit_band_pct=0.20,
            settings=default_settings,
        )
        result = stt_cost_floor_validator(signal, ctx)
        assert result.signal.action == "HOLD"
        assert result.halted is False


class TestKellyCap:
    def test_kelly_cap_clips_oversized_position(
        self, default_settings: Settings
    ) -> None:
        from financialagents.agents.trader.validators import (
            ValidatorContext,
            kelly_cap_validator,
        )

        signal = _make_signal(position_size_pct=0.15)
        ctx = ValidatorContext(
            current_price=100.0,
            circuit_band_pct=0.20,
            settings=default_settings,
        )
        result = kelly_cap_validator(signal, ctx)
        assert result.signal.position_size_pct == pytest.approx(0.05)
        assert result.signal.action == "BUY"

    def test_kelly_cap_leaves_small_position_alone(
        self, default_settings: Settings
    ) -> None:
        from financialagents.agents.trader.validators import (
            ValidatorContext,
            kelly_cap_validator,
        )

        signal = _make_signal(position_size_pct=0.04)
        ctx = ValidatorContext(
            current_price=100.0,
            circuit_band_pct=0.20,
            settings=default_settings,
        )
        result = kelly_cap_validator(signal, ctx)
        assert result.signal.position_size_pct == pytest.approx(0.04)

    def test_kelly_cap_none_is_noop(self, default_settings: Settings) -> None:
        from financialagents.agents.trader.validators import (
            ValidatorContext,
            kelly_cap_validator,
        )

        signal = _make_signal(position_size_pct=None)
        ctx = ValidatorContext(
            current_price=100.0,
            circuit_band_pct=0.20,
            settings=default_settings,
        )
        result = kelly_cap_validator(signal, ctx)
        assert result.signal.position_size_pct is None


class TestRunValidatorsChain:
    def test_validators_collect_all_notes_after_hold(
        self, default_settings: Settings
    ) -> None:
        from financialagents.agents.trader.validators import (
            CIRCUIT_BAND_UNAVAILABLE_NOTE,
            ValidatorContext,
            run_validators,
        )

        signal = _make_signal(
            entry_price=1500.0,
            stop_loss=1450.0,
            target_1=1600.0,
            position_size_pct=0.20,
        )
        ctx = ValidatorContext(
            current_price=1500.0,
            circuit_band_pct=None,
            settings=default_settings,
        )
        result = run_validators(signal, ctx)
        assert result.action == "BUY"
        assert result.circuit_band_note == CIRCUIT_BAND_UNAVAILABLE_NOTE
        assert result.position_size_pct == pytest.approx(0.05)

    def test_chain_runs_all_validators_in_order(
        self, default_settings: Settings
    ) -> None:
        from financialagents.agents.trader.validators import (
            ValidatorContext,
            run_validators,
        )

        current_price = 2450.75
        signal = _make_signal(
            entry_price=1500.0,
            stop_loss=2400.0,
            target_1=2500.0,
            position_size_pct=0.03,
        )
        ctx = ValidatorContext(
            current_price=current_price,
            circuit_band_pct=0.20,
            settings=default_settings,
        )
        result = run_validators(signal, ctx)
        assert result.entry_price == pytest.approx(current_price)
        assert result.action == "BUY"
        assert result.position_size_pct == pytest.approx(0.03)
