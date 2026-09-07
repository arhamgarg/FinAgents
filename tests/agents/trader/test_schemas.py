from __future__ import annotations

import pydantic
import pytest


class TestTradingSignalRequiredFields:
    def test_minimal_construction_action_and_conviction_only(self) -> None:
        from finagents.agents.trader.schemas import TradingSignal

        sig = TradingSignal(action="BUY", conviction=0.7)
        assert sig.action == "BUY"
        assert sig.conviction == pytest.approx(0.7)
        assert sig.entry_price is None
        assert sig.stop_loss is None
        assert sig.rationale == ""
        assert sig.analyst_weight_applied == {}
        assert sig.t2t_flag is False
        assert sig.sizing_method == "atr_based"
        assert sig.time_horizon == "positional"

    def test_missing_action_raises(self) -> None:
        from finagents.agents.trader.schemas import TradingSignal

        with pytest.raises(pydantic.ValidationError):
            TradingSignal(conviction=0.5)  # ty: ignore[missing-argument]  # Verify required-field validation.

    def test_missing_conviction_raises(self) -> None:
        from finagents.agents.trader.schemas import TradingSignal

        with pytest.raises(pydantic.ValidationError):
            TradingSignal(action="BUY")  # ty: ignore[missing-argument]  # Verify required-field validation.

    @pytest.mark.parametrize("bad_conviction", [1.1, 99.9, -0.01, -1.0, 2.0])
    def test_rejects_out_of_range_conviction(self, bad_conviction: float) -> None:
        from finagents.agents.trader.schemas import TradingSignal

        with pytest.raises(pydantic.ValidationError):
            TradingSignal(action="BUY", conviction=bad_conviction)

    @pytest.mark.parametrize("bad_action", ["buy", "SHORT", "", "invalid"])
    def test_rejects_invalid_action_literal(self, bad_action: str) -> None:
        from finagents.agents.trader.schemas import TradingSignal

        with pytest.raises(pydantic.ValidationError):
            TradingSignal(action=bad_action, conviction=0.5)  # ty: ignore[invalid-argument-type]  # Exercise runtime validation/coercion.

    def test_hold_and_sell_are_valid_actions(self) -> None:
        from finagents.agents.trader.schemas import TradingSignal

        assert TradingSignal(action="HOLD", conviction=0.0).action == "HOLD"
        assert TradingSignal(action="SELL", conviction=1.0).action == "SELL"


class TestTradingSignalOptionalFields:
    def test_partial_output_degrades_to_defaults(self) -> None:
        from finagents.agents.trader.schemas import TradingSignal

        sig = TradingSignal(action="HOLD", conviction=0.1)
        assert sig.action == "HOLD"
        assert sig.position_size_pct is None
        assert sig.position_size_shares is None
        assert sig.risk_reward_ratio is None

    def test_sizing_method_literal_accepted(self) -> None:
        from finagents.agents.trader.schemas import TradingSignal

        for method in ("atr_based", "kelly", "fixed_fractional"):
            sig = TradingSignal(action="BUY", conviction=0.5, sizing_method=method)
            assert sig.sizing_method == method

    def test_time_horizon_positional_only(self) -> None:
        from finagents.agents.trader.schemas import TradingSignal

        sig = TradingSignal(action="BUY", conviction=0.5, time_horizon="positional")
        assert sig.time_horizon == "positional"

    @pytest.mark.parametrize(
        "llm_value",
        ["intraday", "swing", "multi-day-to-multi-week", "long-term", "short-term"],
    )
    def test_time_horizon_coerces_any_value_to_positional(self, llm_value: str) -> None:
        from finagents.agents.trader.schemas import TradingSignal

        sig = TradingSignal(action="BUY", conviction=0.5, time_horizon=llm_value)  # ty: ignore[invalid-argument-type]  # Exercise runtime validation/coercion.
        assert sig.time_horizon == "positional", (
            f"Expected 'positional' after coercion of {llm_value!r}, got {sig.time_horizon!r}"
        )

    def test_circuit_band_note_populated(self) -> None:
        from finagents.agents.trader.schemas import TradingSignal

        sig = TradingSignal(
            action="HOLD",
            conviction=0.8,
            circuit_band_note="Stop-loss exceeds 80% of circuit band — forced HOLD",
        )
        assert "forced HOLD" in (sig.circuit_band_note or "")
