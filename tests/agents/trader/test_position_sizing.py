from __future__ import annotations

import math

import pytest


class TestAtrBasedSize:

    def test_normal_case(self) -> None:
        from financialagents.agents.trader.position_sizing import atr_based_size

        pct, shares = atr_based_size(
            capital=100_000,
            risk_per_trade_pct=0.02,
            atr=50.0,
            atr_multiplier=2.0,
            entry_price=2450.0,
        )
        assert shares == 20
        assert pct == pytest.approx(0.49)

    def test_zero_atr_returns_zero_shares(self) -> None:
        from financialagents.agents.trader.position_sizing import atr_based_size

        pct, shares = atr_based_size(
            capital=100_000,
            risk_per_trade_pct=0.02,
            atr=0.0,
            atr_multiplier=2.0,
            entry_price=100.0,
        )
        assert shares == 0
        assert pct == 0.0

    def test_negative_capital_returns_zero(self) -> None:
        from financialagents.agents.trader.position_sizing import atr_based_size

        pct, shares = atr_based_size(
            capital=-1,
            risk_per_trade_pct=0.02,
            atr=50.0,
            atr_multiplier=2.0,
            entry_price=100.0,
        )
        assert pct == 0.0 and shares == 0

    def test_zero_entry_price_returns_zero(self) -> None:
        from financialagents.agents.trader.position_sizing import atr_based_size

        pct, shares = atr_based_size(
            capital=100_000,
            risk_per_trade_pct=0.02,
            atr=50.0,
            atr_multiplier=2.0,
            entry_price=0.0,
        )
        assert pct == 0.0 and shares == 0


class TestKellySize:

    def test_kelly_normal_case(self) -> None:
        from financialagents.agents.trader.position_sizing import kelly_size

        result = kelly_size(
            win_rate=0.6, reward_ratio=2.0, kelly_fraction=0.25
        )
        assert result == pytest.approx(0.05)

    def test_kelly_cap(self) -> None:
        from financialagents.agents.trader.position_sizing import kelly_size

        result = kelly_size(
            win_rate=0.99, reward_ratio=2.0, kelly_fraction=0.25
        )
        assert result <= 0.05 + 1e-9

    def test_kelly_negative_edge_returns_zero(self) -> None:
        from financialagents.agents.trader.position_sizing import kelly_size

        result = kelly_size(
            win_rate=0.3, reward_ratio=2.0, kelly_fraction=0.25
        )
        assert result == 0.0

    def test_kelly_zero_reward_ratio_returns_zero(self) -> None:
        from financialagents.agents.trader.position_sizing import kelly_size

        result = kelly_size(
            win_rate=0.7, reward_ratio=0.0, kelly_fraction=0.25
        )
        assert result == 0.0

    def test_kelly_fraction_scales_below_cap(self) -> None:
        from financialagents.agents.trader.position_sizing import kelly_size

        result = kelly_size(
            win_rate=0.55, reward_ratio=1.0, kelly_fraction=0.1
        )
        assert result == pytest.approx(0.01)


class TestFixedFractionalSize:

    def test_fixed_fractional_normal_case(self) -> None:
        from financialagents.agents.trader.position_sizing import fixed_fractional_size

        pct, shares = fixed_fractional_size(
            capital=100_000,
            risk_per_trade_pct=0.02,
            entry_price=100.0,
            stop_loss=95.0,
        )
        assert shares == 400
        assert pct == pytest.approx(0.40)

    def test_fixed_fractional_division_by_zero_guard(self) -> None:
        from financialagents.agents.trader.position_sizing import fixed_fractional_size

        pct, shares = fixed_fractional_size(
            capital=100_000,
            risk_per_trade_pct=0.02,
            entry_price=100.0,
            stop_loss=100.0,
        )
        assert pct == 0.0 and shares == 0

    def test_fixed_fractional_negative_capital(self) -> None:
        from financialagents.agents.trader.position_sizing import fixed_fractional_size

        pct, shares = fixed_fractional_size(
            capital=-100.0,
            risk_per_trade_pct=0.02,
            entry_price=100.0,
            stop_loss=95.0,
        )
        assert pct == 0.0 and shares == 0


class TestFloorRounding:

    def test_floor_rounding_atr_based(self) -> None:
        from financialagents.agents.trader.position_sizing import atr_based_size

        pct, shares = atr_based_size(
            capital=100_000,
            risk_per_trade_pct=0.02095,
            atr=50.0,
            atr_multiplier=2.0,
            entry_price=2450.0,
        )
        assert shares == 20, f" floor violation: got {shares}, expected 20"
        expected = math.floor((100_000 * 0.02095) / 100.0)
        assert shares == expected

    def test_floor_rounding_fixed_fractional(self) -> None:
        from financialagents.agents.trader.position_sizing import fixed_fractional_size

        pct, shares = fixed_fractional_size(
            capital=100_000,
            risk_per_trade_pct=0.02096,
            entry_price=100.0,
            stop_loss=95.0,
        )
        expected = math.floor(2096.0 / 5.0)
        assert shares == expected == 419
