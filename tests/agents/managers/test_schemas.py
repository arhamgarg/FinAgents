from __future__ import annotations

import json
from pathlib import Path

import pydantic
import pytest


@pytest.fixture
def valid_signal():
    from financialagents.agents.trader.schemas import TradingSignal
    return TradingSignal(action="BUY", conviction=0.7)


class TestPortfolioDecision:
    def test_minimal_valid_construction(self, valid_signal) -> None:
        from financialagents.agents.managers.schemas import PortfolioDecision
        pd = PortfolioDecision(signal=valid_signal, approved=True, decision="approve")
        assert pd.approved is True
        assert pd.decision == "approve"
        assert pd.pm_notes == ""
        assert pd.signal.action == "BUY"

    @pytest.mark.parametrize("decision", ["approve", "modify", "veto"])
    def test_all_literal_decisions_accepted(self, valid_signal, decision: str) -> None:
        from financialagents.agents.managers.schemas import PortfolioDecision
        pd = PortfolioDecision(signal=valid_signal, approved=(decision == "approve"), decision=decision)
        assert pd.decision == decision

    def test_invalid_decision_rejected(self, valid_signal) -> None:
        from financialagents.agents.managers.schemas import PortfolioDecision
        with pytest.raises(pydantic.ValidationError):
            PortfolioDecision(signal=valid_signal, approved=False, decision="sell")  # type: ignore[arg-type]

    def test_missing_signal_raises(self) -> None:
        from financialagents.agents.managers.schemas import PortfolioDecision
        with pytest.raises(pydantic.ValidationError):
            PortfolioDecision(approved=True, decision="approve")  # type: ignore[call-arg]

    def test_pm_notes_populated(self, valid_signal) -> None:
        from financialagents.agents.managers.schemas import PortfolioDecision
        pd = PortfolioDecision(
            signal=valid_signal,
            approved=True,
            decision="modify",
            pm_notes="Reduced position size by 30% due to sector concentration",
        )
        assert "sector concentration" in pd.pm_notes


class TestPortfolioContext:
    def test_default_empty_construction(self) -> None:
        from financialagents.agents.managers.schemas import PortfolioContext
        ctx = PortfolioContext()
        assert ctx.holdings == {}
        assert ctx.sector_exposure == {}
        assert ctx.total_value_inr == 0.0

    def test_populated_construction(self) -> None:
        from financialagents.agents.managers.schemas import PortfolioContext
        ctx = PortfolioContext(
            holdings={"RELIANCE": 10, "TCS": 5},
            sector_exposure={"Energy": 0.4, "IT": 0.6},
            total_value_inr=250_000.0,
        )
        assert ctx.holdings["RELIANCE"] == 10
        assert ctx.sector_exposure["IT"] == pytest.approx(0.6)
        assert ctx.total_value_inr == pytest.approx(250_000.0)

    def test_from_seed_file_missing_returns_empty(self, tmp_path: Path) -> None:
        from financialagents.agents.managers.schemas import PortfolioContext
        missing = tmp_path / "does_not_exist.json"
        ctx = PortfolioContext.from_seed_file(missing)
        assert ctx.holdings == {}
        assert ctx.total_value_inr == 0.0

    def test_from_seed_file_malformed_json_returns_empty(self, tmp_path: Path) -> None:
        from financialagents.agents.managers.schemas import PortfolioContext
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        ctx = PortfolioContext.from_seed_file(bad)
        assert ctx == PortfolioContext()

    def test_from_seed_file_wrong_shape_returns_empty(self, tmp_path: Path) -> None:
        from financialagents.agents.managers.schemas import PortfolioContext
        wrong = tmp_path / "wrong_shape.json"
        wrong.write_text(json.dumps({"holdings": "not-a-dict", "total_value_inr": "not-a-float"}))
        ctx = PortfolioContext.from_seed_file(wrong)
        assert ctx == PortfolioContext()

    def test_from_seed_file_valid_returns_populated(self, tmp_path: Path) -> None:
        from financialagents.agents.managers.schemas import PortfolioContext
        good = tmp_path / "portfolio.json"
        good.write_text(
            json.dumps({"holdings": {"TCS": 7}, "sector_exposure": {"IT": 1.0}, "total_value_inr": 42000.0})
        )
        ctx = PortfolioContext.from_seed_file(good)
        assert ctx.holdings == {"TCS": 7}
        assert ctx.total_value_inr == pytest.approx(42000.0)

    def test_committed_seed_file_loads(self) -> None:
        from financialagents.agents.managers.schemas import PortfolioContext
        ctx = PortfolioContext.from_seed_file(Path(".planning/portfolio.json"))
        assert isinstance(ctx.holdings, dict)
        assert isinstance(ctx.total_value_inr, (int, float))
