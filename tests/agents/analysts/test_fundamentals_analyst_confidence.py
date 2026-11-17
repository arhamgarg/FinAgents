from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finagents.agents.analysts.schemas import FundamentalsAnalystReport
from finagents.config import Settings
from finagents.dataflows.data_result import DataResult
from finagents.dataflows.fundamentals_schema import (
    DebtBalance,
    FundamentalsReport,
    Profitability,
    Shareholding,
    Valuation,
)
from finagents.dataflows.interface import use_data_sources


@pytest.fixture
def agent_state():
    from finagents.agents.utils.agent_states import AgentState

    return AgentState(ticker="TESTCO", date="2024-12-31")


@pytest.fixture
def happy_path_data_layer() -> FundamentalsReport:
    return FundamentalsReport(
        ticker="TESTCO",
        valuation=Valuation(
            pe_ratio=20.0,
            pb_ratio=3.0,
            ev_ebitda=10.0,
            market_cap_cr=50_000.0,
            dividend_yield=1.0,
        ),
        profitability=Profitability(roce=18.0, roe=15.0, operating_margin=20.0, net_margin=12.0),
        debt_balance=DebtBalance(debt_to_equity=0.5, interest_coverage=6.0, current_ratio=1.5),
        shareholding=Shareholding(
            promoter_holding_pct=55.0,
            promoter_pledge_pct=0.0,
            fii_holding_pct=20.0,
            dii_holding_pct=15.0,
        ),
        quarterly_trend=[],
    )


def _build_report(
    *,
    pe_ratio: float | None,
    pb_ratio: float | None,
    roce: float | None,
    debt_to_equity: float | None,
    promoter_holding_pct: float | None,
    llm_emitted_confidence: float = 0.0,
) -> FundamentalsAnalystReport:
    return FundamentalsAnalystReport(
        pe_ratio=pe_ratio,
        pb_ratio=pb_ratio,
        roce=roce,
        debt_to_equity=debt_to_equity,
        promoter_holding_pct=promoter_holding_pct,
        confidence=llm_emitted_confidence,
    )


async def _run_node_with_mocks(
    agent_state,
    data_result: DataResult,
    mock_report: FundamentalsAnalystReport | None,
):
    mock_llm = MagicMock()
    if mock_report is not None:
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_report)
        mock_llm.with_structured_output.return_value = mock_structured_llm
    with (
        use_data_sources(fundamentals=AsyncMock(return_value=data_result)),
        patch(
            "finagents.agents.analysts.fundamentals_analyst.create_llm_client",
            return_value=mock_llm,
        ),
    ):
        from finagents.agents.analysts.fundamentals_analyst import (
            fundamentals_analyst_node,
        )

        return await fundamentals_analyst_node(
            agent_state, {"configurable": {"settings": Settings()}}
        )


class TestFundamentalsAnalystConfidence:
    @pytest.mark.asyncio
    async def test_full_five_fields_yields_1_0(self, agent_state, happy_path_data_layer):
        data_result = DataResult(status="ok", data=happy_path_data_layer, source="screener")
        mock_report = _build_report(
            pe_ratio=20.0,
            pb_ratio=3.0,
            roce=18.0,
            debt_to_equity=0.5,
            promoter_holding_pct=55.0,
            llm_emitted_confidence=0.2,
        )
        result = await _run_node_with_mocks(agent_state, data_result, mock_report)
        report = result["fundamentals_report"]
        assert report.confidence == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_screener_miss_stays_zero(self, agent_state):
        data_result = DataResult(status="empty", data=None, error="Screener returned 404")
        result = await _run_node_with_mocks(agent_state, data_result, None)
        report = result["fundamentals_report"]
        assert report.confidence == 0.0

    @pytest.mark.asyncio
    async def test_partial_three_fields_yields_three_fifths(
        self, agent_state, happy_path_data_layer
    ):
        data_result = DataResult(status="ok", data=happy_path_data_layer, source="screener")
        mock_report = _build_report(
            pe_ratio=20.0,
            pb_ratio=None,
            roce=18.0,
            debt_to_equity=None,
            promoter_holding_pct=55.0,
            llm_emitted_confidence=0.9,
        )
        result = await _run_node_with_mocks(agent_state, data_result, mock_report)
        report = result["fundamentals_report"]
        assert report.confidence == pytest.approx(3 / 5)
