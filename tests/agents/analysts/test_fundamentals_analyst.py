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
    QuarterRow,
    Shareholding,
    Valuation,
)
from finagents.dataflows.interface import use_data_sources


@pytest.fixture
def agent_state():
    from finagents.agents.utils.agent_states import AgentState

    return AgentState(ticker="RELIANCE", date="2024-12-31")


@pytest.fixture
def fundamentals_report() -> FundamentalsReport:
    return FundamentalsReport(
        ticker="RELIANCE",
        valuation=Valuation(
            pe_ratio=24.5,
            pb_ratio=2.1,
            ev_ebitda=12.3,
            market_cap_cr=1_800_000.0,
            dividend_yield=0.5,
        ),
        profitability=Profitability(
            roce=18.5,
            roe=14.2,
            operating_margin=22.0,
            net_margin=10.5,
        ),
        debt_balance=DebtBalance(
            debt_to_equity=0.35,
            interest_coverage=8.2,
            current_ratio=1.4,
        ),
        shareholding=Shareholding(
            promoter_holding_pct=50.2,
            promoter_pledge_pct=0.0,
            fii_holding_pct=22.5,
            dii_holding_pct=15.3,
        ),
        quarterly_trend=[
            QuarterRow(quarter="Q3 FY24", revenue_cr=235000.0, pat_cr=18000.0),
            QuarterRow(quarter="Q2 FY24", revenue_cr=228000.0, pat_cr=17200.0),
        ],
    )


@pytest.fixture
def valid_fundamentals_analyst_report() -> FundamentalsAnalystReport:
    return FundamentalsAnalystReport(
        pe_ratio=24.5,
        pb_ratio=2.1,
        roce=18.5,
        debt_to_equity=0.35,
        quarterly_result_summary="Revenue growing 3% QoQ; PAT up 5% QoQ",
        promoter_holding_pct=50.2,
        fundamental_assessment=(
            "RELIANCE shows healthy fundamentals. P/E of 24.5 is near Nifty 50 avg. "
            "ROCE of 18.5% exceeds the 15% threshold. Low D/E of 0.35 indicates manageable debt. "
            "Strong promoter holding at 50.2%."
        ),
        confidence=0.85,
        data_sources=["screener"],
        notes="Data sourced from Screener.in. All metrics within expected ranges.",
    )


class TestFundamentalsAnalystNode:
    @pytest.mark.asyncio
    async def test_returns_fundamentals_report(
        self,
        agent_state,
        fundamentals_report: FundamentalsReport,
        valid_fundamentals_analyst_report: FundamentalsAnalystReport,
    ):
        data_result = DataResult(status="ok", data=fundamentals_report, source="screener")

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=valid_fundamentals_analyst_report)
        mock_llm = MagicMock()
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

            result = await fundamentals_analyst_node(
                agent_state, {"configurable": {"settings": Settings()}}
            )

        assert "fundamentals_report" in result
        report = result["fundamentals_report"]
        assert isinstance(report, FundamentalsAnalystReport)
        assert report.pe_ratio == 24.5
        assert report.roce == 18.5
        assert report.debt_to_equity == 0.35
        assert report.confidence > 0

    @pytest.mark.asyncio
    async def test_handles_data_error_with_zero_confidence(self, agent_state):
        data_result = DataResult(
            status="error",
            data=None,
            error="Screener.in connection timeout",
        )

        with use_data_sources(fundamentals=AsyncMock(return_value=data_result)):
            from finagents.agents.analysts.fundamentals_analyst import (
                fundamentals_analyst_node,
            )

            result = await fundamentals_analyst_node(
                agent_state, {"configurable": {"settings": Settings()}}
            )

        assert "fundamentals_report" in result
        report = result["fundamentals_report"]
        assert isinstance(report, FundamentalsAnalystReport)
        assert report.confidence == 0.0

    @pytest.mark.asyncio
    async def test_handles_empty_data_with_zero_confidence(self, agent_state):
        data_result = DataResult(status="empty", data=None)

        with use_data_sources(fundamentals=AsyncMock(return_value=data_result)):
            from finagents.agents.analysts.fundamentals_analyst import (
                fundamentals_analyst_node,
            )

            result = await fundamentals_analyst_node(
                agent_state, {"configurable": {"settings": Settings()}}
            )

        assert "fundamentals_report" in result
        report = result["fundamentals_report"]
        assert report.confidence == 0.0

    @pytest.mark.asyncio
    async def test_retries_llm_once_on_failure(
        self,
        agent_state,
        fundamentals_report: FundamentalsReport,
        valid_fundamentals_analyst_report: FundamentalsAnalystReport,
    ):
        data_result = DataResult(status="ok", data=fundamentals_report, source="screener")

        mock_ainvoke = AsyncMock(
            side_effect=[RuntimeError("LLM timeout"), valid_fundamentals_analyst_report]
        )
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = mock_ainvoke
        mock_llm = MagicMock()
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

            result = await fundamentals_analyst_node(
                agent_state, {"configurable": {"settings": Settings()}}
            )

        assert "fundamentals_report" in result
        assert isinstance(result["fundamentals_report"], FundamentalsAnalystReport)
        assert mock_ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_data_sources_populated_from_result(
        self,
        agent_state,
        fundamentals_report: FundamentalsReport,
        valid_fundamentals_analyst_report: FundamentalsAnalystReport,
    ):
        data_result = DataResult(status="ok", data=fundamentals_report, source="screener")

        report_no_sources = FundamentalsAnalystReport(
            pe_ratio=24.5,
            roce=18.5,
            fundamental_assessment="Solid fundamentals.",
            confidence=0.8,
            data_sources=[],
        )
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=report_no_sources)
        mock_llm = MagicMock()
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

            result = await fundamentals_analyst_node(
                agent_state, {"configurable": {"settings": Settings()}}
            )

        assert "screener" in result["fundamentals_report"].data_sources
