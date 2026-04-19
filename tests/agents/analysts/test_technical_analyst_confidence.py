from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from financialagents.agents.analysts.schemas import TechnicalReport
from financialagents.dataflows.data_result import DataResult




@pytest.fixture
def ohlcv_250() -> pd.DataFrame:
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(250) * 0.5)
    dates = pd.date_range("2024-01-01", periods=250, freq="B")
    return pd.DataFrame(
        {
            "open": price + np.random.randn(250) * 0.3,
            "high": price + np.abs(np.random.randn(250)) * 0.5,
            "low": price - np.abs(np.random.randn(250)) * 0.5,
            "close": price,
            "volume": np.random.randint(100_000, 1_000_000, 250).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def agent_state():
    from financialagents.agents.utils.agent_states import AgentState

    return AgentState(ticker="TESTCO", date="2024-12-31")


def _build_report(
    *,
    current_price: float | None,
    rsi: float | None,
    macd_line: float | None,
    macd_hist: float | None,
    bb_upper: float | None,
    bb_lower: float | None,
    sma_50: float | None,
    sma_200: float | None,
    atr_14: float | None,
    llm_emitted_confidence: float = 0.0,
) -> TechnicalReport:
    return TechnicalReport(
        current_price=current_price,
        rsi=rsi,
        macd_line=macd_line,
        macd_hist=macd_hist,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        sma_50=sma_50,
        sma_200=sma_200,
        atr_14=atr_14,
        confidence=llm_emitted_confidence,
    )


async def _run_node_with_mocks(
    agent_state, ohlcv_df: pd.DataFrame, mock_report: TechnicalReport
):
    data_result = DataResult(status="ok", data=ohlcv_df, source="angelone")
    mock_structured_llm = MagicMock()
    mock_structured_llm.ainvoke = AsyncMock(return_value=mock_report)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    indicators_dict = {
        "rsi": mock_report.rsi,
        "macd_line": mock_report.macd_line,
        "macd_signal_val": None,
        "macd_hist": mock_report.macd_hist,
        "bb_upper": mock_report.bb_upper,
        "bb_lower": mock_report.bb_lower,
        "bb_mid": None,
        "sma_50": mock_report.sma_50,
        "sma_200": mock_report.sma_200,
        "ema_20": None,
        "current_price": mock_report.current_price,
        "atr_14": mock_report.atr_14,
    }
    with patch(
        "financialagents.agents.analysts.technical_analyst.get_ohlcv",
        new_callable=AsyncMock,
        return_value=data_result,
    ), patch(
        "financialagents.agents.analysts.technical_analyst.create_llm_client",
        return_value=mock_llm,
    ), patch(
        "financialagents.agents.analysts.technical_analyst.compute_technical_indicators",
        return_value=indicators_dict,
    ), patch(
        "financialagents.dataflows.surveillance_client.get_surveillance_data",
        new_callable=AsyncMock,
        return_value=(None, False),
    ):
        from financialagents.agents.analysts.technical_analyst import (
            technical_analyst_node,
        )

        return await technical_analyst_node(agent_state, {"configurable": {}})




class TestTechnicalAnalystConfidence:
    @pytest.mark.asyncio
    async def test_full_nine_fields_yields_1_0(self, agent_state, ohlcv_250):
        mock_report = _build_report(
            current_price=100.0,
            rsi=55.0,
            macd_line=1.2,
            macd_hist=0.5,
            bb_upper=110.0,
            bb_lower=95.0,
            sma_50=102.0,
            sma_200=98.0,
            atr_14=2.5,
            llm_emitted_confidence=0.0,
        )
        result = await _run_node_with_mocks(agent_state, ohlcv_250, mock_report)
        report = result["technical_report"]
        assert report.confidence == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_atr_14_none_yields_eight_ninths(self, agent_state, ohlcv_250):
        mock_report = _build_report(
            current_price=100.0,
            rsi=55.0,
            macd_line=1.2,
            macd_hist=0.5,
            bb_upper=110.0,
            bb_lower=95.0,
            sma_50=102.0,
            sma_200=98.0,
            atr_14=None,
            llm_emitted_confidence=0.95,
        )
        result = await _run_node_with_mocks(agent_state, ohlcv_250, mock_report)
        report = result["technical_report"]
        assert report.confidence == pytest.approx(8 / 9)

    @pytest.mark.asyncio
    async def test_all_none_yields_0_0(self, agent_state, ohlcv_250):
        mock_report = _build_report(
            current_price=None,
            rsi=None,
            macd_line=None,
            macd_hist=None,
            bb_upper=None,
            bb_lower=None,
            sma_50=None,
            sma_200=None,
            atr_14=None,
            llm_emitted_confidence=0.7,
        )
        result = await _run_node_with_mocks(agent_state, ohlcv_250, mock_report)
        report = result["technical_report"]
        assert report.confidence == 0.0

    @pytest.mark.asyncio
    async def test_two_fields_present_yields_two_ninths(self, agent_state, ohlcv_250):
        mock_report = _build_report(
            current_price=100.0,
            rsi=55.0,
            macd_line=None,
            macd_hist=None,
            bb_upper=None,
            bb_lower=None,
            sma_50=None,
            sma_200=None,
            atr_14=None,
            llm_emitted_confidence=0.5,
        )
        result = await _run_node_with_mocks(agent_state, ohlcv_250, mock_report)
        report = result["technical_report"]
        assert report.confidence == pytest.approx(2 / 9)
