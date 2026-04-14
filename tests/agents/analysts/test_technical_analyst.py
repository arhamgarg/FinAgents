from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import pydantic
from financialagents.agents.analysts.schemas import (
    FundamentalsAnalystReport,
    NewsReport,
    SentimentAnalystReport,
    TechnicalReport,
)
from financialagents.dataflows.data_result import DataResult



@pytest.fixture
def ohlcv_250() -> pd.DataFrame:
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(250) * 0.5)
    dates = pd.date_range("2024-01-01", periods=250, freq="B")
    df = pd.DataFrame(
        {
            "open": price + np.random.randn(250) * 0.3,
            "high": price + np.abs(np.random.randn(250)) * 0.5,
            "low": price - np.abs(np.random.randn(250)) * 0.5,
            "close": price,
            "volume": np.random.randint(100_000, 1_000_000, 250).astype(float),
        },
        index=dates,
    )
    return df


@pytest.fixture
def ohlcv_50() -> pd.DataFrame:
    np.random.seed(7)
    price = 200 + np.cumsum(np.random.randn(50) * 0.5)
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    df = pd.DataFrame(
        {
            "open": price + np.random.randn(50) * 0.3,
            "high": price + np.abs(np.random.randn(50)) * 0.5,
            "low": price - np.abs(np.random.randn(50)) * 0.5,
            "close": price,
            "volume": np.random.randint(100_000, 1_000_000, 50).astype(float),
        },
        index=dates,
    )
    return df


@pytest.fixture
def agent_state_basic():
    from financialagents.agents.utils.agent_states import AgentState
    return AgentState(ticker="RELIANCE", date="2024-12-31")


@pytest.fixture
def valid_technical_report() -> TechnicalReport:
    return TechnicalReport(
        rsi=55.3,
        rsi_interpretation="neutral — neither overbought nor oversold",
        macd_signal="bullish_crossover",
        macd_line=1.2,
        macd_hist=0.5,
        bollinger_position="middle_band",
        bb_upper=115.0,
        bb_lower=95.0,
        ma_crossover="golden_cross",
        sma_50=102.5,
        sma_200=98.3,
        ema_20=104.1,
        current_price=108.5,
        trend_assessment="Bullish trend supported by MACD and MA crossover",
        support_levels=[95.0, 90.0],
        resistance_levels=[115.0, 120.0],
        confidence=0.75,
        data_sources=["angelone"],
        notes="All indicators computed from 250 trading days of OHLCV data.",
    )



class TestComputeTechnicalIndicators:
    def test_returns_expected_keys(self, ohlcv_250: pd.DataFrame):
        from financialagents.agents.analysts.technical_analyst import compute_technical_indicators

        result = compute_technical_indicators(ohlcv_250)

        expected_keys = {
            "rsi",
            "macd_line",
            "macd_signal_val",
            "macd_hist",
            "bb_upper",
            "bb_lower",
            "bb_mid",
            "sma_50",
            "sma_200",
            "ema_20",
            "current_price",
        }
        assert set(result.keys()) == expected_keys

    def test_values_are_floats_or_none(self, ohlcv_250: pd.DataFrame):
        from financialagents.agents.analysts.technical_analyst import compute_technical_indicators

        result = compute_technical_indicators(ohlcv_250)

        for key, val in result.items():
            assert val is None or isinstance(val, float), (
                f"Key '{key}' has unexpected type {type(val)}: {val}"
            )
            if isinstance(val, float):
                assert not (val != val), f"Key '{key}' contains NaN"

    def test_with_250_rows_all_indicators_non_none(self, ohlcv_250: pd.DataFrame):
        from financialagents.agents.analysts.technical_analyst import compute_technical_indicators

        result = compute_technical_indicators(ohlcv_250)

        for key, val in result.items():
            assert val is not None, f"Expected non-None for '{key}' with 250 rows"

    def test_sma_200_none_with_only_50_rows(self, ohlcv_50: pd.DataFrame):
        from financialagents.agents.analysts.technical_analyst import compute_technical_indicators

        result = compute_technical_indicators(ohlcv_50)

        assert result["sma_200"] is None, "SMA-200 should be None with 50 rows"

    def test_current_price_matches_last_close(self, ohlcv_250: pd.DataFrame):
        from financialagents.agents.analysts.technical_analyst import compute_technical_indicators

        result = compute_technical_indicators(ohlcv_250)

        assert result["current_price"] == pytest.approx(
            float(ohlcv_250["close"].iloc[-1]), rel=1e-6
        )

    def test_empty_dataframe_returns_all_none(self):
        from financialagents.agents.analysts.technical_analyst import compute_technical_indicators

        empty_df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )
        result = compute_technical_indicators(empty_df)

        expected_keys = {
            "rsi", "macd_line", "macd_signal_val", "macd_hist",
            "bb_upper", "bb_lower", "bb_mid",
            "sma_50", "sma_200", "ema_20", "current_price",
        }
        assert set(result.keys()) == expected_keys
        for key, val in result.items():
            assert val is None, f"Expected None for '{key}' on empty DataFrame, got {val!r}"



class TestTechnicalAnalystNode:
    @pytest.mark.asyncio
    async def test_returns_technical_report(
        self, ohlcv_250: pd.DataFrame, agent_state_basic, valid_technical_report: TechnicalReport
    ):
        data_result = DataResult(
            status="ok", data=ohlcv_250, source="angelone"
        )

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=valid_technical_report)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        with patch(
            "financialagents.agents.analysts.technical_analyst.get_ohlcv",
            new_callable=AsyncMock,
            return_value=data_result,
        ), patch(
            "financialagents.agents.analysts.technical_analyst.create_llm_client",
            return_value=mock_llm,
        ):
            from financialagents.agents.analysts.technical_analyst import technical_analyst_node

            runnable_config = {"configurable": {}}
            result = await technical_analyst_node(agent_state_basic, runnable_config)

        assert "technical_report" in result
        report = result["technical_report"]
        assert isinstance(report, TechnicalReport)
        assert report.confidence > 0

    @pytest.mark.asyncio
    async def test_handles_ohlcv_error(self, agent_state_basic):
        data_result = DataResult(
            status="error", data=None, error="Angel One API timeout"
        )

        with patch(
            "financialagents.agents.analysts.technical_analyst.get_ohlcv",
            new_callable=AsyncMock,
            return_value=data_result,
        ):
            from financialagents.agents.analysts.technical_analyst import technical_analyst_node

            with pytest.raises(ValueError, match="Technical analyst requires OHLCV data"):
                await technical_analyst_node(agent_state_basic, {"configurable": {}})

    @pytest.mark.asyncio
    async def test_handles_ohlcv_empty(self, agent_state_basic):
        data_result = DataResult(
            status="empty", data=None
        )

        with patch(
            "financialagents.agents.analysts.technical_analyst.get_ohlcv",
            new_callable=AsyncMock,
            return_value=data_result,
        ):
            from financialagents.agents.analysts.technical_analyst import technical_analyst_node

            with pytest.raises(ValueError, match="Technical analyst requires OHLCV data"):
                await technical_analyst_node(agent_state_basic, {"configurable": {}})

    @pytest.mark.asyncio
    async def test_retries_llm_once_on_failure(
        self, ohlcv_250: pd.DataFrame, agent_state_basic, valid_technical_report: TechnicalReport
    ):
        data_result = DataResult(
            status="ok", data=ohlcv_250, source="angelone"
        )

        mock_ainvoke = AsyncMock(
            side_effect=[RuntimeError("LLM timeout"), valid_technical_report]
        )
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = mock_ainvoke
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        with patch(
            "financialagents.agents.analysts.technical_analyst.get_ohlcv",
            new_callable=AsyncMock,
            return_value=data_result,
        ), patch(
            "financialagents.agents.analysts.technical_analyst.create_llm_client",
            return_value=mock_llm,
        ):
            from financialagents.agents.analysts.technical_analyst import technical_analyst_node

            result = await technical_analyst_node(agent_state_basic, {"configurable": {}})

        assert "technical_report" in result
        assert isinstance(result["technical_report"], TechnicalReport)
        assert mock_ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_llm_retry_exhausted(
        self, ohlcv_250: pd.DataFrame, agent_state_basic
    ):
        data_result = DataResult(
            status="ok", data=ohlcv_250, source="angelone"
        )

        mock_ainvoke = AsyncMock(
            side_effect=[RuntimeError("first failure"), RuntimeError("second failure")]
        )
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = mock_ainvoke
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        with patch(
            "financialagents.agents.analysts.technical_analyst.get_ohlcv",
            new_callable=AsyncMock,
            return_value=data_result,
        ), patch(
            "financialagents.agents.analysts.technical_analyst.create_llm_client",
            return_value=mock_llm,
        ):
            from financialagents.agents.analysts.technical_analyst import technical_analyst_node

            with pytest.raises(RuntimeError):
                await technical_analyst_node(agent_state_basic, {"configurable": {}})

    @pytest.mark.asyncio
    async def test_data_sources_contains_ohlcv_source(
        self, ohlcv_250: pd.DataFrame, agent_state_basic, valid_technical_report: TechnicalReport
    ):
        data_result = DataResult(
            status="ok", data=ohlcv_250, source="angelone"
        )

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=valid_technical_report)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        with patch(
            "financialagents.agents.analysts.technical_analyst.get_ohlcv",
            new_callable=AsyncMock,
            return_value=data_result,
        ), patch(
            "financialagents.agents.analysts.technical_analyst.create_llm_client",
            return_value=mock_llm,
        ):
            from financialagents.agents.analysts.technical_analyst import technical_analyst_node

            result = await technical_analyst_node(agent_state_basic, {"configurable": {}})

        report = result["technical_report"]
        assert "angelone" in report.data_sources



class TestSchemaConfidenceValidation:

    @pytest.mark.parametrize("bad_value", [1.1, 99.9, -0.01, -1.0, 2.0])
    def test_technical_report_rejects_out_of_range_confidence(self, bad_value: float):
        with pytest.raises(pydantic.ValidationError):
            TechnicalReport(confidence=bad_value)

    @pytest.mark.parametrize("bad_value", [1.1, 99.9, -0.01, -1.0, 2.0])
    def test_fundamentals_report_rejects_out_of_range_confidence(self, bad_value: float):
        with pytest.raises(pydantic.ValidationError):
            FundamentalsAnalystReport(confidence=bad_value)

    @pytest.mark.parametrize("bad_value", [1.1, 99.9, -0.01, -1.0, 2.0])
    def test_news_report_rejects_out_of_range_confidence(self, bad_value: float):
        with pytest.raises(pydantic.ValidationError):
            NewsReport(confidence=bad_value)

    @pytest.mark.parametrize("bad_value", [1.1, 99.9, -0.01, -1.0, 2.0])
    def test_sentiment_report_rejects_out_of_range_confidence(self, bad_value: float):
        with pytest.raises(pydantic.ValidationError):
            SentimentAnalystReport(confidence=bad_value)

    @pytest.mark.parametrize("good_value", [0.0, 0.5, 1.0])
    def test_technical_report_accepts_valid_confidence(self, good_value: float):
        report = TechnicalReport(confidence=good_value)
        assert report.confidence == good_value
