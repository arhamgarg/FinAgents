from __future__ import annotations

import warnings

import pytest


class TestTechnicalReport:
    def test_validates_with_all_fields_populated(self):
        from financialagents.agents.analysts.schemas import TechnicalReport

        report = TechnicalReport(
            rsi=62.4,
            macd_signal="bullish_crossover",
            confidence=0.85,
            data_sources=["angelone"],
        )
        assert report.rsi == pytest.approx(62.4)
        assert report.macd_signal == "bullish_crossover"
        assert report.confidence == pytest.approx(0.85)
        assert report.data_sources == ["angelone"]

    def test_validates_with_all_optional_fields_none(self):
        from financialagents.agents.analysts.schemas import TechnicalReport

        report = TechnicalReport()
        assert report.rsi is None
        assert report.macd_signal == ""
        assert report.confidence == pytest.approx(0.0)
        assert report.data_sources == []


class TestFundamentalsAnalystReport:
    def test_validates_with_key_fields(self):
        from financialagents.agents.analysts.schemas import FundamentalsAnalystReport

        report = FundamentalsAnalystReport(
            pe_ratio=18.2,
            roce=22.5,
            debt_to_equity=0.3,
            confidence=0.9,
        )
        assert report.pe_ratio == pytest.approx(18.2)
        assert report.roce == pytest.approx(22.5)
        assert report.debt_to_equity == pytest.approx(0.3)
        assert report.confidence == pytest.approx(0.9)


class TestNewsReport:
    def test_validates_with_article_count_and_direction(self):
        from financialagents.agents.analysts.schemas import NewsReport

        report = NewsReport(
            article_count=5,
            sentiment_direction="positive",
            confidence=0.7,
        )
        assert report.article_count == 5
        assert report.sentiment_direction == "positive"
        assert report.confidence == pytest.approx(0.7)


class TestResearchBrief:
    def test_validates_with_key_findings_and_confidences(self):
        from financialagents.agents.managers.schemas import ResearchBrief

        brief = ResearchBrief(
            key_findings=[],
            consensus_view="bullish",
            conflicts=["news vs sentiment"],
            analyst_confidences={"technical": 0.85, "fundamentals": 0.9},
        )
        assert brief.consensus_view == "bullish"
        assert brief.analyst_confidences["fundamentals"] == pytest.approx(0.9)


class TestFinding:
    def test_validates_finding(self):
        from financialagents.agents.managers.schemas import Finding

        finding = Finding(
            analyst="technical",
            key_point="RSI oversold",
            supporting_data="RSI=28",
            weight=0.25,
        )
        assert finding.analyst == "technical"
        assert finding.key_point == "RSI oversold"
        assert finding.weight == pytest.approx(0.25)


class TestAgentStateTypedFields:
    def test_accepts_technical_report_instance(self):
        from financialagents.agents.analysts.schemas import TechnicalReport
        from financialagents.agents.utils.agent_states import AgentState

        mr = TechnicalReport(confidence=0.5, data_sources=["test"])
        state = AgentState(technical_report=mr)
        assert state.technical_report is mr

    def test_accepts_all_typed_report_models(self):
        from financialagents.agents.analysts.schemas import (
            FundamentalsAnalystReport,
            TechnicalReport,
            NewsReport,
        )
        from financialagents.agents.managers.schemas import ResearchBrief
        from financialagents.agents.utils.agent_states import AgentState

        state = AgentState(
            technical_report=TechnicalReport(confidence=0.5),
            fundamentals_report=FundamentalsAnalystReport(confidence=0.8),
            news_report=NewsReport(confidence=0.6),
            research_summary=ResearchBrief(consensus_view="neutral"),
        )
        assert state.technical_report is not None
        assert state.fundamentals_report is not None
        assert state.news_report is not None
        assert state.research_summary is not None


class TestSettingsAnalystWeights:
    def test_analyst_weights_default_values(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from financialagents.config import Settings

            s = Settings(_env_file=None)
        assert s.analyst_weights == {
            "technical": 0.303,
            "fundamentals": 0.455,
            "news": 0.242,
        }
        assert abs(sum(s.analyst_weights.values()) - 1.0) < 1e-9
