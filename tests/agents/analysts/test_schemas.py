import pytest

from finagents.agents.analysts.schemas import TechnicalReport


def test_missing_surveillance_and_atr_have_safe_defaults():
    report = TechnicalReport(rsi=55.0, current_price=2450.75)
    assert report.rsi == 55.0
    assert report.current_price == 2450.75
    assert report.atr_14 is None
    assert report.circuit_band_pct is None
    assert report.t2t_flag is False


def test_trader_sizing_inputs_survive_serialization():
    values = dict(current_price=2450.75, atr_14=12.5, circuit_band_pct=0.20, t2t_flag=True)
    report = TechnicalReport.model_validate_json(
        TechnicalReport.model_validate(values).model_dump_json()
    )
    assert {key: getattr(report, key) for key in values} == values


class TestTechnicalReport:
    def test_validates_with_all_fields_populated(self):
        from finagents.agents.analysts.schemas import TechnicalReport

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
        from finagents.agents.analysts.schemas import TechnicalReport

        report = TechnicalReport()
        assert report.rsi is None
        assert report.macd_signal == ""
        assert report.confidence == pytest.approx(0.0)
        assert report.data_sources == []


class TestFundamentalsAnalystReport:
    def test_validates_with_key_fields(self):
        from finagents.agents.analysts.schemas import FundamentalsAnalystReport

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
        from finagents.agents.analysts.schemas import NewsReport

        report = NewsReport(
            article_count=5,
            sentiment_direction="positive",
            confidence=0.7,
        )
        assert report.article_count == 5
        assert report.sentiment_direction == "positive"
        assert report.confidence == pytest.approx(0.7)
