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
