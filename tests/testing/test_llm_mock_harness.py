from __future__ import annotations

import pytest
from pydantic import BaseModel

from financialagents.config import Settings
from financialagents.llm_clients.factory import create_llm_client
from financialagents.testing.llm_mock import (
    CALL_SITE_SCHEMAS,
    LLMHarnessExhausted,
    LLMHarnessMissing,
    LLMHarnessSchemaError,
    LLMMockHarness,
    failures,
    fixtures,
)




@pytest.mark.parametrize("call_site", list(CALL_SITE_SCHEMAS.keys()))
def test_harness_routes_per_call_site(call_site: str) -> None:
    settings = Settings(_env_file=None)
    canonical = fixtures.HAPPY_PATH_DEFAULTS[call_site]
    schema: type[BaseModel] = CALL_SITE_SCHEMAS[call_site]

    with LLMMockHarness() as h:
        h.set(call_site, canonical)
        llm = create_llm_client(
            "deep_think",
            settings,
            call_site=call_site,
        )
        structured = llm.with_structured_output(schema)
        result = structured.invoke([])

    assert result == canonical
    assert isinstance(result, schema)




def test_unset_call_site_raises() -> None:
    with LLMMockHarness():
        llm = create_llm_client(
            "deep_think",
            Settings(_env_file=None),
            call_site="trader",
        )
        with pytest.raises(LLMHarnessMissing):
            llm.with_structured_output(CALL_SITE_SCHEMAS["trader"]).invoke([])




def test_set_wrong_schema_raises() -> None:
    with LLMMockHarness() as h, pytest.raises(LLMHarnessSchemaError):
        h.set("bull_researcher", fixtures.NEWS_REPORT_DEFAULT)




def test_exhausted_sequence_raises() -> None:
    with LLMMockHarness() as h:
        h.set_sequence("trader", [fixtures.TRADER_SIGNAL_DEFAULT])
        settings = Settings(_env_file=None)
        llm = create_llm_client(
            "deep_think",
            settings,
            call_site="trader",
        )
        structured = llm.with_structured_output(CALL_SITE_SCHEMAS["trader"])
        _first = structured.invoke([])
        assert _first is fixtures.TRADER_SIGNAL_DEFAULT
        with pytest.raises(LLMHarnessExhausted):
            structured.invoke([])




def test_length_error_raises() -> None:
    import openai

    err = failures.length_error()
    assert isinstance(err, openai.LengthFinishReasonError)
    assert failures.length_error.SENTINEL == "LengthFinishReasonError"
    assert "LengthFinishReasonError" in repr(err)




def test_validation_error_raises() -> None:
    from pydantic import ValidationError

    err = failures.validation_error()
    assert isinstance(err, ValidationError)
    assert failures.validation_error.SENTINEL == "ValidationError"




def test_runtime_error_raises() -> None:
    err = failures.runtime_error_after_retry()
    assert isinstance(err, RuntimeError)
    assert str(err) == "simulated LLM failure after retry (harness)"
    assert "simulated LLM failure after retry" in str(err)
    assert failures.runtime_error_after_retry.SENTINEL == "LLM error after retry"




def test_retry_then_succeed_sequence() -> None:
    sequence = failures.retry_then_succeed(fixtures.TRADER_SIGNAL_DEFAULT)
    assert isinstance(sequence, list)
    assert len(sequence) == 2
    assert isinstance(sequence[0], BaseException)
    assert sequence[1] is fixtures.TRADER_SIGNAL_DEFAULT

    with LLMMockHarness() as h:
        h.set_sequence("trader", sequence)
        llm = create_llm_client(
            "deep_think",
            Settings(_env_file=None),
            call_site="trader",
        )
        structured = llm.with_structured_output(CALL_SITE_SCHEMAS["trader"])
        with pytest.raises(BaseException):  # noqa: B017,PT011
            structured.invoke([])
        result = structured.invoke([])
        assert result is fixtures.TRADER_SIGNAL_DEFAULT




def test_provider_class_assertion() -> None:
    import langchain_groq

    with pytest.raises(RuntimeError, match="provider client instantiated under hermetic test"):
        langchain_groq.ChatGroq(model="x", api_key="y")
