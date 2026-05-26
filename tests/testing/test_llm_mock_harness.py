from __future__ import annotations

from typing import cast

import pytest
from pydantic import BaseModel

from financialagents.config import Settings
from financialagents.llm_clients.factory import create_llm_client
from financialagents.testing.llm_mock import (
    CALL_SITE_SCHEMAS,
    LLMHarnessError,
    LLMHarnessExhausted,
    LLMHarnessMissing,
    LLMHarnessSchemaError,
    LLMMockHarness,
    failures,
    fixtures,
)
from financialagents.testing.llm_mock._call_sites import CallSite




@pytest.mark.parametrize("call_site", list(CALL_SITE_SCHEMAS.keys()))
def test_harness_routes_per_call_site(call_site: str) -> None:
    settings = Settings(_env_file=None, openrouter_api_key="fake")
    site = cast(CallSite, call_site)
    canonical = fixtures.HAPPY_PATH_DEFAULTS[site]
    schema: type[BaseModel] = CALL_SITE_SCHEMAS[site]

    with LLMMockHarness() as h:
        h.set(site, canonical)
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
            Settings(_env_file=None, openrouter_api_key="fake"),
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
        settings = Settings(_env_file=None, openrouter_api_key="fake")
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
            Settings(_env_file=None, openrouter_api_key="fake"),
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




def test_nested_harness_restores_outer_sentinel() -> None:
    _factory = LLMMockHarness.__enter__.__globals__["_factory"]

    outer = LLMMockHarness()
    inner = LLMMockHarness()

    with outer:
        assert _factory._ACTIVE_HARNESS is outer

        with inner:
            assert _factory._ACTIVE_HARNESS is inner

        assert _factory._ACTIVE_HARNESS is outer

    assert _factory._ACTIVE_HARNESS is None



def test_exception_hierarchy_invariant() -> None:
    assert issubclass(LLMHarnessMissing, LLMHarnessError), (
        "LLMHarnessMissing must subclass LLMHarnessError"
    )
    assert issubclass(LLMHarnessSchemaError, LLMHarnessError)
    assert issubclass(LLMHarnessExhausted, LLMHarnessError)


@pytest.mark.parametrize("method_name", ["set", "set_sequence"])
def test_off_list_call_site_raises_llm_harness_missing(method_name: str) -> None:
    if method_name == "set":
        payload = fixtures.TECH_REPORT_DEFAULT
    else:
        payload = [fixtures.TECH_REPORT_DEFAULT]

    with LLMMockHarness() as h:
        with pytest.raises(LLMHarnessMissing) as exc_info:
            getattr(h, method_name)("not_a_real_call_site", payload)

    assert "not_a_real_call_site" in str(exc_info.value)
    assert "technical_analyst" in str(exc_info.value) or "trader" in str(exc_info.value)




@pytest.mark.parametrize(
    "call_site,fixture_name",
    [
        ("aggressive_risk", "NEUTRAL_DEFAULT"),
        ("aggressive_risk", "CONSERVATIVE_DEFAULT"),
        ("neutral_risk", "AGGRESSIVE_DEFAULT"),
        ("neutral_risk", "CONSERVATIVE_DEFAULT"),
        ("conservative_risk", "AGGRESSIVE_DEFAULT"),
        ("conservative_risk", "NEUTRAL_DEFAULT"),
    ],
)
def test_cross_persona_registration_rejected(
    call_site: str, fixture_name: str
) -> None:
    with LLMMockHarness() as h:
        with pytest.raises(LLMHarnessSchemaError, match="cross-persona") as exc_info:
            h.set(call_site, getattr(fixtures, fixture_name))  # type: ignore[arg-type]

    msg = str(exc_info.value)
    assert call_site in msg


@pytest.mark.parametrize(
    "call_site,fixture_name",
    [
        ("aggressive_risk", "AGGRESSIVE_DEFAULT"),
        ("neutral_risk", "NEUTRAL_DEFAULT"),
        ("conservative_risk", "CONSERVATIVE_DEFAULT"),
    ],
)
def test_same_persona_registration_still_passes(
    call_site: str, fixture_name: str
) -> None:
    with LLMMockHarness() as h:
        h.set(call_site, getattr(fixtures, fixture_name))  # type: ignore[arg-type]


def test_happy_path_full_registration_still_succeeds() -> None:
    h = LLMMockHarness()
    for call_site, instance in fixtures.HAPPY_PATH_DEFAULTS.items():
        h.set(call_site, instance)  # type: ignore[arg-type]
    assert len(h._responses) == 13, (
        f"expected 13 registered call_sites, got {len(h._responses)}"
    )


def test_cross_persona_set_sequence_rejected() -> None:
    with LLMMockHarness() as h:
        with pytest.raises(LLMHarnessSchemaError, match="cross-persona"):
            h.set_sequence(
                "conservative_risk",  # type: ignore[arg-type]
                [fixtures.AGGRESSIVE_DEFAULT],
            )


def test_persona_origin_attr_name_locked() -> None:
    from financialagents.testing.llm_mock import _harness as _harness_mod
    from financialagents.testing.llm_mock import fixtures as fixtures_mod

    assert hasattr(fixtures_mod, "_LLM_MOCK_PERSONA_ORIGIN_ATTR"), (
        "fixtures._LLM_MOCK_PERSONA_ORIGIN_ATTR has been renamed or removed — "
        "update _harness.py to import the new name lock)."
    )

    assert fixtures_mod._LLM_MOCK_PERSONA_ORIGIN_ATTR == "_LLM_MOCK_PERSONA_ORIGIN"

    attr = fixtures_mod._LLM_MOCK_PERSONA_ORIGIN_ATTR
    assert getattr(fixtures_mod.AGGRESSIVE_DEFAULT, attr) == "aggressive_risk"
    assert getattr(fixtures_mod.NEUTRAL_DEFAULT, attr) == "neutral_risk"
    assert getattr(fixtures_mod.CONSERVATIVE_DEFAULT, attr) == "conservative_risk"

    assert hasattr(_harness_mod, "_LLM_MOCK_PERSONA_ORIGIN_ATTR"), (
        "_harness.py must import _LLM_MOCK_PERSONA_ORIGIN_ATTR from fixtures "
        " — hardcoding the string literal there is the silent-drift trap."
    )
    assert _harness_mod._LLM_MOCK_PERSONA_ORIGIN_ATTR == fixtures_mod._LLM_MOCK_PERSONA_ORIGIN_ATTR




def test_with_structured_output_schema_mismatch_raises() -> None:
    from financialagents.agents.analysts.schemas import TechnicalReport

    with LLMMockHarness() as h:
        h.set("trader", fixtures.TRADER_SIGNAL_DEFAULT)
        llm = create_llm_client(
            "deep_think",
            Settings(_env_file=None, openrouter_api_key="fake"),
            call_site="trader",
        )
        with pytest.raises(LLMHarnessSchemaError) as exc_info:
            llm.with_structured_output(TechnicalReport)

    msg = str(exc_info.value)
    assert "TechnicalReport" in msg, f"missing TechnicalReport in: {msg}"
    assert "TradingSignal" in msg, f"missing TradingSignal in: {msg}"
    assert "trader" in msg, f"missing call_site 'trader' in: {msg}"


def test_with_structured_output_matching_schema_still_returns_runnable() -> None:
    from langchain_core.runnables import Runnable

    trading_signal_cls = CALL_SITE_SCHEMAS["trader"]

    with LLMMockHarness() as h:
        h.set("trader", fixtures.TRADER_SIGNAL_DEFAULT)
        llm = create_llm_client(
            "deep_think",
            Settings(_env_file=None, openrouter_api_key="fake"),
            call_site="trader",
        )
        runnable = llm.with_structured_output(trading_signal_cls)
        assert isinstance(runnable, Runnable)
        result = runnable.invoke([])
        assert result is fixtures.TRADER_SIGNAL_DEFAULT


def test_with_structured_output_survives_sys_modules_purge() -> None:
    import importlib  # noqa: PLC0415
    import sys  # noqa: PLC0415

    original_cls = CALL_SITE_SCHEMAS["trader"]

    mod_name = "financialagents.agents.trader.schemas"
    saved = sys.modules.pop(mod_name, None)
    try:
        reloaded_mod = importlib.import_module(mod_name)
        reimported_cls = reloaded_mod.TradingSignal

        assert reimported_cls is not original_cls, (
            "test setup failed: re-import did not produce a distinct class "
            "object — the fragility is not being exercised"
        )

        assert reimported_cls.__qualname__ == original_cls.__qualname__
        assert reimported_cls.__module__ == original_cls.__module__

        with LLMMockHarness() as h:
            h.set("trader", fixtures.TRADER_SIGNAL_DEFAULT)
            llm = create_llm_client(
                "deep_think",
                Settings(_env_file=None, openrouter_api_key="fake"),
                call_site="trader",
            )
            runnable = llm.with_structured_output(reimported_cls)
            assert runnable is not None
    finally:
        if saved is not None:
            sys.modules[mod_name] = saved




def test_with_structured_output_runnable_bound_to_producer_harness() -> None:
    trading_signal_cls = CALL_SITE_SCHEMAS["trader"]

    a_signal = trading_signal_cls(
        action="BUY",
        conviction=0.9,
        entry_price=2500.0,
        stop_loss=2450.0,
        target_1=2600.0,
        target_2=2700.0,
        position_size_pct=0.03,
        position_size_shares=30,
        risk_reward_ratio=2.0,
        rationale="outer harness fixture",
        time_horizon="positional",
    )
    b_signal = trading_signal_cls(
        action="SELL",
        conviction=0.4,
        entry_price=2400.0,
        stop_loss=2450.0,
        target_1=2300.0,
        target_2=2200.0,
        position_size_pct=0.02,
        position_size_shares=20,
        risk_reward_ratio=1.0,
        rationale="inner harness fixture",
        time_horizon="positional",
    )

    with LLMMockHarness() as outer:
        outer.set("trader", a_signal)
        outer_llm = create_llm_client(
            "deep_think",
            Settings(_env_file=None, openrouter_api_key="fake"),
            call_site="trader",
        )
        outer_runnable = outer_llm.with_structured_output(trading_signal_cls)

        first = outer_runnable.invoke([])
        assert first is a_signal

        with LLMMockHarness() as inner:
            inner.set("trader", b_signal)

            inner_llm = create_llm_client(
                "deep_think",
                Settings(_env_file=None, openrouter_api_key="fake"),
                call_site="trader",
            )
            inner_runnable = inner_llm.with_structured_output(trading_signal_cls)

            assert inner_runnable.invoke([]) is b_signal

            assert outer_runnable.invoke([]) is a_signal

        assert outer_runnable.invoke([]) is a_signal




def test_with_structured_output_locks_to_producer_not_active_at_call_time() -> None:
    trading_signal_cls = CALL_SITE_SCHEMAS["trader"]

    a_signal = trading_signal_cls(
        action="BUY",
        conviction=0.9,
        entry_price=2500.0,
        stop_loss=2450.0,
        target_1=2600.0,
        target_2=2700.0,
        position_size_pct=0.03,
        position_size_shares=30,
        risk_reward_ratio=2.0,
        rationale="outer-produced signal",
        time_horizon="positional",
    )
    b_signal = trading_signal_cls(
        action="SELL",
        conviction=0.4,
        entry_price=2400.0,
        stop_loss=2450.0,
        target_1=2300.0,
        target_2=2200.0,
        position_size_pct=0.02,
        position_size_shares=20,
        risk_reward_ratio=1.0,
        rationale="inner-produced signal",
        time_horizon="positional",
    )

    with LLMMockHarness() as outer:
        outer.set("trader", a_signal)
        outer_llm = create_llm_client(
            "deep_think",
            Settings(_env_file=None, openrouter_api_key="fake"),
            call_site="trader",
        )

        with LLMMockHarness() as inner:
            inner.set("trader", b_signal)

            outer_runnable = outer_llm.with_structured_output(trading_signal_cls)
            assert outer_runnable.invoke([]) is a_signal, (
                " violation: outer-produced runnable dispatched to the "
                "wrong harness — producer identity was not preserved across "
                "the nested-harness boundary"
            )




def test_dispatch_sequence_lock_under_concurrent_invokes() -> None:
    import asyncio  # noqa: PLC0415

    trading_signal_cls = CALL_SITE_SCHEMAS["trader"]

    n = 8
    sequence = [
        trading_signal_cls(
            action="BUY",
            conviction=0.5 + (i * 0.01),
            entry_price=2500.0,
            stop_loss=2450.0,
            target_1=2600.0,
            target_2=2700.0,
            position_size_pct=0.03,
            position_size_shares=30,
            risk_reward_ratio=2.0,
            rationale=f"sequence item {i}",
            time_horizon="positional",
        )
        for i in range(n)
    ]

    with LLMMockHarness() as h:
        h.set_sequence("trader", list(sequence))  # type: ignore[arg-type]
        llm = create_llm_client(
            "deep_think",
            Settings(_env_file=None, openrouter_api_key="fake"),
            call_site="trader",
        )
        structured = llm.with_structured_output(trading_signal_cls)

        async def _consume_one() -> object:
            return await structured.ainvoke([])

        async def _run_all() -> list[object]:
            return await asyncio.gather(*[_consume_one() for _ in range(n)])

        observed = asyncio.run(_run_all())

    assert {id(o) for o in observed} == {id(s) for s in sequence}, (
        "concurrent ainvoke calls observed a non-monotonic sequence consume "
        "— the threading.Lock is missing or broken"
    )




def test_with_structured_output_include_raw_raises_not_implemented() -> None:
    trading_signal_cls = CALL_SITE_SCHEMAS["trader"]

    with LLMMockHarness() as h:
        h.set("trader", fixtures.TRADER_SIGNAL_DEFAULT)
        llm = create_llm_client(
            "deep_think",
            Settings(_env_file=None, openrouter_api_key="fake"),
            call_site="trader",
        )
        with pytest.raises(NotImplementedError, match="include_raw=True"):
            llm.with_structured_output(trading_signal_cls, include_raw=True)




def test_harness_chat_model_generate_raises_on_bypass() -> None:
    with LLMMockHarness() as h:
        h.set("trader", fixtures.TRADER_SIGNAL_DEFAULT)
        llm = create_llm_client(
            "deep_think",
            Settings(_env_file=None, openrouter_api_key="fake"),
            call_site="trader",
        )
        with pytest.raises(LLMHarnessError, match="_generate was invoked"):
            llm.invoke("hello")
