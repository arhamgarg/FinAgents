from __future__ import annotations

import threading
from collections import Counter
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from finagents.llm_clients import factory as _factory
from finagents.testing.llm_mock._call_sites import CALL_SITE_SCHEMAS, CallSite
from finagents.testing.llm_mock.fixtures import _LLM_MOCK_PERSONA_ORIGIN_ATTR



class LLMHarnessError(Exception):
    pass


class LLMHarnessMissing(LLMHarnessError):
    pass


class LLMHarnessSchemaError(LLMHarnessError):
    pass


class LLMHarnessExhausted(LLMHarnessError):
    pass






class _HarnessChatModel(BaseChatModel):

    call_site: str
    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "harness-fake-chat-model"

    def _generate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise LLMHarnessError(
            "_HarnessChatModel._generate was invoked — harness only routes "
            "through with_structured_output. A production node bypassed "
            f"structured output at call_site={self.call_site!r}."
        )

    def with_structured_output(  # type: ignore[override]
        self,
        schema: Any,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        if include_raw:
            raise NotImplementedError(
                "_HarnessChatModel does not yet support include_raw=True. "
                "LangChain returns a {'raw':..., 'parsed':..., 'parsing_error':...} "
                "dict in that mode; the harness returns the bare BaseModel. "
                "If a production node needs include_raw=True, extend the harness "
                "to wrap the dispatched instance in that dict shape."
            )

        call_site = self.call_site
        producer = getattr(self, "_producer_harness", None)
        if producer is None:
            raise LLMHarnessMissing(
                "with_structured_output called on a _HarnessChatModel with "
                f"no producer (call_site={call_site!r}) — the model was "
                "constructed outside _fake_chat_model_for. Likely a test "
                "bug or a direct _HarnessChatModel(...) construction."
            )

        expected_schema = CALL_SITE_SCHEMAS.get(cast(CallSite, call_site)) if call_site else None
        if expected_schema is not None and (
            getattr(schema, "__qualname__", None) != expected_schema.__qualname__
            or getattr(schema, "__module__", None) != expected_schema.__module__
        ):
            raise LLMHarnessSchemaError(
                f"call_site={call_site!r}: with_structured_output requested "
                f"{getattr(schema, '__name__', repr(schema))} but registered "
                f"schema is {expected_schema.__name__}"
            )

        def _invoke(_messages: Any) -> Any:
            return producer._dispatch(call_site)

        async def _ainvoke(_messages: Any) -> Any:
            return producer._dispatch(call_site)

        return RunnableLambda(_invoke, afunc=_ainvoke)




class LLMMockHarness:

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self._responses: dict[str, BaseModel] = {}
        self._sequences: dict[str, tuple[list[BaseModel | BaseException], int]] = {}
        self._previous_harness: "LLMMockHarness | None" = None
        self._sequence_lock = threading.Lock()

    @classmethod
    def happy_path(cls) -> "LLMMockHarness":
        h = cls()
        from finagents.testing.llm_mock import fixtures

        for site, instance in fixtures.HAPPY_PATH_DEFAULTS.items():
            h.set(site, instance)
        return h

    def __enter__(self) -> "LLMMockHarness":
        self._previous_harness = _factory._ACTIVE_HARNESS
        _factory._set_active_harness(self)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        _factory._set_active_harness(self._previous_harness)
        self._previous_harness = None


    def set(self, call_site: CallSite, instance: BaseModel) -> None:
        if call_site not in CALL_SITE_SCHEMAS:
            raise LLMHarnessMissing(
                f"call_site={call_site!r} is not in the 13-ID CallSite Literal. "
                f"Valid IDs: {sorted(CALL_SITE_SCHEMAS)}"
            )
        expected = CALL_SITE_SCHEMAS[call_site]
        if not isinstance(instance, expected):
            raise LLMHarnessSchemaError(
                f"call_site={call_site!r} expects {expected.__name__}, "
                f"got {type(instance).__name__}"
            )
        origin = getattr(instance, _LLM_MOCK_PERSONA_ORIGIN_ATTR, None)
        if origin is not None and origin != call_site:
            raise LLMHarnessSchemaError(
                f"call_site={call_site!r}: cross-persona registration rejected. "
                f"This canonical instance was tagged origin={origin!r}; "
                f"register it at the {origin!r} slot, not {call_site!r}."
            )
        self._responses[call_site] = instance

    def set_sequence(
        self,
        call_site: CallSite,
        sequence: list[BaseModel | BaseException],
    ) -> None:
        if call_site not in CALL_SITE_SCHEMAS:
            raise LLMHarnessMissing(
                f"call_site={call_site!r} is not in the 13-ID CallSite Literal. "
                f"Valid IDs: {sorted(CALL_SITE_SCHEMAS)}"
            )
        if not sequence:
            raise LLMHarnessExhausted(
                f"call_site={call_site!r}: empty sequence (must contain >= 1 item)"
            )
        expected = CALL_SITE_SCHEMAS[call_site]
        for i, item in enumerate(sequence):
            if isinstance(item, BaseException):
                continue
            if not isinstance(item, expected):
                raise LLMHarnessSchemaError(
                    f"call_site={call_site!r} sequence index {i}: expected "
                    f"{expected.__name__}, got {type(item).__name__}"
                )
            origin = getattr(item, _LLM_MOCK_PERSONA_ORIGIN_ATTR, None)
            if origin is not None and origin != call_site:
                raise LLMHarnessSchemaError(
                    f"call_site={call_site!r} sequence index {i}: cross-persona "
                    f"registration rejected. This canonical instance was tagged "
                    f"origin={origin!r}; register it at the {origin!r} slot, "
                    f"not {call_site!r}."
                )
        self._sequences[call_site] = (list(sequence), 0)


    def _fake_chat_model_for(self, call_site: str | None) -> BaseChatModel:
        model = _HarnessChatModel(call_site=call_site or "")
        object.__setattr__(model, "_producer_harness", self)
        return model

    def _dispatch(self, call_site: str | None) -> BaseModel:
        if not call_site:
            raise LLMHarnessMissing(
                "create_llm_client invoked without call_site= under harness. "
                "Production code must pass call_site=<13-id Literal> per."
            )

        with self._sequence_lock:
            self.calls[call_site] += 1
            if call_site in self._sequences:
                items, idx = self._sequences[call_site]
                if idx >= len(items):
                    raise LLMHarnessExhausted(
                        f"call_site={call_site!r} sequence exhausted at index {idx}"
                    )
                nxt = items[idx]
                self._sequences[call_site] = (items, idx + 1)
            else:
                nxt = None
        if nxt is not None:
            if isinstance(nxt, BaseException):
                raise nxt
            return nxt

        if call_site in self._responses:
            return self._responses[call_site]

        raise LLMHarnessMissing(
            f"call_site={call_site!r} not registered. "
            f"Use harness.set({call_site!r},...) or "
            "LLMMockHarness.happy_path() to pre-register defaults."
        )
