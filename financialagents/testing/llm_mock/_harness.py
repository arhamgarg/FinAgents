from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from financialagents.llm_clients import factory as _factory
from financialagents.testing.llm_mock._call_sites import CALL_SITE_SCHEMAS, CallSite



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
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=""))]
        )

    def with_structured_output(  # type: ignore[override]
        self,
        schema: Any,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        call_site = self.call_site

        def _invoke(_messages: Any) -> Any:
            active = _factory._ACTIVE_HARNESS
            if active is None:
                raise LLMHarnessMissing(
                    "harness model invoked without an active harness "
                    f"(call_site={call_site!r})"
                )
            return active._dispatch(call_site)

        async def _ainvoke(_messages: Any) -> Any:
            active = _factory._ACTIVE_HARNESS
            if active is None:
                raise LLMHarnessMissing(
                    "harness model invoked without an active harness "
                    f"(call_site={call_site!r})"
                )
            return active._dispatch(call_site)

        return RunnableLambda(_invoke, afunc=_ainvoke)




class LLMMockHarness:

    def __init__(self) -> None:
        self._responses: dict[str, BaseModel] = {}
        self._sequences: dict[str, tuple[list[BaseModel | BaseException], int]] = {}

    @classmethod
    def happy_path(cls) -> "LLMMockHarness":
        h = cls()
        from financialagents.testing.llm_mock import fixtures

        for site, instance in fixtures.HAPPY_PATH_DEFAULTS.items():
            h.set(site, instance)  # ty: ignore[invalid-argument-type]
        return h

    def __enter__(self) -> "LLMMockHarness":
        _factory._set_active_harness(self)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        _factory._set_active_harness(None)


    def set(self, call_site: CallSite, instance: BaseModel) -> None:
        expected = CALL_SITE_SCHEMAS[call_site]
        if not isinstance(instance, expected):
            raise LLMHarnessSchemaError(
                f"call_site={call_site!r} expects {expected.__name__}, "
                f"got {type(instance).__name__}"
            )
        self._responses[call_site] = instance

    def set_sequence(
        self,
        call_site: CallSite,
        sequence: list[BaseModel | BaseException],
    ) -> None:
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
        self._sequences[call_site] = (list(sequence), 0)


    def _fake_chat_model_for(self, call_site: str | None) -> BaseChatModel:
        return _HarnessChatModel(call_site=call_site or "")

    def _dispatch(self, call_site: str | None) -> BaseModel:
        if not call_site:
            raise LLMHarnessMissing(
                "create_llm_client invoked without call_site= under harness. "
                "Production code must pass call_site=<13-id Literal> per."
            )

        if call_site in self._sequences:
            items, idx = self._sequences[call_site]
            if idx >= len(items):
                raise LLMHarnessExhausted(
                    f"call_site={call_site!r} sequence exhausted at index {idx}"
                )
            nxt = items[idx]
            self._sequences[call_site] = (items, idx + 1)
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
