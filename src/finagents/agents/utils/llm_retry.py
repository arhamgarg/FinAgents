from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.runnables import Runnable


class StructuredOutputError(RuntimeError):
    def __init__(self, first: Exception, retry: Exception) -> None:
        super().__init__(f"LLM error after retry: {retry!r} (initial: {first!r})")
        self.first = first
        self.retry = retry


async def ainvoke_with_retry[T](
    structured_llm: Runnable[Any, Any],
    prompt: Any,
    parse: Callable[[Any], T],
) -> T:
    try:
        return parse(await structured_llm.ainvoke(prompt))
    except Exception as first:
        try:
            return parse(await structured_llm.ainvoke(prompt))
        except Exception as retry:
            raise StructuredOutputError(first, retry) from retry
