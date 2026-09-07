from __future__ import annotations

import functools
from typing import Any

from langchain_core.language_models import BaseChatModel


def _apply_json_mode_default(llm: BaseChatModel) -> BaseChatModel:
    original_wso = llm.with_structured_output

    @functools.wraps(original_wso)
    def _json_mode_wso(
        schema: Any = None,
        *,
        method: str = "json_mode",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Any:
        return original_wso(
            schema,
            method=method,
            include_raw=include_raw,
            **kwargs,
        )

    object.__setattr__(llm, "with_structured_output", _json_mode_wso)
    return llm


def create_cerebras_client(
    model: str,
    api_key: str | None,
    temperature: float | None = None,
) -> BaseChatModel:
    if not api_key:
        raise ValueError(
            "CEREBRAS_API_KEY is required to use the Cerebras provider. "
            "Set it in your.env file or environment."
        )
    from langchain_cerebras import ChatCerebras

    kwargs: dict[str, object] = {"model": model, "api_key": api_key}
    if temperature is not None:
        kwargs["temperature"] = temperature

    llm = ChatCerebras(**kwargs)  # ty: ignore[invalid-argument-type]
    return _apply_json_mode_default(llm)
