from __future__ import annotations

from openai import LengthFinishReasonError as _OpenAILengthFinishReasonError
from openai.types.chat import ChatCompletion as _ChatCompletion
from pydantic import BaseModel, ValidationError




def length_error() -> _OpenAILengthFinishReasonError:
    fake = _ChatCompletion.model_construct(
        id="harness",
        choices=[],
        created=0,
        model="harness-fake",
        object="chat.completion",
    )
    return _OpenAILengthFinishReasonError(completion=fake)


length_error.SENTINEL = "LengthFinishReasonError"  # ty: ignore[unresolved-attribute]




class _SentinelModel(BaseModel):

    x: int


def _make_validation_error() -> ValidationError:
    try:
        _SentinelModel(x="harness-induced-failure")  # ty: ignore[invalid-argument-type]
    except ValidationError as err:
        return err
    raise RuntimeError("unreachable: _SentinelModel(x=str) must raise ValidationError")


def validation_error() -> ValidationError:
    return _make_validation_error()


validation_error.SENTINEL = "ValidationError"  # ty: ignore[unresolved-attribute]




def runtime_error_after_retry() -> RuntimeError:
    return RuntimeError("simulated LLM failure after retry (harness)")


runtime_error_after_retry.SENTINEL = "LLM error after retry"  # ty: ignore[unresolved-attribute]




def retry_then_succeed(
    value: BaseModel,
    first_error: str = "length",
) -> list[BaseModel | BaseException]:
    if first_error == "length":
        return [length_error(), value]
    elif first_error == "validation":
        return [validation_error(), value]
    raise ValueError(f"first_error must be 'length' or 'validation', got {first_error!r}")
