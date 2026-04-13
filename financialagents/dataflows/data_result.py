from __future__ import annotations

from datetime import datetime, timezone
from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

DataStatus = Literal["ok", "partial", "empty", "error"]


class DataResult(BaseModel, Generic[T]):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: DataStatus
    data: Optional[T] = None
    source: str = ""
    error: Optional[str] = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stale: bool = False
