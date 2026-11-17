from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from finagents.clock import utc_now

DataStatus = Literal["ok", "partial", "empty", "error"]


class DataResult[T](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    status: DataStatus
    data: T | None = None
    source: str = ""
    error: str | None = None
    fetched_at: datetime = Field(default_factory=utc_now)
    stale: bool = False
