from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class NewsArticle(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    url: str
    published_at: datetime | None = None
    source: str
    snippet: str | None = None
