from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class NewsArticle(BaseModel):
    title: str
    url: str
    published_at: Optional[datetime] = None
    source: str
    snippet: Optional[str] = None
