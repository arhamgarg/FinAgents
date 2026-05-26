from __future__ import annotations

import re

_SECRET_PATTERN = re.compile(
    r"((?:OPENROUTER_API_KEY|CEREBRAS_API_KEY|GROQ_API_KEY|"
    r"ANGELONE_[A-Z_]+|REDDIT_[A-Z_]+|(?<![A-Z_])api_key))"
    r"[=:]\s*['\"]?\S+['\"]?",
    re.IGNORECASE,
)


def redact_secrets(text: str) -> str:
    return _SECRET_PATTERN.sub(lambda m: f"{m.group(1)}=***", text)


__all__ = ["_SECRET_PATTERN", "redact_secrets"]
