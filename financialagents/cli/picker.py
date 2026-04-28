from __future__ import annotations

import os
import sys
from typing import Optional

from financialagents.config import Settings


def list_configured_providers(settings: Settings) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if settings.openrouter_api_key:
        out.append(("openrouter", settings.deep_think_llm))
    if settings.cerebras_api_key:
        out.append(("cerebras", settings.deep_think_llm))
    if settings.groq_api_key:
        out.append(("groq", settings.deep_think_llm))
    out.append(("ollama", settings.deep_think_llm))
    return out


def should_skip_picker(
    settings: Settings,
    *,
    provider_flag: Optional[str],
) -> bool:
    providers = list_configured_providers(settings)
    if len(providers) <= 1:
        return True
    if provider_flag is not None:
        return True
    if "LLM_PROVIDER" in os.environ:
        return True
    if not sys.stdin.isatty():
        return True
    return False


def prompt_provider(settings: Settings) -> tuple[str, str]:
    from rich.console import Console
    from rich.prompt import Prompt

    providers = list_configured_providers(settings)
    if should_skip_picker(settings, provider_flag=None):
        return providers[0]
    console = Console()
    choices = [p[0] for p in providers]
    console.print("Configured providers:")
    for name, model in providers:
        console.print(f"  - {name}: {model}")
    chosen = Prompt.ask(
        "Select provider", choices=choices, default=choices[0],
    )
    for name, model in providers:
        if name == chosen:
            return name, model
    return providers[0]


select_provider = prompt_provider
