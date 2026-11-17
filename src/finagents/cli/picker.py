from __future__ import annotations

import os
import sys

from rich.console import Console
from rich.prompt import Prompt

from finagents.config import Settings, resolve_model


def list_configured_providers(settings: Settings) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if settings.openrouter_api_key:
        out.append(("openrouter", resolve_model(settings, "openrouter", "deep")))
    if settings.cerebras_api_key:
        out.append(("cerebras", resolve_model(settings, "cerebras", "deep")))
    if settings.groq_api_key:
        out.append(("groq", resolve_model(settings, "groq", "deep")))
    return out


def should_skip_picker(
    settings: Settings,
    *,
    provider_flag: str | None,
) -> bool:
    providers = list_configured_providers(settings)
    if len(providers) <= 1:
        return True
    if provider_flag is not None:
        return True
    if "LLM_PROVIDER" in os.environ:
        return True
    return not sys.stdin.isatty()


def prompt_provider(settings: Settings) -> tuple[str, str]:

    providers = list_configured_providers(settings)
    if should_skip_picker(settings, provider_flag=None):
        return providers[0]
    console = Console()
    choices = [p[0] for p in providers]
    console.print("Configured providers:")
    for name, model in providers:
        console.print(f"  - {name}: {model}")
    chosen = Prompt.ask(
        "Select provider",
        choices=choices,
        default=choices[0],
    )
    for name, model in providers:
        if name == chosen:
            return name, model
    return providers[0]


select_provider = prompt_provider
