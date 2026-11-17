from __future__ import annotations

import asyncio

from rich.console import Console
from rich.prompt import Confirm, Prompt

from finagents.cli.picker import prompt_provider
from finagents.cli.run_command import run_pipeline
from finagents.config import Settings


def run_wizard() -> None:

    console = Console()
    console.print("[bold]FinAgents[/bold] interactive wizard")
    ticker = Prompt.ask("Ticker", default="RELIANCE")
    settings = Settings()
    provider, model = prompt_provider(settings)
    if not Confirm.ask(
        f"Run pipeline on {ticker} via {provider}:{model}?",
        default=True,
    ):
        console.print("[dim]Cancelled by user.[/dim]")
        return
    asyncio.run(
        run_pipeline(
            ticker=ticker,
            date=None,
            json_output=False,
            provider_override=provider,
            debug=False,
        )
    )
