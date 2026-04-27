from __future__ import annotations


def run_wizard() -> None:
    import asyncio

    from rich.console import Console
    from rich.prompt import Confirm, Prompt

    from financialagents.cli.picker import prompt_provider
    from financialagents.cli.run_command import run_pipeline
    from financialagents.config import Settings

    console = Console()
    console.print("[bold]FinancialAgents[/bold] interactive wizard")
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
