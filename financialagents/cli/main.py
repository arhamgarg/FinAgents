from __future__ import annotations

import asyncio
import logging
from typing import Optional

import typer

app = typer.Typer(
    no_args_is_help=False,
    add_completion=True,
    rich_markup_mode="rich",
)


def _version_callback(value: bool) -> None:
    if value:
        from financialagents import __version__

        typer.echo(f"financialagents {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug logging + tracebacks.",
    ),
) -> None:
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    ctx.obj = {"debug": debug}
    if ctx.invoked_subcommand is None:
        from financialagents.cli.wizard import run_wizard

        run_wizard()


@app.command("run")
def cmd_run(
    ctx: typer.Context,
    ticker: str = typer.Argument(..., help="NSE/BSE ticker symbol."),
    date: Optional[str] = typer.Option(
        None,
        "--date",
        help="YYYY-MM-DD; defaults to IST today.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print PortfolioDecision JSON to stdout.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Override LLM provider (groq|ollama).",
    ),
) -> None:
    from financialagents.cli.run_command import run_pipeline

    debug = bool((ctx.obj or {}).get("debug", False))
    exit_code = asyncio.run(
        run_pipeline(
            ticker=ticker,
            date=date,
            json_output=json_output,
            provider_override=provider,
            debug=debug,
        )
    )
    raise typer.Exit(exit_code)


@app.command("history")
def cmd_history(
    ticker: Optional[str] = typer.Option(None, "--ticker"),
    since: Optional[str] = typer.Option(None, "--since"),
    limit: int = typer.Option(20, "--limit"),
    all_: bool = typer.Option(False, "--all"),
    expand: bool = typer.Option(False, "--expand"),
    prune: bool = typer.Option(False, "--prune"),
    before: Optional[str] = typer.Option(None, "--before"),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Skip the destructive confirmation prompt, T-5-11).",
    ),
) -> None:
    from financialagents.cli.history import run_history

    exit_code = asyncio.run(
        run_history(
            ticker=ticker,
            since=since,
            limit=limit,
            show_all=all_,
            expand=expand,
            prune=prune,
            before=before,
            yes=yes,
        )
    )
    raise typer.Exit(exit_code)


@app.command("show")
def cmd_show(run_id: str = typer.Argument(...)) -> None:
    from financialagents.cli.show import run_show

    exit_code = asyncio.run(run_show(run_id))
    raise typer.Exit(exit_code)


@app.command("dashboard")
def cmd_dashboard(
    port: int = typer.Option(8501, "--port"),
    host: str = typer.Option("localhost", "--host"),
) -> None:
    from financialagents.cli.dashboard_launch import launch_dashboard

    raise typer.Exit(launch_dashboard(port=port, host=host))


@app.command("export")
def cmd_export(
    format_: str = typer.Option("json", "--format"),
    ticker: Optional[str] = typer.Option(None, "--ticker"),
    since: Optional[str] = typer.Option(None, "--since"),
    until: Optional[str] = typer.Option(None, "--until"),
    out: Optional[str] = typer.Option(None, "--out"),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Skip the overwrite-confirmation prompt for --out (T-5-12).",
    ),
) -> None:
    from financialagents.cli.export import run_export

    exit_code = asyncio.run(
        run_export(
            format_=format_,
            ticker=ticker,
            since=since,
            until=until,
            out=out,
            yes=yes,
        )
    )
    raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
