# FinAgents

Multi-agent trading-signal research for Indian equities (NSE/BSE), built with
LangGraph. Three analysts, investment and risk debates, a trader, and a
portfolio manager produce explained decisions, not broker orders.

## Quickstart

```bash
git clone https://github.com/arhamgarg/FinAgents
cd FinAgents
uv sync
# Set a provider key in your shell or a project-local .env first.
uv run fa run RELIANCE
uv run fa dashboard
```

Requires Python `>=3.12,<3.14` and a key for the selected provider:
`OPENROUTER_API_KEY` (default), `CEREBRAS_API_KEY`, or `GROQ_API_KEY`.
Select with `LLM_PROVIDER` or `fa run --provider`. Live Angel One data also
requires its API key, client ID, PIN, and TOTP secret; absent credentials
produce degraded data, not fabricated prices.

`fa --help` lists the run, history, show, export, and dashboard commands.
Bare `fa` starts the interactive wizard; `finagents` is the equivalent long name.

## Project guide

See [PROJECT.md](PROJECT.md) for installation, configuration defaults, the
Python API, pipeline architecture, safeguards, storage, failure handling,
development, and acceptance reports.

## License

[GNU AGPL-3.0](LICENSE).
