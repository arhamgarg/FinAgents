# FinancialAgents

Multi-agent LLM trading-signal framework for the Indian stock market
(NSE / BSE). Orchestrates a stateful pipeline of analysts, researchers,
a trader, three risk personas, and a portfolio manager to produce
well-reasoned trading signals via LangGraph.

The pipeline runs locally against your Ollama models by default (zero
LLM cost) or against Groq when you set `GROQ_API_KEY`. Market data flows
from Angel One SmartAPI; news and fundamentals are scraped from
Screener.in / NSE / MoneyControl.

## Quickstart

Prerequisites: Python 3.12 or 3.13 and [`uv`](https://docs.astral.sh/uv/)
installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
git clone <repo>
cd financial-agents
uv sync --all-extras
cp .env.example .env       # add ANGELONE_* and (optionally) GROQ_API_KEY
uv run fa run RELIANCE     # run the full pipeline; default provider = ollama
uv run fa dashboard        # browser-based dashboard at http://localhost:8501
```

`uv run fa --help` lists every subcommand. `uv run fa` (no arguments)
launches an interactive wizard that walks you through provider, model,
and ticker selection.

## CLI

Five subcommands under `fa` (also installed as `financialagents`):

| Command | Purpose |
|---------|---------|
| `fa run <ticker>` | Run the full pipeline. Flags: `--date YYYY-MM-DD`, `--provider {ollama,groq}`, `--json`, `--debug`. Defaults: today in IST, `LLM_PROVIDER` env or `ollama`. |
| `fa history` | Browse the signal log. Flags: `--ticker`, `--since`, `--limit N`, `--all`, `--expand`, `--prune --before <date>`, `--yes`. |
| `fa show <run-id>` | Render the full Signal panel for a recorded run. |
| `fa dashboard` | Launch the Streamlit dashboard. Flags: `--port`, `--host`. |
| `fa export` | Export the signal log. Flags: `--format {json,csv}`, `--ticker`, `--since`, `--until`, `--out FILE`, `--yes`. |

Bare `fa` (no arguments) launches the interactive wizard.

Shell completion: `fa --install-completion` (Typer-provided; prints the
rc-file installation hint for your shell).

Verbosity: `--debug` enables Python `logging.DEBUG` and full tracebacks.
There is no `-v` / `-vv`.

Exit codes: `0` on any decision (BUY / SELL / HOLD / veto). Non-zero only
on errors (network failure, invalid ticker, bad arguments). Cancel
mid-run with Ctrl+C — the partial state is persisted to the signal log
under status `cancelled` and can be inspected via `fa show <run-id>`.

## Python API

```python
import asyncio
from financialagents import FinancialAgentsGraph

async def main():
    graph = FinancialAgentsGraph(full_pipeline=True)
    state = await graph.apropagate("RELIANCE", "2026-04-26")
    decision = state.final_trade_decision  # PortfolioDecision
    print(decision.signal.action, decision.approved, decision.pm_notes)

asyncio.run(main())
```

`FinancialAgentsGraph` is the only public symbol. `__version__` is sourced
from package metadata (`importlib.metadata.version("financialagents")`),
so `pyproject.toml`'s `[project] version` field is the single source of
truth.

## Signal log location

The CLI and dashboard share a single SQLite database at the platform's
user-data directory:

| OS | Default location |
|----|------------------|
| macOS | `~/Library/Application Support/financialagents/signals.db` |
| Linux | `~/.local/share/financialagents/signals.db` |
| Windows | `%LOCALAPPDATA%\financialagents\signals.db` |

Override the location with the `FA_DATA_DIR` environment variable:

```bash
FA_DATA_DIR=/path/to/dir uv run fa run RELIANCE
```

Schema migrations run automatically on first open. The migration runner
snapshots the database to `signals-<UTC-timestamp>.db` next to the
original before applying any pending migration. There is no auto-prune;
use `fa history --prune --before <date>` to clean up.

## Dashboard

`uv run fa dashboard --port 8501` launches the Streamlit web UI in a
subprocess. Three pages:

1. **Run** — sidebar form (ticker, date, provider, model, rounds) +
   live pipeline progress + verdict-first Signal Card.
2. **History** — sortable `st.dataframe` of past signals; click a row
   to open the Signal page.
3. **Signal** — `/Signal?id=<run_id>` permalink-addressable detail view
   with Bull / Bear debate columns and risk-persona side-by-side
   assessments.

Closing the terminal (or Ctrl+C in it) cleanly terminates the dashboard
subprocess.

## Development

```bash
uv run pytest -q                # full test suite
uv run ruff check .             # lint
uv run ty check financialagents # typecheck
```

the rationale behind every choice.

## License

TBD.
