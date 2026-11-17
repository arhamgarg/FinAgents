# FinAgents project handbook

FinAgents produces explained trading signals for Indian equities (NSE/BSE) using
a LangGraph pipeline. It is a research and decision-support application, not an
order-execution system. The repository is `arhamgarg/FinAgents`, the distribution
and Python package are `finagents`, and the command names are `fa` and `finagents`.

This handbook describes the code as it is. Where it disagrees with the code or
with a reproducible test, the code and the test decide.

## Contents

- [Purpose and scope](#purpose-and-scope)
- [Installation and configuration](#installation-and-configuration)
- [Interfaces and persistence](#interfaces-and-persistence)
- [Pipeline architecture](#pipeline-architecture)
- [Failure handling](#failure-handling)
- [Development](#development)
- [Acceptance reports](#acceptance-reports)

## Purpose and scope

FinAgents adapts the TradingAgents multi-agent approach to Indian market data,
local constraints, and a typed stateful pipeline. Separate analysts feed a
research manager. Adversarial investment and risk discussions lead to a trader
proposal and a portfolio-manager decision. The debate exposes competing reasons.
It does not substitute LLM confidence for market evidence.

### Delivery-only scope

The only time horizon is `positional` (delivery-only). The T2T flag is
contextual data, not a reason to veto a delivery trade automatically. `SELL` is
an output action, not implemented short-selling or an order sent to a broker.
There is no automated execution, derivatives support, backtester, realized-P&L
tracker, multi-user authentication, or live tick runtime.

Sentiment collection and sector rotation are not part of the pipeline, and
there is no reflection or agent memory. SQLite signal history is an audit log,
not agent memory or portfolio accounting.

## Installation and configuration

Use the checked-in `uv.lock` and Python `>=3.12,<3.14`:

```bash
git clone https://github.com/arhamgarg/FinAgents
cd FinAgents
uv sync
# Configure a provider key in the shell or project-local .env first.
uv run fa run RELIANCE
uv run fa dashboard
```

`Settings` in [config.py](src/finagents/config.py) reads case-insensitive environment
variables and a UTF-8 `.env`. Unknown `.env` fields are ignored. Constructing
`Settings` requires at least one LLM key. A key for a different provider does
not make the selected provider usable: if the active provider's key is absent,
FinAgents warns and fails when it calls the provider. Do not commit `.env`,
credentials, session tokens, or TOTP secrets.

| Provider | Key | Deep-model default | Quick-model default |
|---|---|---|---|
| OpenRouter (default) | `OPENROUTER_API_KEY` | `openrouter/auto` | `openrouter/auto` |
| Cerebras | `CEREBRAS_API_KEY` | `gpt-oss-120b` | `llama3.1-8b` |
| Groq | `GROQ_API_KEY` | `openai/gpt-oss-120b` | `openai/gpt-oss-20b` |

These are repository defaults, not a claim that each hosted model is available.
`LLM_PROVIDER` selects the provider. The optional `DEEP_THINK_LLM` and
`QUICK_THINK_LLM` overrides apply to every provider. Consumers call
`resolve_model(settings, provider, tier)` instead of reading the optional fields
as resolved model names. The picker, CLI, dashboard, LLM factory, and
acceptance-report metadata share this resolver. There is no local-model or
no-key fallback.

Angel One uses `ANGELONE_API_KEY`, `ANGELONE_CLIENT_ID`, `ANGELONE_PIN`, and
`ANGELONE_TOTP_SECRET`. Missing credentials degrade data availability. They do
not supply fabricated prices. The primary vendors are Angel One for OHLCV,
Screener for fundamentals, and Moneycontrol for news. Each client owns its
fallback chain. The vendor map is not a generic plugin registry.

| Setting | Default and meaning |
|---|---|
| `MAX_DEBATE_ROUNDS` | 3 investment rounds |
| `MAX_RISK_DISCUSS_ROUNDS` | 2 risk rounds |
| `ANALYST_WEIGHTS` | technical 0.303, fundamentals 0.455, news 0.242 |
| `RISK_PERSONA_WEIGHTS` | aggressive 0.25, neutral 0.50, conservative 0.25 |
| `POSITION_SIZING_METHOD` | `atr_based`; alternatives `kelly`, `fixed_fractional` |
| `RISK_PER_TRADE_PCT` | 0.02 |
| `KELLY_FRACTION` | 0.25 |
| `ATR_STOP_MULTIPLIER` | 2.0 |
| `PORTFOLIO_VALUE_INR` | 100000.0, sizing capital, not a synchronized account balance |
| `DEFAULT_TIME_HORIZON` | `positional` only |
| `FA_DATA_DIR` | optional signal-log directory override |

Dictionary-valued environment settings use JSON. Construct validated `Settings`
at boundaries. Pydantic `model_copy(update=...)` does not validate updates.

### Technology choices

`uv.lock` holds the exact installed dependency graph. `pyproject.toml` holds the
declared dependencies, build backend, and entry points. Do not keep a second
pin list by hand. The declared stack includes LangGraph 1.1.6, LangChain 1.2.15,
Pydantic Settings 2.13.1, SmartAPI 1.5.5, pandas 3.0.2, pandas-ta 0.4.71b0,
httpx 0.28.1, aiosqlite 0.22.1, Typer 0.24.1, Rich 14.3.3, and Streamlit 1.56.0.
Dedicated LangChain adapters serve the three LLM providers.

LangGraph supplies explicit topology, reducers, and nested debates. Pydantic
validates boundaries. httpx and aiosqlite give asynchronous I/O. SQLite keeps
local persistence simple. Typer with Rich and Streamlit give terminal and
browser views over the same events and stored state. Hatchling builds the `src`
layout package. `pyproject.toml` declares `logzero` because SmartAPI imports it without
listing it in its wheel metadata. The Python upper bound comes from the
pandas-ta and Numba compatibility chain, so do not widen it on syntax grounds
alone. Ruff and ty target Python 3.12, the package minimum. CI runs on Python
3.12 and 3.13.

## Interfaces and persistence

### Command line and dashboard

Bare `fa` starts the interactive provider, model, and ticker wizard. `fa --help`
lists the five commands. Both executable names invoke the same Typer app. The
global options `--debug` and `--version` go before the command, as in
`fa --debug run RELIANCE`.

| Command | Purpose and important options |
|---|---|
| `fa run TICKER` | Full pipeline; `--date`, `--provider`, `--json` |
| `fa history` | `--ticker`, `--since`, `--limit`, `--all`, `--expand`; explicit `--prune --before DATE` with confirmation or `--yes` |
| `fa show RUN_ID` | Stored decision or partial-state diagnostic |
| `fa export` | `--format json\|csv`, `--ticker`, `--since`, `--until`, `--out`, `--yes` |
| `fa dashboard` | Streamlit subprocess; `--port`, `--host` |

`--debug` enables debug logging and full tracebacks. There are no `-v` or `-vv`
levels. It also writes `traces/RUN_ID.jsonl` beside the signals database with
the package version, provider and model names, debate rounds, and LLM span
status and timing. Trace records omit prompts, responses, credentials, and
exception text. Span IDs are local callback identifiers, not provider request
IDs. Traces are diagnostic evidence, not replay artifacts. Debug logs are
separate from traces, so review them before sharing.

Typer provides `--install-completion`. TTY runs show Rich live progress, and
non-TTY runs write log lines. `--json` prints machine-readable decision JSON on
stdout and progress on stderr. BUY, SELL, HOLD, and veto are decisions, not
command failures. Cancellation exits with code 130, and errors exit nonzero. A
zero exit status does not mean a profitable or executable trade.

The dashboard has Run, History, and Signal pages. The Run form selects ticker,
date, provider, model, and rounds. Progress ends in a verdict-first card with
entry, stop, target, size, and PM notes. History selects a persisted row, and
Signal supports `/Signal?id=<8-hex-run-id>`. Debate rounds, analyst reports, and
three risk-persona views give drill-down context. The dashboard validates URL
run IDs before it reads storage. The launcher passes an argv list to a
subprocess, not a shell command. UI rendering must not enable unsafe HTML for
LLM content.

### Interface design

Both interfaces show the verdict first: action and approval state, four metrics
(Entry, Stop, Target, Size), PM rationale, then the Bull and Bear debate, risk
personas, and analyst-report detail. Status stays distinct from action, so HOLD
alone does not mean a PM veto. Missing metrics render explicitly, not as zero.
Use native Streamlit and Rich primitives rather than a web-component or chart
framework. Streamlit theme settings live in `.streamlit/config.toml`. Terminal
formatting uses semantic named colors, not pixel sizes.

The palette is background `#0F1419`, secondary `#1B2129`, text `#E6EDF3`,
border `#30363D`, green `#2EA043`, red `#F85149`, and amber `#D29922`.
Green means BUY, approved, or success. Red means SELL, veto, or error. Amber
means HOLD, modified, or cancelled. Labels and glyphs carry the meaning without
relying on color. Persona divergence displays green at 5 percentage points or
less, amber above 5 and up to 10, and red above 10. This display scale is
separate from the 30-point acceptance criterion.

For new layout work, use the spacing scale 4, 8, 16, 24, 32, 48, and 64, and
the type sizes body 16, label 14, heading 20, and display 28 at weights 400 and
600. Streamlit does not expose every typography setting, so do not inject unsafe
HTML to approximate one. Keep numeric alignment, rupee and thousand separators,
percentage precision, and IST-aware dates.

Copy is calm and specific. "Run pipeline", "Ready when you are", and the
RELIANCE, TCS, and INFY quick picks define the primary interaction. Empty
history explains how to create a signal. A missing run ID directs the user to
history. Errors distinguish failure from cancellation, and destructive prompts
default to No. Tests and the renderers define the exact copy.

### Python API

```python
import asyncio
from finagents import FinAgentsGraph

async def main():
    graph = FinAgentsGraph(full_pipeline=True)
    state = await graph.apropagate("RELIANCE", "2026-04-26")
    print(state.final_trade_decision)

asyncio.run(main())
```

`FinAgentsGraph()` defaults to the analysis-only graph, so opt into the full
pipeline explicitly. `propagate()` wraps `apropagate()` with `asyncio.run`.
Inside a running event loop, use `await apropagate(...)`. `compiled()` exposes
the underlying graph to event consumers. Each public invocation gets a unique
thread ID, and `AgentState.model_validate` rehydrates the result. The package
export is lazy, so importing `finagents` does not initialize network clients.
`__version__` comes from distribution metadata.

### Data and storage

Data clients return explicit success, empty, or error outcomes through
`DataResult`. Rate limiting, retry policies, session handling, and fallback
vendors live in `src/finagents/dataflows/`. Session creation imports SmartConnect
lazily, so importing the package does no networking. The market-data cache at
`.cache/finagents.db` is disposable. Its TTLs are 86400 seconds for the scrip
master, OHLCV, fundamentals, and surveillance, and 1800 seconds for news. This
cache is separate from durable signal history.

The signal log defaults to the platform user-data directory under `finagents`:
macOS `~/Library/Application Support/finagents/signals.db`, Linux
`~/.local/share/finagents/signals.db`, and Windows
`%LOCALAPPDATA%\finagents\signals.db`. `FA_DATA_DIR` overrides the directory.
If the default `finagents` directory does not exist and a legacy
`financialagents` directory does, first use moves the legacy directory into
place with its history and snapshots. A custom `FA_DATA_DIR` is never moved. The
market cache is not migrated.

SQLite uses WAL, NORMAL synchronization, and a 5000 ms busy timeout. A run is
inserted as `running` and updated to `success`, `cancelled`, or `errored`. The
log stores indexed summary fields and the JSON state payload. Exports omit the
payload and use a stable 13-column summary. Reads handle an absent database
without creating history. SQL is parameterized. Before applying pending schema
migrations, FinAgents writes a backup named `signals-<UTC-timestamp>.db`. There
is no automatic history pruning. Keep backups, and review explicit prune or
overwrite requests before acting on them.

The concentrated example portfolio is [tests/fixtures/portfolio.json](tests/fixtures/portfolio.json):
RELIANCE 150, TCS 50, HDFCBANK 80, INFY 60; sector exposures 0.22, 0.35, 0.28,
and 0.15 for Energy, IT, Banking, and Other; total value ₹1,500,000. It is a
test fixture, not live holdings. `PortfolioContext.from_seed_file(path)` loads
it explicitly. The public graph does not read it automatically.

## Pipeline architecture

```text
START → technical ───┐
      → fundamentals ├→ validation → research manager
      → news ────────┘      ├→ END (analysis-only)
                            └→ investment debate → thesis → trader
                               → risk debate → portfolio manager → END
```

The graph edges live in `src/finagents/graph/setup.py` and `src/finagents/graph/subgraphs/`.
The investment debate subgraph produces the investment thesis. Parent wrappers
convert between `AgentState` and the smaller subgraph states, and they must keep
the parent's analyst and market-data fields. The analysis-only and full graphs
share one three-analyst fan-out.

| LLM tier | Call sites |
|---|---|
| Quick | technical, fundamentals, news; aggressive, neutral, conservative risk personas |
| Deep | research manager, bull researcher, bear researcher, investment thesis, trader, risk consolidation, portfolio manager |
| No LLM | validation, debate initialization, routing, reducers, sizing, deterministic validators |

The research manager applies analyst weights once. The trader reads the
resulting brief and thesis plus four trusted technical fields: `current_price`,
`atr_14`, `circuit_band_pct`, and `t2t_flag`. Sending the analyst rationales
again would count their influence twice.

Analyst confidence measures data completeness, not an LLM self-rating or a
calibrated probability. Technical confidence is the number of populated fields
out of nine: current price, RSI, MACD line, MACD histogram, Bollinger upper,
Bollinger lower, SMA50, SMA200, and ATR14. EMA20, prose, and guard fields do not
count. Fundamentals confidence is the number of populated fields out of five:
P/E, P/B, ROCE, debt/equity, and promoter holding. Narrative fields do not
count. News confidence is 0 for no articles, 0.3 for 1 or 2, 0.6 for 3 to 5,
and 1.0 for 6 or more, counted from the fetched results.

### Debate state and retries

`InvestDebateState` and `RiskDebateState` keep history in integer-keyed
dictionaries merged with `merge_by_key`, where the last write for a round wins.
A retry therefore does not duplicate entries. Ordered accessors (`bull_turns`,
`bear_turns`, `risk_turns`) hide the storage details. The bear node completes an
investment round and increments its counter. Risk consolidation does the same
for a risk round. Routers stop at the configured round counts, with recursion
limits as a backstop. A new invocation must not reuse an earlier ticker and
date's checkpointer identity.

Risk personas run in sequence: aggressive, neutral, conservative, then
consolidation. Each emits a `PersonaAssessment` with argument, counterargument,
recommended size, and confidence. Later personas see earlier assessments, and
later rounds start from the previous consolidation. Weighted size synthesis is
the numeric consumer of the persona weights. Different prompts and temperatures
encourage disagreement. The 30-percentage-point divergence criterion is a
manual acceptance check, not a runtime clamp.

### Trader and portfolio-manager safeguards

Price grounding runs before position sizing, so sizing uses the trusted
`current_price`, never an LLM-invented entry. ATR sizing floors
`(capital × risk) / (ATR × multiplier)` to whole shares. Without ATR, sizing
falls back to fixed-fractional sizing on the entry-to-stop distance. Fractional
Kelly uses `(b × p − (1 − p)) / b`, scaled by the configured fraction, and a
nonpositive edge gives zero. A 5% position cap applies whatever the sizing
method, and shares scale with it. These are heuristics, not calibrated return or
risk guarantees.

The validators run in order: price grounding, circuit-band check, cost floor,
size cap. All of them run, so an earlier HOLD does not hide later guard notes.
The distance to the circuit limit must not exceed 80% of the effective band. A
missing band falls back to 20% and adds an explanatory note. The target must sit
at least 0.003 (0.3%) above entry. **These are code constants, not a verified
statement of present exchange bands, broker fees, or tax rules.** The fallback
for missing data can be permissive, so do not treat FinAgents as an
execution-grade exchange-rule validator.

Only the portfolio manager writes `final_trade_decision: PortfolioDecision`.
`approve`, `modify`, and `veto` are explicit decisions. `modify` may change only
position-size fields, not the trader's prices, targets, or direction, and the
validators run again afterwards. A veto embeds HOLD with zero size and keeps the
rejected thesis and price context. PM reasoning must address the portfolio
context instead of approving the proposal unchanged. HOLD and missing numeric
fields need explicit UI handling, so never multiply or format `None` as a
number.

## Failure handling

Structured-output calls retry and then fall back to a safe typed result. When
an analyst exhausts its retries, it emits an empty report with zero confidence
instead of raising. A raised exception inside the parallel analyst step would
discard the whole fan-in and leave an empty state that looks successful.
Research and debate failures keep their diagnostic text. A trader or PM failure
falls back to HOLD or veto. Failures stay visible instead of looking like
successful signals, and a fallback is not evidence of correct reasoning.

### Streaming and cancellation

`stream_pipeline_events` adapts LangGraph `updates` and `values` events for both
interfaces. With subgraphs enabled, each event is a three-tuple of namespace,
mode, and payload. Do not unpack it as a two-tuple. A child `values` snapshot is
not a complete parent `AgentState`, so it must not overwrite the rolling root
snapshot. `AgentState` forbids unknown fields, so a wrong-schema snapshot fails
visibly. Consumers persist terminal errors and partial state, never a success
with a missing final decision.

In the CLI, the first Ctrl+C requests graceful cancellation and keeps partial
state. A second Ctrl+C stops hard. The dashboard runs work in a background
thread with a queue and a cooperative cancellation event. A new run cancels
stale worker handles before it resets session state and starts a new producer.
Closing a browser tab does not run the Streamlit page synchronously, so
cancellation after a tab close is best-effort. It happens at the next kickoff or
at a request boundary or timeout, not instantly. Error logs and persisted error
messages redact provider and session secrets.

## Development

Use the existing uv environment and tools. Do not add a second packaging
framework.

```bash
uv run --locked gate
# Focused feedback while editing:
uv run pytest tests/integration -q --tb=short
```

`uv run --locked gate` (`scripts/gate.py`) runs the same fail-fast sequence locally and in CI: locked
`uv sync`, `ruff check`, `ruff format --check`, `ty check` with
`--error-on-warning`, `lint-imports` for the package layers, the
`scripts/check_*.py` checks (whitespace, file size, comments, naming, secrets),
and the full offline `pytest` suite with coverage. The secrets check needs
gitleaks 8.30.1 or later on `PATH` and an `origin/main` ref to diff against. It
also installs a pre-commit hook that scans staged changes, keeping any existing
hook as `.git/hooks/pre-commit.bak`.

Tests live under `tests/`. Use async tests and fixture-backed httpx and LLM
calls instead of live providers. Keep network initialization out of imports.
Maintain typed schemas, partial-state update contracts, reducer idempotency,
deterministic guardrails, and the stdout and stderr split. Add a focused
regression test for changed behavior. Format the files you touch. If a
whole-repository formatting pass is requested, put its churn in a separate
style commit. Use Conventional Commits and descriptive branch names such as
`docs/unify-project-documentation`.

Report the exact checks you ran and the failures you saw. A diagnostic is not
proven pre-existing until it is compared against an unchanged baseline.

### Offline test harness

`src/finagents/testing/` provides `LLMMockHarness`, response factories, failure
sentinels, sequences, and named scenarios. Patch at the LLM factory boundary so
graph and node tests run the real routing, validation, fallback, and retry code
without spending tokens. Registrations are specific to a call site and checked
against its schema. The risk personas share an output schema but keep their
origin identity, so registering the wrong persona fails. Sequence consumption
is locked, and an exhausted sequence fails instead of cycling its fixtures.

The root fixture disables dotenv loading, supplies fake provider credentials,
isolates the signal and market databases, and rejects network connections and
DNS lookups, including attempts that application fallbacks would swallow.
Integration tests use populated market, fundamentals, news, and surveillance
fixtures, and harness call counters prove that all 13 LLM call sites run. CLI
and Streamlit AppTest round trips exercise the real graph, persistence, and
result views. Cold-import checks run in subprocesses. Graph-wiring tests patch
lookup sites without reloading shared modules. Offline success does not replace
live-provider or human acceptance checks.

A nested harness context restores the outer context on exit. A chat model and
its structured-output runnable stay bound to the harness that produced the
model, even when used in another context. The harness does not fake direct
unstructured generation or `include_raw=True` with empty messages. Tests guard
against accidental live-provider creation. Canonical fixtures are isolated from
mutation. Do not treat model-copy or reload identity limits as production schema
semantics.

## Acceptance reports

The acceptance scripts write timestamped Markdown blocks under the gitignored
`reports/` directory and create the directory and file on first write. Pass
`--output` to keep separate runs elsewhere. Run one writer per output file,
because the scripts do not lock concurrent appends. Review reports for secrets
before sharing, and do not commit generated reports.

| Script under `scripts/` | Default report file | What it measures |
|---|---|---|
| `uat_trader_hold_rate.py` | `trader-hold-rate.md` | Directional cohort HOLD rate by data richness |
| `uat_risk_debate.py` | `risk-debate.md` | Persona size spread per ticker |
| `uat_pm_gate.py` | `pm-gate.md` | Modification rate and substantive approval notes |
| `uat_reliance.py` | `reliance.md` | RELIANCE end-to-end decision and guard evidence |

```bash
# Offline argument and format smoke checks, not live acceptance evidence.
uv run python scripts/uat_trader_hold_rate.py --dry-run
uv run python scripts/uat_risk_debate.py --dry-run
uv run python scripts/uat_pm_gate.py --dry-run
uv run python scripts/uat_reliance.py --dry-run
```

Dry runs make no LLM calls and write no report. Without `--dry-run`, the scripts
can make network requests and incur provider charges. Run them from the
repository root. Every script accepts `--date`. The risk-debate and PM-gate
scripts read a checked-in ticker batch file in `scripts/` (RELIANCE, TCS,
HDFCBANK, INFY, SUZLON), resolved relative to the script, and accept
`--ticker-file` or `--tickers` instead. The trader script selects ten tickers
from surveillance data by default and accepts `--tickers`. With a real
acceptance run, record the ticker list, date, provider and resolved models,
settings, input source, sample size, failures, and observations.

The HOLD-rate criterion uses directional theses and confidence thresholds. A
ticker is data-rich when at least two of three analyst confidences are nonzero
and overall confidence is at least 0.6. The data-poor threshold is 0.4. Each
populated cohort must stay below 50% HOLD. An empty cohort is N/A, and two
empty cohorts do not pass. The thesis direction scan treats BUY, accumulate,
and long as one direction and SELL, reduce, trim, exit, and short as the other.
Mixed text is not directional. This scan can misclassify nuanced prose. It is an
acceptance measure, not production trade parsing.

Risk divergence requires `max(size) − min(size) ≥ 0.30` for each ticker. The PM
gate requires either a modification rate of at least 0.20 or a nonempty set of
approvals that all have substantive notes. A substantive note has at least 40
characters and uses portfolio vocabulary: concentration, sector, cash headroom,
overweight, underweight, or portfolio value. Error and missing rows stay in the
report as evidence. Passing this heuristic does not establish the quality of
every decision. The RELIANCE script writes round counts, circuit and cost
checks, PM notes, and the full decision JSON.
