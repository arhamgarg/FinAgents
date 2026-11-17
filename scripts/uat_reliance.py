#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from finagents.clock import local_now

if TYPE_CHECKING:
    from finagents.agents.trader.schemas import TradingSignal
    from finagents.agents.utils.agent_states import AgentState
    from finagents.config import Settings

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_DEFAULT_OUTPUT = Path("reports/uat/reliance.md")
_SECTION_MARKER = "## RELIANCE E2E"
_TICKER = "RELIANCE"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual RELIANCE end-to-end acceptance run for the full pipeline"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Path to the Markdown acceptance report (default: %(default)s)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=local_now().strftime("%Y-%m-%d"),
        help="Date passed to FinAgentsGraph.apropagate (default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Smoke test: verify args + output path + marker without invoking the LLM pipeline",
    )
    return parser.parse_args()


class CheckResult(NamedTuple):
    passed: bool
    detail: str


def _circuit_band_check(signal: TradingSignal, circuit_band_pct: float | None) -> CheckResult:
    entry = getattr(signal, "entry_price", None)
    stop = getattr(signal, "stop_loss", None)
    if entry is None or stop is None:
        return CheckResult(False, "FAIL — entry_price or stop_loss is None")
    if circuit_band_pct is None:
        return CheckResult(False, "FAIL — circuit_band_pct unavailable (NSE sec_list miss)")
    if entry <= 0:
        return CheckResult(False, f"FAIL — non-positive entry_price={entry}")
    ratio = abs(entry - stop) / entry
    threshold = circuit_band_pct * 0.8
    verdict = "PASS" if ratio <= threshold else "FAIL"
    return CheckResult(
        ratio <= threshold,
        f"{verdict} — abs(entry-stop)/entry = {ratio:.4f}; "
        f"threshold = circuit_band_pct({circuit_band_pct:.4f}) * 0.8 = {threshold:.4f}",
    )


def _stt_floor_check(signal: TradingSignal) -> CheckResult:
    entry = getattr(signal, "entry_price", None)
    target_1 = getattr(signal, "target_1", None)
    if entry is None or target_1 is None:
        return CheckResult(False, "FAIL — entry_price or target_1 is None")
    if entry <= 0:
        return CheckResult(False, f"FAIL — non-positive entry_price={entry}")
    upside = target_1 / entry - 1
    threshold = 0.003
    verdict = "PASS" if upside >= threshold else "FAIL"
    return CheckResult(
        upside >= threshold,
        f"{verdict} — target_1/entry - 1 = {upside:.4f}; threshold = {threshold:.4f}",
    )


def _acceptance_failures(state: AgentState, settings: Settings) -> list[str]:
    from scripts.uat_pm_gate import _is_non_generic  # noqa: PLC0415

    failures = []
    ids = state.investment_debate_state
    rds = state.risk_debate_state
    bull = len(ids.bull_turns()) if ids is not None else 0
    bear = len(ids.bear_turns()) if ids is not None else 0
    risk = len(rds.risk_turns()) if rds is not None else 0
    if not (bull == bear == settings.max_debate_rounds and bull > 0):
        failures.append(
            f"Investment rounds: bull={bull}, bear={bear}; expected {settings.max_debate_rounds} "
            "each"
        )
    if not (risk == settings.max_risk_discuss_rounds and risk > 0):
        failures.append(f"Risk rounds: {risk}; expected {settings.max_risk_discuss_rounds}")

    decision = state.final_trade_decision
    if decision is None:
        return [*failures, "PortfolioDecision is missing"]
    band = state.technical_report.circuit_band_pct if state.technical_report is not None else None
    for label, check in (
        ("Circuit-band check", _circuit_band_check(decision.signal, band)),
        ("STT floor check", _stt_floor_check(decision.signal)),
    ):
        if not check.passed:
            failures.append(f"{label}: {check.detail}")
    if decision.decision not in {"approve", "modify", "veto"}:
        failures.append(f"Invalid PM decision: {decision.decision!r}")
    if not (decision.pm_notes or "").strip():
        failures.append("PM notes are missing")
    elif decision.decision == "approve" and not _is_non_generic(decision.pm_notes):
        failures.append("PM approval notes fail the non-generic justification check")
    return failures


async def _run_pipeline(date: str, settings: Settings) -> AgentState:
    from finagents.graph.trading_graph import (  # noqa: PLC0415
        FinAgentsGraph,
    )

    graph = FinAgentsGraph(full_pipeline=True)
    return await graph.apropagate(_TICKER, date, settings)


def _format_evidence_block(
    state: AgentState,
    timestamp: str,
    provider_label: str,
    deep_think_model: str,
    quick_think_model: str,
) -> str:
    ids = state.investment_debate_state
    rds = state.risk_debate_state
    bull_count = len(ids.bull_turns()) if ids is not None else 0
    bear_count = len(ids.bear_turns()) if ids is not None else 0
    risk_count = len(rds.risk_turns()) if rds is not None else 0

    decision = state.final_trade_decision
    if decision is None:
        return (
            f"\n### Run {timestamp}\n\n"
            f"- **LLM:** {provider_label} (deep={deep_think_model}, "
            f"quick={quick_think_model})\n"
            f"- **Bull/bear rounds:** {bull_count} / {bear_count}\n"
            f"- **Risk rounds:** {risk_count}\n"
            f"- **Circuit-band check:** N/A — final_trade_decision is None\n"
            f"- **STT floor check:** N/A — final_trade_decision is None\n"
            f"- **PortfolioDecision:** MISSING (PM never wrote)\n\n"
            f"FAIL: pipeline produced no PortfolioDecision.\n"
        )

    signal = decision.signal
    circuit_band_pct = (
        state.technical_report.circuit_band_pct if state.technical_report is not None else None
    )
    circuit = _circuit_band_check(signal, circuit_band_pct)
    stt = _stt_floor_check(signal)
    payload_json = decision.model_dump_json(indent=2)

    return (
        f"\n### Run {timestamp}\n\n"
        f"- **LLM:** {provider_label} (deep={deep_think_model}, "
        f"quick={quick_think_model})\n"
        f"- **Bull/bear rounds:** {bull_count} / {bear_count}\n"
        f"- **Risk rounds:** {risk_count}\n"
        f"- **Circuit-band check:** {circuit.detail}\n"
        f"- **STT floor check:** {stt.detail}\n"
        f"- **PortfolioDecision:** `{decision.decision}` "
        f"(approved={decision.approved})\n"
        f"- **pm_notes:** {decision.pm_notes}\n\n"
        f"<details>\n<summary>Full PortfolioDecision JSON payload</summary>\n\n"
        f"```json\n{payload_json}\n```\n\n</details>\n"
    )


def _append_to_verification_md(output: Path, block: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = output.read_text() if output.exists() else ""
    if _SECTION_MARKER in existing:
        new_content = existing + block
    else:
        new_content = existing + f"\n{_SECTION_MARKER}\n{block}"
    output.write_text(new_content)


async def _amain() -> int:
    args = _parse_args()

    if args.dry_run:
        print(f"[DRY RUN] Would run {_TICKER} full pipeline for date={args.date}.")
        print(f"[DRY RUN] Output target: {args.output}")
        print(f"[DRY RUN] Section marker: {_SECTION_MARKER!r}")
        if args.output.exists():
            existing = args.output.read_text()
            if _SECTION_MARKER in existing:
                print("[DRY RUN] Marker found — would append under existing section.")
            else:
                print("[DRY RUN] Marker absent — would append marker + block at EOF.")
        else:
            print("[DRY RUN] Report and parent directories would be created on first use.")
        return 0

    from finagents.config import Settings, resolve_model  # noqa: PLC0415

    settings = Settings()
    provider_label = getattr(settings, "llm_provider", "unknown")
    deep_think_model = resolve_model(settings, settings.llm_provider, "deep")
    quick_think_model = resolve_model(settings, settings.llm_provider, "quick")

    print(
        f"[acceptance] Running {_TICKER} full pipeline with "
        f"provider={provider_label}, deep={deep_think_model}, "
        f"quick={quick_think_model}, date={args.date}..."
    )
    try:
        state = await _run_pipeline(args.date, settings)
    except Exception as exc:
        timestamp = local_now().strftime("%Y-%m-%d %H:%M:%S")
        block = (
            f"\n### Run {timestamp} BLOCKED\n\n"
            f"- **LLM:** {provider_label} (deep={deep_think_model}, "
            f"quick={quick_think_model})\n"
            f"- **Outcome:** Pipeline raised: `{type(exc).__name__}: {exc}`\n"
            f"- **Audit trail:** Groq can return an HTTP 403 geo-block on this "
            f"code path; if the exception above is HTTPStatusError 403, switch "
            f"to a different provider via `LLM_PROVIDER=openrouter` (or `cerebras`) and re-run.\n"
        )
        _append_to_verification_md(args.output, block)
        print(f"[acceptance] BLOCKED: {exc}", file=sys.stderr)
        print(f"[acceptance] Audit trail appended to {args.output} §{_SECTION_MARKER}.")
        return 1

    timestamp = local_now().strftime("%Y-%m-%d %H:%M:%S")
    block = _format_evidence_block(
        state, timestamp, provider_label, deep_think_model, quick_think_model
    )
    failures = _acceptance_failures(state, settings)
    block += f"\n**Acceptance:** {'FAIL' if failures else 'PASS'}\n"
    block += "".join(f"- {failure}\n" for failure in failures)
    _append_to_verification_md(args.output, block)
    print(block)
    print(f"[acceptance] Evidence appended to {args.output} §{_SECTION_MARKER}.")
    return 1 if failures else 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())
