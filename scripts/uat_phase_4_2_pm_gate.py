#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_DEFAULT_OUTPUT = Path(
    ".planning/phases/04.2-risk-debate-pm-end-to-end/04.2-VERIFICATION.md"
)
_DEFAULT_TICKER_FILE = Path(
    ".planning/phases/04.2-risk-debate-pm-end-to-end/uat-ticker-batch.txt"
)
_SECTION_MARKER = "## PM Gate"
_MODIFICATION_RATE_THRESHOLD = 0.20
_NON_GENERIC_MIN_CHARS = 40

NON_GENERIC_TOKENS: tuple[str, ...] = (
    "concentration",
    "sector",
    "cash headroom",
    "overweight",
    "underweight",
    "portfolio value",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual dual-gate UAT for "
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated ticker list override (default: read from --ticker-file)",
    )
    parser.add_argument(
        "--ticker-file",
        type=Path,
        default=_DEFAULT_TICKER_FILE,
        help="Path to single-line CSV ticker file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Path to 04.2-VERIFICATION.md (default: %(default)s)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date passed to FinancialAgentsGraph.apropagate (default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Smoke test: verify args + ticker selection without invoking the LLM pipeline",
    )
    return parser.parse_args()


def _parse_tickers_arg(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.ticker_file.exists():
        text = args.ticker_file.read_text().strip()
        return [t.strip().upper() for t in text.split(",") if t.strip()]
    raise SystemExit(
        f"Provide --tickers or create {args.ticker_file} (single-line CSV)"
    )


def _is_non_generic(pm_notes: str | None) -> bool:
    if not pm_notes or len(pm_notes) < _NON_GENERIC_MIN_CHARS:
        return False
    lower = pm_notes.lower()
    return any(tok in lower for tok in NON_GENERIC_TOKENS)


def _modification_type(decision: str) -> str:
    if decision == "modify":
        return "size-delta"
    return "n/a"


async def _run_pipeline(ticker: str, date: str):
    from financialagents.graph.trading_graph import (
        FinancialAgentsGraph,
    )

    graph = FinancialAgentsGraph(full_pipeline=True)
    return await graph.apropagate(ticker, date)


class TickerRow(NamedTuple):
    ticker: str
    decision: str
    pm_notes: str
    non_generic: bool
    error: str = ""


def _format_row(row: TickerRow) -> str:
    if row.error:
        return (
            f"| {row.ticker} | error | n/a | "
            f"FAIL (exception: {row.error}) |"
        )
    classification = "non-generic" if row.non_generic else "generic"
    if row.decision != "approve":
        classification = "n/a (decision != approve)"
    pm_excerpt = (row.pm_notes or "")[:60].replace("|", "\\|")
    if len(row.pm_notes or "") > 60:
        pm_excerpt += "…"
    return (
        f"| {row.ticker} | {row.decision} | "
        f"{_modification_type(row.decision)} | {classification} — "
        f"`{pm_excerpt}` |"
    )


class GateVerdict(NamedTuple):
    modification_rate: float
    modify_count: int
    approve_count: int
    veto_count: int
    error_count: int
    total: int
    non_generic_approves: int
    passed: bool
    explanation: str


def _compute_verdict(rows: list[TickerRow]) -> GateVerdict:
    total = len(rows)
    modify_count = sum(1 for r in rows if r.decision == "modify")
    approve_count = sum(1 for r in rows if r.decision == "approve")
    veto_count = sum(1 for r in rows if r.decision == "veto")
    error_count = sum(1 for r in rows if r.error)
    non_generic_approves = sum(1 for r in rows if r.decision == "approve" and r.non_generic)

    modification_rate = modify_count / total if total else 0.0
    rate_gate = modification_rate >= _MODIFICATION_RATE_THRESHOLD
    coverage_gate = (
        approve_count > 0 and non_generic_approves == approve_count
    )

    if rate_gate and coverage_gate:
        explanation = (
            f"PASS (modification_rate={modification_rate:.2f} >= "
            f"{_MODIFICATION_RATE_THRESHOLD:.2f}; "
            f"all {approve_count} approves are non-generic)"
        )
    elif rate_gate:
        explanation = (
            f"PASS (modification_rate={modification_rate:.2f} >= "
            f"{_MODIFICATION_RATE_THRESHOLD:.2f})"
        )
    elif coverage_gate:
        explanation = (
            f"PASS (all {approve_count} approves are non-generic; "
            f"modification_rate={modification_rate:.2f})"
        )
    else:
        explanation = (
            f"FAIL (modification_rate={modification_rate:.2f} < "
            f"{_MODIFICATION_RATE_THRESHOLD:.2f} AND "
            f"{approve_count - non_generic_approves}/{approve_count} "
            f"approves are generic)"
        )
    return GateVerdict(
        modification_rate=modification_rate,
        modify_count=modify_count,
        approve_count=approve_count,
        veto_count=veto_count,
        error_count=error_count,
        total=total,
        non_generic_approves=non_generic_approves,
        passed=rate_gate or coverage_gate,
        explanation=explanation,
    )


def _build_evidence_block(rows: list[TickerRow], timestamp: str) -> str:
    verdict = _compute_verdict(rows)
    lines = [
        f"\n### Run {timestamp}\n",
        "| Ticker | Decision | Modification type | pm_notes classification (generic / non-generic) |",
        "|--------|----------|-------------------|-------------------------------------------------|",
    ]
    lines.extend(_format_row(r) for r in rows)
    lines.append("")
    lines.append(
        f"**Counts:** approve={verdict.approve_count}, modify={verdict.modify_count}, "
        f"veto={verdict.veto_count}, errors={verdict.error_count}, total={verdict.total}"
    )
    lines.append(
        f"**Verdict:** modification_rate={verdict.modification_rate:.2f}; "
        f"non_generic_approve_coverage={verdict.non_generic_approves}/{verdict.approve_count} "
        f"→ {verdict.explanation}"
    )
    return "\n".join(lines) + "\n"


def _append_to_verification_md(output: Path, block: str) -> None:
    if not output.exists():
        raise FileNotFoundError(
            f"VERIFICATION.md not found at {output} — Task 2 should "
            f"have scaffolded it before this UAT runs."
        )
    existing = output.read_text()
    if _SECTION_MARKER in existing:
        new_content = existing + block
    else:
        new_content = existing + f"\n{_SECTION_MARKER}\n{block}"
    output.write_text(new_content)


async def _amain() -> int:
    args = _parse_args()
    tickers = _parse_tickers_arg(args)

    if args.dry_run:
        print(f"[DRY RUN] Tickers ({len(tickers)}): {', '.join(tickers)}")
        print(f"[DRY RUN] Output target: {args.output}")
        print(f"[DRY RUN] Section marker: {_SECTION_MARKER!r}")
        print(
            f"[DRY RUN] Dual gate: modification_rate >= "
            f"{_MODIFICATION_RATE_THRESHOLD:.2f} OR all approves non-generic "
            f"(>= {_NON_GENERIC_MIN_CHARS} chars + token from "
            f"{NON_GENERIC_TOKENS!r})"
        )
        return 0

    rows: list[TickerRow] = []
    for i, ticker in enumerate(tickers, 1):
        print(f"[UAT] [{i}/{len(tickers)}] {ticker}...", flush=True)
        try:
            state = await _run_pipeline(ticker, args.date)
        except Exception as exc:
            rows.append(
                TickerRow(
                    ticker=ticker,
                    decision="error",
                    pm_notes="",
                    non_generic=False,
                    error=str(exc)[:120],
                )
            )
            continue
        decision_obj = state.final_trade_decision
        if decision_obj is None:
            rows.append(
                TickerRow(
                    ticker=ticker,
                    decision="missing",
                    pm_notes="(no PortfolioDecision emitted by PM)",
                    non_generic=False,
                )
            )
            continue
        rows.append(
            TickerRow(
                ticker=ticker,
                decision=decision_obj.decision,
                pm_notes=decision_obj.pm_notes,
                non_generic=_is_non_generic(decision_obj.pm_notes)
                if decision_obj.decision == "approve"
                else False,
            )
        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = _build_evidence_block(rows, timestamp)
    _append_to_verification_md(args.output, block)
    print(block)
    print(f"[UAT] Evidence appended to {args.output} §{_SECTION_MARKER}.")

    verdict = _compute_verdict(rows)
    if not verdict.passed:
        print(f"[UAT] FAIL: {verdict.explanation}", file=sys.stderr)
        return 1
    print(f"[UAT] PASS: {verdict.explanation}")
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())
