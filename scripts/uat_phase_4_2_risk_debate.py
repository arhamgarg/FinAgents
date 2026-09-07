#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_DEFAULT_OUTPUT = Path("reports/uat/risk-debate.md")
_DEFAULT_TICKER_FILE = _REPO_ROOT / "scripts" / "uat-ticker-batch.txt"
_SECTION_MARKER = "## Persona Divergence"
_DIVERGENCE_THRESHOLD = 0.30


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual persona-divergence UAT for "
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
        help="Path to the Markdown UAT report (default: %(default)s)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date passed to FinAgentsGraph.apropagate (default: today)",
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
    raise SystemExit(f"Provide --tickers or create {args.ticker_file} (single-line CSV)")


class PersonaSizes(NamedTuple):
    aggressive: float
    neutral: float
    conservative: float


def _parse_size(assessment_json: str | None) -> float:
    if not assessment_json:
        return 0.0
    try:
        data = json.loads(assessment_json)
        return float(data.get("recommended_position_size_pct", 0.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def _extract_persona_sizes(risk_debate_state) -> PersonaSizes:
    if risk_debate_state is None:
        return PersonaSizes(0.0, 0.0, 0.0)
    return PersonaSizes(
        aggressive=_parse_size(risk_debate_state.aggressive_assessment),
        neutral=_parse_size(risk_debate_state.neutral_assessment),
        conservative=_parse_size(risk_debate_state.conservative_assessment),
    )


async def _run_pipeline(ticker: str, date: str):
    from finagents.graph.trading_graph import (
        FinAgentsGraph,
    )

    graph = FinAgentsGraph(full_pipeline=True)
    return await graph.apropagate(ticker, date)


class TickerRow(NamedTuple):
    ticker: str
    sizes: PersonaSizes
    divergence: float
    passed: bool
    error: str = ""


def _format_row(row: TickerRow) -> str:
    if row.error:
        return f"| {row.ticker} | n/a | n/a | n/a | n/a | FAIL (exception: {row.error}) |"
    verdict = "PASS" if row.passed else "FAIL"
    return (
        f"| {row.ticker} | {row.sizes.aggressive:.3f} | "
        f"{row.sizes.neutral:.3f} | {row.sizes.conservative:.3f} | "
        f"{row.divergence:.3f} | {verdict} |"
    )


def _build_evidence_block(rows: list[TickerRow], timestamp: str) -> str:
    table_lines = [
        f"\n### Run {timestamp}\n",
        "| Ticker | Aggressive size | Neutral size | Conservative size | Divergence | PASS/FAIL |",
        "|--------|-----------------|--------------|-------------------|------------|-----------|",
    ]
    table_lines.extend(_format_row(r) for r in rows)
    pass_count = sum(1 for r in rows if r.passed and not r.error)
    total = len(rows)
    verdict_label = "PASS" if pass_count == total else "FAIL"
    table_lines.append("")
    table_lines.append(
        f"**Verdict:** {pass_count} / {total} tickers PASS "
        f"(threshold: divergence >= {_DIVERGENCE_THRESHOLD:.2f}) — {verdict_label}"
    )
    return "\n".join(table_lines) + "\n"


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
    tickers = _parse_tickers_arg(args)

    if args.dry_run:
        print(f"[DRY RUN] Tickers ({len(tickers)}): {', '.join(tickers)}")
        print(f"[DRY RUN] Output target: {args.output}")
        print(f"[DRY RUN] Section marker: {_SECTION_MARKER!r}")
        print(f"[DRY RUN] Acceptance: divergence >= {_DIVERGENCE_THRESHOLD:.2f} per ticker")
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
                    sizes=PersonaSizes(0.0, 0.0, 0.0),
                    divergence=0.0,
                    passed=False,
                    error=str(exc)[:120],
                )
            )
            continue
        sizes = _extract_persona_sizes(state.risk_debate_state)
        size_values = (sizes.aggressive, sizes.neutral, sizes.conservative)
        divergence = max(size_values) - min(size_values)
        passed = divergence >= _DIVERGENCE_THRESHOLD
        rows.append(
            TickerRow(
                ticker=ticker,
                sizes=sizes,
                divergence=divergence,
                passed=passed,
            )
        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = _build_evidence_block(rows, timestamp)
    _append_to_verification_md(args.output, block)
    print(block)
    print(f"[UAT] Evidence appended to {args.output} §{_SECTION_MARKER}.")

    failed = [r for r in rows if not r.passed]
    if failed:
        print(
            f"[UAT] FAIL: {len(failed)} of {len(rows)} tickers below divergence "
            f"threshold {_DIVERGENCE_THRESHOLD:.2f} — persona-prompt "
            "iteration recommended (max 3 iterations).",
            file=sys.stderr,
        )
        return 1

    print(
        f"[UAT] PASS: all {len(rows)} tickers cleared divergence threshold "
        f"{_DIVERGENCE_THRESHOLD:.2f} — acceptance satisfied."
    )
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())
