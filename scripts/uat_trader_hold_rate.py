#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual hold-rate UAT for trader_node "
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated ticker list override (default: adversarial selection)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            ".planning/phases/04.1-trader-position-sizing-guardrails/"
            "04.1-VERIFICATION.md"
        ),
        help="Path to VERIFICATION.md file to append the hold-rate table to",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Smoke test: verify ticker selection + table emitter without invoking the LLM pipeline",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date passed to FinancialAgentsGraph.propagate (default: today)",
    )
    parser.add_argument(
        "--sec-list-fixture",
        type=Path,
        default=Path("tests/fixtures/nse/sec_list_sample.csv"),
        help="Offline fixture used by --dry-run; ignored in non-dry-run mode (which fetches live)",
    )
    return parser.parse_args()


T2T_SERIES = frozenset({"BE", "BZ", "ST"})


class TickerRow(NamedTuple):
    symbol: str
    series: str
    band: str
    remarks: str


def load_sec_list(source: Path | bytes) -> list[TickerRow]:
    if isinstance(source, Path):
        content = source.read_bytes()
    else:
        content = source
    rows: list[TickerRow] = []
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    for row in reader:
        rows.append(
            TickerRow(
                symbol=row["Symbol"].strip().upper(),
                series=row["Series"].strip().upper(),
                band=row["Band"].strip(),
                remarks=row.get("Remarks", "").strip(),
            )
        )
    return rows


def select_adversarial_batch(rows: list[TickerRow], size: int = 10) -> list[str]:
    eligible = [r for r in rows if r.band != "No Band"]
    preferred = [r for r in eligible if r.band in {"10", "20"}]
    t2t = sorted([r.symbol for r in preferred if r.series in T2T_SERIES])
    normal = sorted([r.symbol for r in preferred if r.series not in T2T_SERIES])
    picked: list[str] = t2t[: min(2, len(t2t))] + normal[: size - min(2, len(t2t))]
    if len(picked) < size:
        fallback = sorted([r.symbol for r in eligible if r.symbol not in picked])
        picked += fallback[: size - len(picked)]
    return picked[:size]


class UatResult(NamedTuple):
    ticker: str
    overall_confidence: float
    debate_direction: str
    trader_action: str
    hold_reason: str
    nonzero_analyst_count: int
    analyst_confidences: dict[str, float]


def _format_per_analyst(conf: dict[str, float]) -> str:
    return (
        f"t={conf.get('technical', 0.0):.2f} "
        f"f={conf.get('fundamentals', 0.0):.2f} "
        f"n={conf.get('news', 0.0):.2f}"
    )


DATA_RICH_MIN_NONZERO = 2
DATA_RICH_CONF_THRESHOLD = 0.6
DATA_POOR_CONF_THRESHOLD = 0.4
HOLD_RATE_PASS_PCT = 50.0


_RE_BUY = re.compile(r"\b(?:BUY|ACCUMULATE|LONG)\b", re.IGNORECASE)
_RE_SELL = re.compile(r"\b(?:SELL|REDUCE|TRIM|EXIT|SHORT)\b", re.IGNORECASE)


def scan_debate_direction(investment_thesis: str) -> str:
    has_buy = bool(_RE_BUY.search(investment_thesis or ""))
    has_sell = bool(_RE_SELL.search(investment_thesis or ""))
    if has_buy and not has_sell:
        return "BUY"
    if has_sell and not has_buy:
        return "SELL"
    return "MIXED"


def _tier_hold_rate(
    results: list[UatResult],
    tier_predicate,
    conf_threshold: float,
) -> tuple[int, int, float | None]:
    cohort = [
        r
        for r in results
        if tier_predicate(r)
        and r.overall_confidence >= conf_threshold
        and r.debate_direction in {"BUY", "SELL"}
    ]
    total = len(cohort)
    holds = sum(1 for r in cohort if r.trader_action == "HOLD")
    pct = (holds / total * 100.0) if total else None
    return holds, total, pct


def _format_tier_line(label: str, threshold: float, holds: int, total: int, pct: float | None) -> str:
    if pct is None:
        return (
            f"- **{label}** (overall_confidence >= {threshold:.1f} AND directional): "
            f"{holds} / {total} = N/A (empty tier — counts as N/A per "
        )
    verdict = "PASS" if pct < HOLD_RATE_PASS_PCT else "FAIL"
    return (
        f"- **{label}** (overall_confidence >= {threshold:.1f} AND directional): "
        f"{holds} / {total} = {pct:.1f}% — {verdict} (criterion: < {HOLD_RATE_PASS_PCT:.0f}%)"
    )


def format_results_table(results: list[UatResult]) -> str:
    lines = [
        "| ticker | overall_confidence | per_analyst (t/f/n) | debate_direction | trader_action | nonzero_analysts | hold_reason |",
        "|--------|-------------------|---------------------|------------------|---------------|------------------|-------------|",
    ]
    for r in results:
        reason = (r.hold_reason or "").replace("|", "\\|")[:80]
        lines.append(
            f"| {r.ticker} | {r.overall_confidence:.2f} | {_format_per_analyst(r.analyst_confidences)} "
            f"| {r.debate_direction} | {r.trader_action} | {r.nonzero_analyst_count}/3 | {reason} |"
        )

    rich_holds, rich_total, rich_pct = _tier_hold_rate(
        results,
        lambda r: r.nonzero_analyst_count >= DATA_RICH_MIN_NONZERO,
        DATA_RICH_CONF_THRESHOLD,
    )
    poor_holds, poor_total, poor_pct = _tier_hold_rate(
        results,
        lambda r: r.nonzero_analyst_count < DATA_RICH_MIN_NONZERO,
        DATA_POOR_CONF_THRESHOLD,
    )

    lines.append("")
    lines.append("**Tiered hold-rate (ROADMAP SC-5, / **")
    lines.append(_format_tier_line("data-rich", DATA_RICH_CONF_THRESHOLD, rich_holds, rich_total, rich_pct))
    lines.append(_format_tier_line("data-poor", DATA_POOR_CONF_THRESHOLD, poor_holds, poor_total, poor_pct))
    return "\n".join(lines) + "\n"


async def run_pipeline(ticker: str, date: str) -> UatResult:
    from financialagents.graph.trading_graph import FinancialAgentsGraph

    graph = FinancialAgentsGraph(full_pipeline=True)
    state = await graph.apropagate(ticker, date)
    overall = (
        state.research_summary.overall_confidence
        if state.research_summary is not None
        else 0.0
    )
    per_analyst: dict[str, float] = (
        dict(state.research_summary.analyst_confidences)
        if state.research_summary is not None
        else {}
    )
    nonzero_count = sum(1 for v in per_analyst.values() if v > 0)
    thesis = (
        state.investment_debate_state.investment_thesis
        if state.investment_debate_state is not None
        else ""
    )
    signal = state.trader_signal
    if signal is None:
        return UatResult(
            ticker, overall, scan_debate_direction(thesis),
            "HOLD", "trader_signal missing", nonzero_count, per_analyst,
        )
    reason = signal.circuit_band_note or ""
    if not reason and signal.action == "HOLD":
        reason = (signal.rationale or "")[:80]
    return UatResult(
        ticker, overall, scan_debate_direction(thesis),
        signal.action, reason, nonzero_count, per_analyst,
    )


def main() -> int:
    args = _parse_args()

    if args.dry_run:
        if not args.sec_list_fixture.exists():
            print(
                f"ERROR: --dry-run fixture not found at {args.sec_list_fixture}",
                file=sys.stderr,
            )
            return 2
        rows = load_sec_list(args.sec_list_fixture)
    else:
        import httpx
        NSE_URL = "https://nsearchives.nseindia.com/content/equities/sec_list.csv"
        resp = httpx.get(
            NSE_URL,
            headers={"User-Agent": "financialagents/1.0"},
            timeout=30.0,
        )
        resp.raise_for_status()
        rows = load_sec_list(resp.content)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = select_adversarial_batch(rows, size=10)

    no_band_symbols = {r.symbol for r in rows if r.band == "No Band"}
    leaked = [t for t in tickers if t in no_band_symbols]
    if leaked:
        print(
            f"ERROR: OQ-1 violation — 'No Band' tickers in batch: {leaked}",
            file=sys.stderr,
        )
        return 3

    if args.dry_run:
        print(f"[dry-run] Selected {len(tickers)} tickers: {', '.join(tickers)}")
        print("[dry-run] OQ-1 'No Band' exclusion: OK")
        synthetic = [
            UatResult(
                t, 0.70, "BUY", "BUY", "",
                3 if i % 2 == 0 else 1,
                {"technical": 0.78, "fundamentals": 0.6, "news": 0.6}
                if i % 2 == 0
                else {"technical": 0.78, "fundamentals": 0.0, "news": 0.0},
            )
            for i, t in enumerate(tickers)
        ]
        table = format_results_table(synthetic)
        print("[dry-run] Synthetic results table:")
        print(table)
        return 0

    print(f"Running full pipeline on {len(tickers)} tickers for date={args.date}...")
    results: list[UatResult] = []
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {ticker}...", flush=True)
        try:
            result = asyncio.run(run_pipeline(ticker, args.date))
        except Exception as exc:
            result = UatResult(
                ticker, 0.0, "MIXED", "HOLD", f"pipeline error: {exc}"[:80], 0, {}
            )
        results.append(result)

    table = format_results_table(results)
    print("\n" + table)

    if not args.output.exists():
        print(f"ERROR: VERIFICATION.md not found at {args.output}", file=sys.stderr)
        print(
            "Run Task 2 (scaffold 04.1-VERIFICATION.md) before running this UAT.",
            file=sys.stderr,
        )
        return 4
    existing = args.output.read_text()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = (
        f"\n### Run {stamp}\n\n"
        f"**Batch:** {', '.join(tickers)}\n\n"
        f"{table}\n"
    )
    marker = "## Hold-Rate Results"
    if marker in existing:
        new_content = existing + block
    else:
        new_content = existing + f"\n{marker}\n{block}"
    args.output.write_text(new_content)
    print(f"Results appended to {args.output}")

    rich_holds, rich_total, rich_pct = _tier_hold_rate(
        results,
        lambda r: r.nonzero_analyst_count >= DATA_RICH_MIN_NONZERO,
        DATA_RICH_CONF_THRESHOLD,
    )
    poor_holds, poor_total, poor_pct = _tier_hold_rate(
        results,
        lambda r: r.nonzero_analyst_count < DATA_RICH_MIN_NONZERO,
        DATA_POOR_CONF_THRESHOLD,
    )

    if rich_total == 0 and poor_total == 0:
        print(
            "WARNING: both tiers empty — no directional+confident tickers "
            "in either tier; hold-rate undefined; adversarial batch too weak."
        )
        return 1

    rich_fail = rich_pct is not None and rich_pct >= HOLD_RATE_PASS_PCT
    poor_fail = poor_pct is not None and poor_pct >= HOLD_RATE_PASS_PCT

    rich_msg = (
        "data-rich: N/A (empty)" if rich_pct is None
        else f"data-rich: {rich_holds}/{rich_total} = {rich_pct:.1f}% "
             f"({'FAIL' if rich_fail else 'PASS'})"
    )
    poor_msg = (
        "data-poor: N/A (empty)" if poor_pct is None
        else f"data-poor: {poor_holds}/{poor_total} = {poor_pct:.1f}% "
             f"({'FAIL' if poor_fail else 'PASS'})"
    )

    if rich_fail or poor_fail:
        print(
            f"FAIL: tiered hold-rate — {rich_msg}; {poor_msg} — "
            " iteration required (max 3 iterations)."
        )
        return 1

    print(
        f"PASS: tiered hold-rate — {rich_msg}; {poor_msg} — "
        " phase gate satisfied /."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
