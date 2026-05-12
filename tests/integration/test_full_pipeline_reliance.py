from __future__ import annotations

import asyncio
from collections.abc import Callable
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from financialagents.agents.managers.schemas import PortfolioDecision
from financialagents.agents.utils.agent_states import AgentState
from financialagents.config import Settings
from financialagents.dataflows.data_result import DataResult
from financialagents.testing.llm_mock import LLMMockHarness, failures, fixtures


def _synthetic_ohlcv_250() -> pd.DataFrame:
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(250) * 0.5)
    dates = pd.date_range("2024-01-01", periods=250, freq="B")
    return pd.DataFrame(
        {
            "open": price + np.random.randn(250) * 0.3,
            "high": price + np.abs(np.random.randn(250)) * 0.5,
            "low": price - np.abs(np.random.randn(250)) * 0.5,
            "close": price,
            "volume": np.random.randint(100_000, 1_000_000, 250).astype(float),
        },
        index=dates,
    )


def _run_pipeline_with_harness(
    max_debate_rounds: int,
    max_risk_discuss_rounds: int,
    harness_setup_fn: Callable[[LLMMockHarness], None] | None = None,
) -> AgentState:
    from financialagents.graph.setup import build_full_pipeline

    ohlcv_result: DataResult[pd.DataFrame] = DataResult(
        status="ok", data=_synthetic_ohlcv_250(), source="angelone-synthetic"
    )
    empty_fundamentals: DataResult = DataResult(
        status="empty", data=None, source="harness-empty"
    )
    empty_news: DataResult = DataResult(
        status="empty", data=None, source="harness-empty"
    )

    with (
        patch(
            "financialagents.agents.analysts.technical_analyst.get_ohlcv",
            new_callable=AsyncMock,
            return_value=ohlcv_result,
        ),
        patch(
            "financialagents.agents.analysts.fundamentals_analyst.get_fundamentals",
            new_callable=AsyncMock,
            return_value=empty_fundamentals,
        ),
        patch(
            "financialagents.agents.analysts.news_analyst.get_news",
            new_callable=AsyncMock,
            return_value=empty_news,
        ),
        LLMMockHarness.happy_path() as h,
    ):
        if harness_setup_fn is not None:
            harness_setup_fn(h)
        graph = build_full_pipeline()
        settings = Settings(_env_file=None)
        settings.max_debate_rounds = max_debate_rounds
        settings.max_risk_discuss_rounds = max_risk_discuss_rounds
        config = {
            "configurable": {
                "settings": settings,
                "thread_id": "reliance-ci-smoke",
            }
        }
        initial = AgentState(ticker="RELIANCE", date="2024-12-31")
        result = asyncio.run(graph.ainvoke(initial, config=config))
    return AgentState.model_validate(result) if isinstance(result, dict) else result


class TestFullPipelineReliance:

    def test_full_pipeline_emits_portfolio_decision(self) -> None:
        final = _run_pipeline_with_harness(
            max_debate_rounds=1, max_risk_discuss_rounds=1
        )

        assert final.final_trade_decision is not None
        assert isinstance(final.final_trade_decision, PortfolioDecision)
        assert final.final_trade_decision.decision in {"approve", "modify", "veto"}
        assert final.final_trade_decision.pm_notes
        assert final.final_trade_decision.signal is not None
        assert final.final_trade_decision.signal.action in {"BUY", "HOLD", "SELL"}

    def test_full_pipeline_invest_round_count_preserved(self) -> None:
        final = _run_pipeline_with_harness(
            max_debate_rounds=2, max_risk_discuss_rounds=1
        )

        ids = final.investment_debate_state
        assert ids is not None
        assert len(ids.bull_turns()) == 2, (
            f" round-count off-by-one: bull_turns={ids.bull_turns()}"
        )
        assert len(ids.bear_turns()) == 2, (
            f" round-count off-by-one: bear_turns={ids.bear_turns()}"
        )
        assert ids.investment_thesis is not None and len(ids.investment_thesis) > 0

    def test_full_pipeline_risk_round_count(self) -> None:
        final = _run_pipeline_with_harness(
            max_debate_rounds=1, max_risk_discuss_rounds=2
        )

        rds = final.risk_debate_state
        assert rds is not None, (
            " risk_init_node must seed risk_debate_state before subgraph"
        )
        assert len(rds.risk_turns()) == 2, (
            f" round-count off-by-one: risk_turns={rds.risk_turns()}"
        )
        assert rds.synthesized_position_size_pct == 0.030

    def test_full_pipeline_recursion_limit_applied(self) -> None:
        import financialagents.graph.setup as setup_mod
        from financialagents.agents.utils.agent_states import RiskDebateState

        captured_configs: list[dict] = []

        class _SpySubgraph:
            async def ainvoke(self, state, config=None):
                captured_configs.append(config or {})
                return state.model_dump() if hasattr(state, "model_dump") else state

        original_cache = setup_mod._RISK_DEBATE_SUBGRAPH
        setup_mod._RISK_DEBATE_SUBGRAPH = _SpySubgraph()  # ty: ignore[invalid-assignment]
        try:
            seeded_state = AgentState(
                ticker="RELIANCE",
                date="2024-01-15",
                risk_debate_state=RiskDebateState(max_rounds=2),
            )
            asyncio.run(setup_mod.call_risk_debate_subgraph(seeded_state))
        finally:
            setup_mod._RISK_DEBATE_SUBGRAPH = original_cache

        assert captured_configs, "wrapper never called the subgraph — wiring broken"
        assert captured_configs[0].get("recursion_limit") == 16, (
            f" formula violated: expected recursion_limit=16, "
            f"got {captured_configs[0]}"
        )




@pytest.mark.parametrize(
    "scenario",
    ["happy", "single_node_fail", "multi_node_fail", "retry_then_success"],
)
def test_full_pipeline_failure_scenarios(scenario: str) -> None:

    def setup(h: LLMMockHarness) -> None:
        if scenario == "happy":
            pass
        elif scenario == "single_node_fail":
            h.set_sequence(
                "bull_researcher",
                [failures.length_error(), failures.length_error()],
            )
        elif scenario == "multi_node_fail":
            active_failure_sites = ("bull_researcher", "bear_researcher")
            dormant_failure_sites = ("news_analyst",)
            for site in active_failure_sites + dormant_failure_sites:
                h.set_sequence(
                    site,
                    [failures.length_error(), failures.length_error()],
                )
        elif scenario == "retry_then_success":
            h.set_sequence(
                "trader",
                failures.retry_then_succeed(fixtures.TRADER_SIGNAL_DEFAULT),
            )
        else:
            raise AssertionError(f"unhandled scenario: {scenario!r}")

    final = _run_pipeline_with_harness(
        max_debate_rounds=1,
        max_risk_discuss_rounds=1,
        harness_setup_fn=setup,
    )

    assert final.final_trade_decision is not None, (
        f"scenario={scenario!r}: pipeline did not produce a PortfolioDecision"
    )
    assert isinstance(final.final_trade_decision, PortfolioDecision), (
        f"scenario={scenario!r}: final_trade_decision is not a PortfolioDecision"
    )

    sentinel = failures.length_error.SENTINEL  # ty: ignore[unresolved-attribute]
    assert final.investment_debate_state is not None
    bulls = final.investment_debate_state.bull_turns()
    if scenario in ("single_node_fail", "multi_node_fail"):
        assert any(sentinel in t for t in bulls), (
            f"scenario={scenario!r}: expected SENTINEL {sentinel!r} substring "
            f"in bull_history but got {bulls!r}"
        )
    elif scenario == "retry_then_success":
        assert not any(sentinel in t for t in bulls), (
            f"scenario={scenario!r}: trader retry succeeded so no sentinel "
            f"should have fired in bull_history; got {bulls!r}"
        )
