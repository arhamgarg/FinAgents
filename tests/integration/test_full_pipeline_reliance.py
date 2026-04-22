from __future__ import annotations

import asyncio
import importlib
from unittest.mock import patch

from financialagents.agents.analysts.schemas import (
    FundamentalsAnalystReport,
    NewsReport,
    TechnicalReport,
)
from financialagents.agents.managers.schemas import PortfolioDecision, ResearchBrief
from financialagents.agents.risk_mgmt.schemas import PersonaAssessment
from financialagents.agents.trader.schemas import TradingSignal
from financialagents.agents.utils.agent_states import AgentState
from financialagents.config import Settings


_TECH_REPORT = TechnicalReport(
    confidence=0.8,
    current_price=2500.0,
    atr_14=35.0,
    circuit_band_pct=0.20,
    t2t_flag=False,
)
_FUND_REPORT = FundamentalsAnalystReport(confidence=0.8)
_NEWS_REPORT = NewsReport(confidence=0.7)
_BRIEF = ResearchBrief(
    consensus_view="bullish on RELIANCE",
    overall_confidence=0.75,
)


def _build_trader_signal() -> TradingSignal:
    return TradingSignal(
        action="BUY",
        conviction=0.7,
        entry_price=2500.0,
        stop_loss=2450.0,
        target_1=2600.0,
        target_2=2700.0,
        position_size_pct=0.03,
        position_size_shares=30,
        risk_reward_ratio=2.0,
        rationale="mocked trader signal for CI smoke",
    )


def _build_pm_decision(signal: TradingSignal) -> PortfolioDecision:
    return PortfolioDecision(
        signal=signal,
        approved=True,
        decision="approve",
        pm_notes=(
            "approved — sector concentration low; cash headroom available; "
            "size matches risk-debate consolidation"
        ),
    )


async def _fake_tech(state, config=None):
    return {"technical_report": _TECH_REPORT}


async def _fake_fund(state, config=None):
    return {"fundamentals_report": _FUND_REPORT}


async def _fake_news(state, config=None):
    return {"news_report": _NEWS_REPORT}


async def _fake_rm(state, config=None):
    return {"research_summary": _BRIEF}


async def _fake_bull(state, config=None):
    return {"bull_history": {state.current_round: f"bull-{state.current_round}"}}


async def _fake_bear(state, config=None):
    return {
        "bear_history": {state.current_round: f"bear-{state.current_round}"},
        "current_round": state.current_round + 1,
    }


async def _fake_plan(state, config=None):
    bulls = state.bull_turns()
    return {"investment_plan": f"plan after {len(bulls)} rounds — accumulate"}


async def _fake_trader(state, config=None):
    return {"trader_signal": _build_trader_signal()}


async def _fake_aggressive(state, config=None):
    pa = PersonaAssessment(
        main_argument="aggressive thesis — go heavier on conviction",
        recommended_position_size_pct=0.05,
        confidence=0.8,
    )
    return {"aggressive_assessment": pa.model_dump_json()}


async def _fake_neutral(state, config=None):
    pa = PersonaAssessment(
        main_argument="neutral thesis — split the difference",
        counter_to_prior_persona="disagree with aggressive size",
        recommended_position_size_pct=0.03,
        confidence=0.6,
    )
    return {"neutral_assessment": pa.model_dump_json()}


async def _fake_conservative(state, config=None):
    pa = PersonaAssessment(
        main_argument="conservative thesis — prefer half-size",
        counter_to_prior_persona="prefer smaller exposure than neutral",
        recommended_position_size_pct=0.01,
        confidence=0.9,
    )
    return {"conservative_assessment": pa.model_dump_json()}


async def _fake_consolidation(state, config=None):
    round_key = state.current_round
    return {
        "risk_debate_history": {
            round_key: f"weighted 3% position view (round {round_key})"
        },
        "current_round": state.current_round + 1,
        "synthesized_position_size_pct": 0.030,
    }


async def _fake_pm(state, config=None):
    sig = (
        state.trader_signal.model_copy()
        if state.trader_signal is not None
        else _build_trader_signal()
    )
    return {"final_trade_decision": _build_pm_decision(sig)}


def _build_node_patches() -> list:
    return [
        patch(
            "financialagents.agents.analysts.technical_analyst.technical_analyst_node",
            _fake_tech,
        ),
        patch(
            "financialagents.agents.analysts.fundamentals_analyst.fundamentals_analyst_node",
            _fake_fund,
        ),
        patch(
            "financialagents.agents.analysts.news_analyst.news_analyst_node",
            _fake_news,
        ),
        patch(
            "financialagents.agents.managers.research_manager.research_manager_node",
            _fake_rm,
        ),
        patch(
            "financialagents.agents.researchers.bull_researcher.bull_researcher_node",
            _fake_bull,
        ),
        patch(
            "financialagents.agents.researchers.bear_researcher.bear_researcher_node",
            _fake_bear,
        ),
        patch(
            "financialagents.agents.researchers.investment_plan.investment_plan_node",
            _fake_plan,
        ),
        patch(
            "financialagents.agents.trader.trader.trader_node",
            _fake_trader,
        ),
        patch(
            "financialagents.agents.risk_mgmt.aggressive.aggressive_risk_node",
            _fake_aggressive,
        ),
        patch(
            "financialagents.agents.risk_mgmt.neutral.neutral_risk_node",
            _fake_neutral,
        ),
        patch(
            "financialagents.agents.risk_mgmt.conservative.conservative_risk_node",
            _fake_conservative,
        ),
        patch(
            "financialagents.agents.risk_mgmt.consolidation.risk_debate_consolidation_node",
            _fake_consolidation,
        ),
        patch(
            "financialagents.agents.managers.portfolio_manager.portfolio_manager_node",
            _fake_pm,
        ),
    ]


def _run_pipeline_with_patches(
    max_debate_rounds: int,
    max_risk_discuss_rounds: int,
    extra_patches: list | None = None,
) -> AgentState:
    import financialagents.graph.setup as setup_mod
    import financialagents.graph.subgraphs.invest_debate as invest_sub_mod
    import financialagents.graph.subgraphs.risk_debate as risk_sub_mod

    patches = _build_node_patches() + (extra_patches or [])
    started: list = []
    try:
        for p in patches:
            p.start()
            started.append(p)

        importlib.reload(invest_sub_mod)
        importlib.reload(risk_sub_mod)
        importlib.reload(setup_mod)
        setup_mod._reset_invest_debate_subgraph_cache()
        setup_mod._reset_risk_debate_subgraph_cache()

        graph = setup_mod.build_full_pipeline()
        settings = Settings(_env_file=None)
        settings.max_debate_rounds = max_debate_rounds
        settings.max_risk_discuss_rounds = max_risk_discuss_rounds
        config = {
            "configurable": {
                "settings": settings,
                "thread_id": "reliance-ci-smoke",
            }
        }
        initial = AgentState(ticker="RELIANCE", date="2024-01-15")
        result = asyncio.run(graph.ainvoke(initial, config=config))
    finally:
        for p in reversed(started):
            p.stop()
        importlib.reload(invest_sub_mod)
        importlib.reload(risk_sub_mod)
        importlib.reload(setup_mod)
        setup_mod._reset_invest_debate_subgraph_cache()
        setup_mod._reset_risk_debate_subgraph_cache()

    return AgentState.model_validate(result) if isinstance(result, dict) else result


class TestFullPipelineReliance:

    def test_full_pipeline_emits_portfolio_decision(self) -> None:
        final = _run_pipeline_with_patches(
            max_debate_rounds=1, max_risk_discuss_rounds=1
        )

        assert final.final_trade_decision is not None
        assert isinstance(final.final_trade_decision, PortfolioDecision)
        assert final.final_trade_decision.decision in {"approve", "modify", "veto"}
        assert final.final_trade_decision.pm_notes
        assert final.final_trade_decision.signal is not None
        assert final.final_trade_decision.signal.action in {"BUY", "HOLD", "SELL"}

    def test_full_pipeline_invest_round_count_preserved(self) -> None:
        final = _run_pipeline_with_patches(
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
        assert ids.investment_plan is not None and len(ids.investment_plan) > 0

    def test_full_pipeline_risk_round_count(self) -> None:
        final = _run_pipeline_with_patches(
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
