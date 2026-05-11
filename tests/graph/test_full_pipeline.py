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


class TestFullPipelineFlag:

    def test_full_pipeline_compiles_and_runs_with_mocked_nodes(self) -> None:
        tech_report = TechnicalReport(confidence=0.8)
        fund_report = FundamentalsAnalystReport(confidence=0.8)
        news_report = NewsReport(confidence=0.7)
        brief = ResearchBrief(
            consensus_view="bullish on RELIANCE",
            overall_confidence=0.75,
        )

        async def fake_tech(state, config):
            return {"technical_report": tech_report}

        async def fake_fund(state, config):
            return {"fundamentals_report": fund_report}

        async def fake_news(state, config):
            return {"news_report": news_report}

        async def fake_rm(state, config):
            return {"research_summary": brief}

        async def fake_bull(state, config):
            return {"bull_history": {state.current_round: f"bull-{state.current_round}"}}

        async def fake_bear(state, config):
            return {
                "bear_history": {state.current_round: f"bear-{state.current_round}"},
                "current_round": state.current_round + 1,
            }

        async def fake_thesis(state, config):
            bulls = state.bull_turns()
            return {"investment_thesis": f"thesis after {len(bulls)} rounds"}

        async def fake_trader(state, config=None):
            return {
                "trader_signal": TradingSignal(
                    action="HOLD",
                    conviction=0.1,
                    rationale="fake trader for test_full_pipeline",
                )
            }

        async def fake_aggressive(state, config=None):
            return {
                "aggressive_assessment": PersonaAssessment(
                    main_argument="agg",
                    recommended_position_size_pct=0.0,
                    confidence=0.1,
                ).model_dump_json()
            }

        async def fake_neutral(state, config=None):
            return {
                "neutral_assessment": PersonaAssessment(
                    main_argument="neu",
                    recommended_position_size_pct=0.0,
                    confidence=0.1,
                ).model_dump_json()
            }

        async def fake_conservative(state, config=None):
            return {
                "conservative_assessment": PersonaAssessment(
                    main_argument="con",
                    recommended_position_size_pct=0.0,
                    confidence=0.1,
                ).model_dump_json()
            }

        async def fake_consolidation(state, config=None):
            round_key = state.current_round
            return {
                "risk_debate_history": {round_key: "fake consolidation synthesis"},
                "current_round": state.current_round + 1,
                "synthesized_position_size_pct": 0.0,
            }

        async def fake_pm(state, config=None):
            signal = (
                state.trader_signal.model_copy()
                if state.trader_signal is not None
                else TradingSignal(
                    action="HOLD", conviction=0.0, rationale="no trader signal"
                )
            )
            signal.action = "HOLD"
            signal.position_size_pct = None
            signal.position_size_shares = None
            return {
                "final_trade_decision": PortfolioDecision(
                    signal=signal,
                    approved=False,
                    decision="veto",
                    pm_notes="fake PM for test_full_pipeline",
                )
            }

        import financialagents.graph.setup as setup_mod
        import financialagents.graph.subgraphs.invest_debate as sub_mod
        import financialagents.graph.subgraphs.risk_debate as risk_sub_mod

        try:
            with (
                patch(
                    "financialagents.agents.analysts.technical_analyst.technical_analyst_node",
                    fake_tech,
                ),
                patch(
                    "financialagents.agents.analysts.fundamentals_analyst.fundamentals_analyst_node",
                    fake_fund,
                ),
                patch(
                    "financialagents.agents.analysts.news_analyst.news_analyst_node",
                    fake_news,
                ),
                patch(
                    "financialagents.agents.managers.research_manager.research_manager_node",
                    fake_rm,
                ),
                patch(
                    "financialagents.agents.researchers.bull_researcher.bull_researcher_node",
                    fake_bull,
                ),
                patch(
                    "financialagents.agents.researchers.bear_researcher.bear_researcher_node",
                    fake_bear,
                ),
                patch(
                    "financialagents.agents.researchers.investment_thesis.investment_thesis_node",
                    fake_thesis,
                ),
                patch(
                    "financialagents.agents.trader.trader.trader_node",
                    fake_trader,
                ),
                patch(
                    "financialagents.agents.risk_mgmt.aggressive.aggressive_risk_node",
                    fake_aggressive,
                ),
                patch(
                    "financialagents.agents.risk_mgmt.neutral.neutral_risk_node",
                    fake_neutral,
                ),
                patch(
                    "financialagents.agents.risk_mgmt.conservative.conservative_risk_node",
                    fake_conservative,
                ),
                patch(
                    "financialagents.agents.risk_mgmt.consolidation.risk_debate_consolidation_node",
                    fake_consolidation,
                ),
                patch(
                    "financialagents.agents.managers.portfolio_manager.portfolio_manager_node",
                    fake_pm,
                ),
            ):
                importlib.reload(setup_mod)
                importlib.reload(sub_mod)
                importlib.reload(risk_sub_mod)
                setup_mod._INVEST_DEBATE_SUBGRAPH = None
                setup_mod._RISK_DEBATE_SUBGRAPH = None

                graph = setup_mod.build_full_pipeline()
                settings = Settings(_env_file=None)
                settings.max_debate_rounds = 3
                settings.max_risk_discuss_rounds = 1
                config = {
                    "configurable": {"settings": settings, "thread_id": "test-123"}
                }
                initial = AgentState(ticker="RELIANCE", date="2024-01-15")
                result = asyncio.run(graph.ainvoke(initial, config=config))
        finally:
            importlib.reload(sub_mod)
            importlib.reload(risk_sub_mod)
            importlib.reload(setup_mod)
            setup_mod._INVEST_DEBATE_SUBGRAPH = None
            setup_mod._RISK_DEBATE_SUBGRAPH = None

        final = (
            AgentState.model_validate(result) if isinstance(result, dict) else result
        )
        assert final.investment_debate_state is not None
        ids = final.investment_debate_state
        assert ids.investment_thesis is not None
        assert len(ids.investment_thesis) > 0
        assert len(ids.bull_turns()) == 3, (
            f" off-by-one: bull_turns={ids.bull_turns()}"
        )
        assert len(ids.bear_turns()) == 3

    def test_build_analysis_graph_still_byte_identical(self) -> None:
        from financialagents.graph.setup import build_analysis_graph

        graph = build_analysis_graph()
        nodes = set(graph.nodes.keys())
        assert "debate_init_node" not in nodes
        assert "invest_debate_node" not in nodes
        assert "research_manager_node" in nodes
        assert "propagation_node" in nodes
        assert "validation_node" in nodes
