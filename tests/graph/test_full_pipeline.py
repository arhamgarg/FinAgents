from finagents import FinAgentsGraph
from finagents.config import Settings
from tests.support.pipeline import pipeline_scenario


async def test_full_pipeline_flag():
    with pipeline_scenario():
        state = await FinAgentsGraph(full_pipeline=True).apropagate(
            "RELIANCE", "2024-12-31", Settings(max_debate_rounds=3, max_risk_discuss_rounds=1)
        )
    assert state.final_trade_decision is not None
    assert state.investment_debate_state is not None
    assert state.investment_debate_state.investment_thesis
    assert len(state.investment_debate_state.bull_turns()) == 3
    assert len(state.investment_debate_state.bear_turns()) == 3


async def test_analysis_mode_stops_after_research():
    with pipeline_scenario() as h:
        state = await FinAgentsGraph().apropagate("RELIANCE", "2024-12-31")
    assert state.research_summary is not None
    assert state.final_trade_decision is None
    assert set(h.calls) == {
        "technical_analyst",
        "fundamentals_analyst",
        "news_analyst",
        "research_manager",
    }
