import pytest

from finagents.agents.utils.agent_states import AgentState, RiskDebateState
from finagents.config import Settings
from finagents.graph.setup import build_full_pipeline, call_risk_debate_subgraph
from finagents.testing.llm_mock import CALL_SITE_SCHEMAS, failures, fixtures
from tests.pipeline import pipeline_scenario


@pytest.mark.parametrize("invest_rounds,risk_rounds", [(1, 1), (3, 2)])
async def test_populated_pipeline_exercises_every_call_site(invest_rounds, risk_rounds):
    with pipeline_scenario() as h:
        graph = build_full_pipeline()
        state = AgentState.model_validate(
            await graph.ainvoke(
                AgentState(ticker="RELIANCE", date="2024-12-31"),
                config={
                    "configurable": {
                        "settings": Settings(
                            max_debate_rounds=invest_rounds, max_risk_discuss_rounds=risk_rounds
                        )
                    }
                },
            )
        )
    assert state.final_trade_decision is not None
    assert state.final_trade_decision.pm_notes
    assert state.final_trade_decision.signal.action in {"BUY", "SELL", "HOLD"}
    assert state.investment_debate_state is not None
    assert len(state.investment_debate_state.bull_turns()) == invest_rounds
    assert len(state.investment_debate_state.bear_turns()) == invest_rounds
    assert state.investment_debate_state.investment_thesis
    assert state.risk_debate_state is not None
    assert len(state.risk_debate_state.risk_turns()) == risk_rounds
    assert state.risk_debate_state.synthesized_position_size_pct == pytest.approx(0.030)
    expected: dict[str, int] = dict.fromkeys(CALL_SITE_SCHEMAS, 1)
    expected.update(bull_researcher=invest_rounds, bear_researcher=invest_rounds)
    expected.update(
        dict.fromkeys(
            ["aggressive_risk", "neutral_risk", "conservative_risk", "risk_debate_consolidation"],
            risk_rounds,
        )
    )
    assert h.calls == expected


@pytest.mark.parametrize(
    "sites,retry",
    [
        (["bull_researcher"], False),
        (["technical_analyst", "fundamentals_analyst", "news_analyst"], False),
        (["trader"], True),
    ],
)
async def test_failure_scenarios_consume_registered_failures(sites, retry):
    with pipeline_scenario() as h:
        for site in sites:
            response = fixtures.HAPPY_PATH_DEFAULTS[site]
            h.set_sequence(
                site, [failures.length_error(), response if retry else failures.length_error()]
            )
        result = await build_full_pipeline().ainvoke(
            AgentState(ticker="RELIANCE", date="2024-12-31"),
            config={
                "configurable": {
                    "settings": Settings(max_debate_rounds=1, max_risk_discuss_rounds=1)
                }
            },
        )
    state = AgentState.model_validate(result)
    assert state.final_trade_decision is not None
    assert all(h.calls[site] == 2 for site in sites)
    if "news_analyst" in sites:
        assert state.news_report is not None
        assert state.fundamentals_report is not None
        assert state.technical_report is not None
        assert state.news_report.confidence == 0
        assert state.fundamentals_report.confidence == 0
        assert state.technical_report.confidence == 0


async def test_risk_recursion_limit_applied(monkeypatch):
    from unittest.mock import AsyncMock
    import finagents.graph.setup as setup

    risk = RiskDebateState(max_rounds=2)
    invoke = AsyncMock(return_value=risk.model_dump())
    monkeypatch.setattr(setup, "_RISK_DEBATE_SUBGRAPH", type("Graph", (), {"ainvoke": invoke})())
    await call_risk_debate_subgraph(AgentState(risk_debate_state=risk))
    assert invoke.call_args.kwargs["config"]["recursion_limit"] == 16
