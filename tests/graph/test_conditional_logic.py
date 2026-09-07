from __future__ import annotations


class TestInvestDebateRouter:
    def test_loop_continues_when_current_round_less_than_max(self) -> None:
        from finagents.agents.utils.agent_states import InvestDebateState
        from finagents.graph.conditional_logic import invest_debate_router

        for r in (0, 1, 2):
            state = InvestDebateState(current_round=r, max_rounds=3)
            assert invest_debate_router(state) == "bull_researcher_node", (
                f"round={r} should route to bull_researcher_node when max_rounds=3"
            )

    def test_exits_at_boundary(self) -> None:
        from finagents.agents.utils.agent_states import InvestDebateState
        from finagents.graph.conditional_logic import invest_debate_router

        state = InvestDebateState(current_round=3, max_rounds=3)
        assert invest_debate_router(state) == "investment_thesis_node"

    def test_exits_on_overshoot(self) -> None:
        from finagents.agents.utils.agent_states import InvestDebateState
        from finagents.graph.conditional_logic import invest_debate_router

        state = InvestDebateState(current_round=4, max_rounds=3)
        assert invest_debate_router(state) == "investment_thesis_node"

    def test_single_round_case(self) -> None:
        from finagents.agents.utils.agent_states import InvestDebateState
        from finagents.graph.conditional_logic import invest_debate_router

        assert (
            invest_debate_router(InvestDebateState(current_round=0, max_rounds=1))
            == "bull_researcher_node"
        )
        assert (
            invest_debate_router(InvestDebateState(current_round=1, max_rounds=1))
            == "investment_thesis_node"
        )

    def test_router_is_pure(self) -> None:
        from finagents.agents.utils.agent_states import InvestDebateState
        from finagents.graph.conditional_logic import invest_debate_router

        state = InvestDebateState(current_round=1, max_rounds=3)
        first = invest_debate_router(state)
        second = invest_debate_router(state)
        assert first == second == "bull_researcher_node"


class TestRiskDebateRouter:

    def test_loop_continues_when_current_round_less_than_max(self) -> None:
        from finagents.agents.utils.agent_states import RiskDebateState
        from finagents.graph.conditional_logic import risk_debate_router

        for r in (0, 1):
            state = RiskDebateState(current_round=r, max_rounds=2)
            assert risk_debate_router(state) == "aggressive_risk_node"

    def test_exits_at_boundary(self) -> None:
        from finagents.agents.utils.agent_states import RiskDebateState
        from finagents.graph.conditional_logic import risk_debate_router

        state = RiskDebateState(current_round=2, max_rounds=2)
        assert risk_debate_router(state) == "END"

    def test_exits_on_overshoot(self) -> None:
        from finagents.agents.utils.agent_states import RiskDebateState
        from finagents.graph.conditional_logic import risk_debate_router

        state = RiskDebateState(current_round=3, max_rounds=2)
        assert risk_debate_router(state) == "END"

    def test_single_round_case(self) -> None:
        from finagents.agents.utils.agent_states import RiskDebateState
        from finagents.graph.conditional_logic import risk_debate_router

        assert (
            risk_debate_router(RiskDebateState(current_round=0, max_rounds=1))
            == "aggressive_risk_node"
        )
        assert risk_debate_router(RiskDebateState(current_round=1, max_rounds=1)) == "END"

    def test_router_is_pure(self) -> None:
        from finagents.agents.utils.agent_states import RiskDebateState
        from finagents.graph.conditional_logic import risk_debate_router

        state = RiskDebateState(current_round=1, max_rounds=2)
        assert risk_debate_router(state) == risk_debate_router(state)
