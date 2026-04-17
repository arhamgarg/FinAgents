from __future__ import annotations


class TestInvestDebateRouter:
    def test_loop_continues_when_current_round_less_than_max(self) -> None:
        from financialagents.agents.utils.agent_states import InvestDebateState
        from financialagents.graph.conditional_logic import invest_debate_router
        for r in (0, 1, 2):
            state = InvestDebateState(current_round=r, max_rounds=3)
            assert invest_debate_router(state) == "bull_researcher_node", (
                f"round={r} should route to bull_researcher_node when max_rounds=3"
            )

    def test_exits_at_boundary(self) -> None:
        from financialagents.agents.utils.agent_states import InvestDebateState
        from financialagents.graph.conditional_logic import invest_debate_router
        state = InvestDebateState(current_round=3, max_rounds=3)
        assert invest_debate_router(state) == "investment_plan_node"

    def test_exits_on_overshoot(self) -> None:
        from financialagents.agents.utils.agent_states import InvestDebateState
        from financialagents.graph.conditional_logic import invest_debate_router
        state = InvestDebateState(current_round=4, max_rounds=3)
        assert invest_debate_router(state) == "investment_plan_node"

    def test_single_round_case(self) -> None:
        from financialagents.agents.utils.agent_states import InvestDebateState
        from financialagents.graph.conditional_logic import invest_debate_router
        assert invest_debate_router(InvestDebateState(current_round=0, max_rounds=1)) == "bull_researcher_node"
        assert invest_debate_router(InvestDebateState(current_round=1, max_rounds=1)) == "investment_plan_node"

    def test_router_is_pure(self) -> None:
        from financialagents.agents.utils.agent_states import InvestDebateState
        from financialagents.graph.conditional_logic import invest_debate_router
        state = InvestDebateState(current_round=1, max_rounds=3)
        first = invest_debate_router(state)
        second = invest_debate_router(state)
        assert first == second == "bull_researcher_node"
