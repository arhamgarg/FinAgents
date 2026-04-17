from __future__ import annotations


class TestMergeByKey:
    def test_empty_left_returns_right_copy(self) -> None:
        from financialagents.graph.reducers import merge_by_key
        result = merge_by_key({}, {1: "a"})
        assert result == {1: "a"}

    def test_empty_right_returns_left_copy(self) -> None:
        from financialagents.graph.reducers import merge_by_key
        result = merge_by_key({1: "a"}, {})
        assert result == {1: "a"}

    def test_both_empty(self) -> None:
        from financialagents.graph.reducers import merge_by_key
        assert merge_by_key({}, {}) == {}

    def test_disjoint_keys_merge(self) -> None:
        from financialagents.graph.reducers import merge_by_key
        result = merge_by_key({1: "a"}, {2: "b"})
        assert result == {1: "a", 2: "b"}

    def test_right_wins_on_collision(self) -> None:
        from financialagents.graph.reducers import merge_by_key
        result = merge_by_key({1: "first-attempt"}, {1: "retry-attempt"})
        assert result == {1: "retry-attempt"}

    def test_partial_collision(self) -> None:
        from financialagents.graph.reducers import merge_by_key
        result = merge_by_key({1: "a", 2: "b"}, {2: "B", 3: "c"})
        assert result == {1: "a", 2: "B", 3: "c"}

    def test_result_is_new_dict_not_alias(self) -> None:
        from financialagents.graph.reducers import merge_by_key
        left = {1: "a"}
        right = {2: "b"}
        result = merge_by_key(left, right)
        result[99] = "z"
        assert 99 not in left
        assert 99 not in right


class TestMergeByKeyRetryIdempotencyProperty:

    def test_same_round_key_retry_replaces(self) -> None:
        from financialagents.graph.reducers import merge_by_key
        state_after_first = merge_by_key({}, {1: "bull-round-1-attempt-1"})
        assert state_after_first == {1: "bull-round-1-attempt-1"}
        state_after_retry = merge_by_key(state_after_first, {1: "bull-round-1-attempt-2"})
        assert len(state_after_retry) == 1
        assert state_after_retry == {1: "bull-round-1-attempt-2"}

    def test_successive_rounds_accumulate(self) -> None:
        from financialagents.graph.reducers import merge_by_key
        state: dict[int, str] = {}
        for round_num in (1, 2, 3):
            state = merge_by_key(state, {round_num: f"bull-round-{round_num}"})
        assert len(state) == 3
        assert state == {
            1: "bull-round-1",
            2: "bull-round-2",
            3: "bull-round-3",
        }
