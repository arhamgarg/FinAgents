from __future__ import annotations


def count_non_none(values: list[object]) -> int:
    return sum(1 for v in values if v is not None)
