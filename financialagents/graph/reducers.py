from __future__ import annotations


def merge_by_key(left: dict[int, str], right: dict[int, str]) -> dict[int, str]:
    if not left:
        return dict(right)
    if not right:
        return dict(left)
    return {**left, **right}
