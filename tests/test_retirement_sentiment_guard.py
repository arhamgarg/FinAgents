from __future__ import annotations

from pathlib import Path

import pytest

PRODUCTION_ROOT = Path(__file__).resolve().parent.parent / "src" / "finagents"

RETIRED_SYMBOLS = (
    "SentimentAnalyst",
    "sentiment_analyst_node",
    "SentimentAnalystReport",
    "sentiment_client",
    "sentiment_schema",
    "SentimentReport",
)

RETIRED_SETTINGS_FIELDS = (
    "reddit_client_id",
    "reddit_client_secret",
    "reddit_user_agent",
    "stocktwits_token",
)


def _iter_production_py_files() -> list[Path]:
    return [p for p in PRODUCTION_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


@pytest.mark.parametrize("symbol", RETIRED_SYMBOLS)
def test_retired_sentiment_symbol_absent_from_production(symbol: str) -> None:
    offenders: list[tuple[Path, int, str]] = []
    for path in _iter_production_py_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if symbol in line:
                offenders.append((path.relative_to(PRODUCTION_ROOT.parent), lineno, line.strip()))

    assert not offenders, (
        f"Retired symbol {symbol!r} found in production source — "
        f"retirement has regressed:\n" + "\n".join(f"  {p}:{n}: {line}" for p, n, line in offenders)
    )


@pytest.mark.parametrize("field", RETIRED_SETTINGS_FIELDS)
def test_retired_settings_field_absent_from_config(field: str) -> None:
    from finagents.config import Settings

    assert field not in Settings.model_fields, (
        f"Retired Settings field {field!r} re-appeared in config.py — "
        "data-layer retirement has regressed."
    )


def test_agent_state_has_no_sentiment_report_field() -> None:
    from finagents.agents.utils.agent_states import AgentState

    assert "sentiment_report" not in AgentState.model_fields


def test_analyst_weights_default_sums_to_one_with_three_keys() -> None:
    from finagents.config import Settings

    weights = Settings(openrouter_api_key="fake").analyst_weights
    assert set(weights) == {"technical", "fundamentals", "news"}
    assert abs(sum(weights.values()) - 1.0) < 1e-9
