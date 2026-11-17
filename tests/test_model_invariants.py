from __future__ import annotations

import importlib
import pkgutil

from pydantic import BaseModel

import finagents

STREAMLIT_SCRIPTS = ("finagents.dashboard.app", "finagents.dashboard.pages.")
MUTABLE_MODELS = {
    "finagents.agents.utils.agent_states.AgentState": "LangGraph graph state",
    "finagents.agents.utils.agent_states.InvestDebateState": "LangGraph subgraph state",
    "finagents.agents.utils.agent_states.RiskDebateState": "LangGraph subgraph state",
    "finagents.testing.llm_mock._harness._HarnessChatModel": "LangChain chat model",
}


def _finagents_models() -> dict[str, type[BaseModel]]:
    for module in pkgutil.walk_packages(finagents.__path__, "finagents."):
        if not module.name.startswith(STREAMLIT_SCRIPTS):
            importlib.import_module(module.name)
    models: dict[str, type[BaseModel]] = {}
    pending: list[type[BaseModel]] = [BaseModel]
    while pending:
        for subclass in pending.pop().__subclasses__():
            pending.append(subclass)
            if (
                subclass.__module__.partition(".")[0] == "finagents"
                and subclass.__pydantic_generic_metadata__["origin"] is None
            ):
                models[f"{subclass.__module__}.{subclass.__qualname__}"] = subclass
    return models


def test_models_are_frozen_unless_listed() -> None:
    models = _finagents_models()
    mutable = sorted(
        name
        for name, model in models.items()
        if not model.model_config.get("frozen") and name not in MUTABLE_MODELS
    )
    assert mutable == []
    assert sorted(set(MUTABLE_MODELS) - set(models)) == []


def test_settings_ignore_unknown_env_keys() -> None:
    from finagents.config import Settings

    assert Settings.model_config.get("extra") == "ignore"
    assert Settings.model_config.get("frozen") is True
