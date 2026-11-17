from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from finagents.config import Settings


def run_settings(config: RunnableConfig | None) -> Settings:
    configurable = config.get("configurable", {}) if config else {}
    settings = configurable.get("settings")
    if not isinstance(settings, Settings):
        raise TypeError(
            "RunnableConfig['configurable']['settings'] must be a Settings instance, "
            f"got {type(settings).__name__}"
        )
    return settings
