from __future__ import annotations

import pytest


@pytest.fixture
def settings_groq_only(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from finagents.config import Settings

    return Settings(groq_api_key="fake-groq")  # type: ignore[call-arg]


@pytest.fixture
def settings_openrouter_only(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from finagents.config import Settings

    return Settings(
        openrouter_api_key="fake-or",
        groq_api_key=None,
    )  # type: ignore[call-arg]


@pytest.fixture
def settings_all_three(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from finagents.config import Settings

    return Settings(  # type: ignore[call-arg]
        openrouter_api_key="fake-or",
        cerebras_api_key="fake-cb",
        groq_api_key="fake-gq",
    )


class TestProviderPicker:
    def test_lists_providers_with_groq(self, settings_groq_only) -> None:
        from finagents.cli.picker import list_configured_providers
        from finagents.config import resolve_model

        rows = list_configured_providers(settings_groq_only)
        names = [r[0] for r in rows]
        assert "groq" in names
        for name, model in rows:
            assert model == resolve_model(settings_groq_only, name, "deep")

    def test_lists_providers_openrouter_only(self, settings_openrouter_only) -> None:
        from finagents.cli.picker import list_configured_providers
        from finagents.config import resolve_model

        rows = list_configured_providers(settings_openrouter_only)
        assert rows == [
            ("openrouter", resolve_model(settings_openrouter_only, "openrouter", "deep"))
        ]

    def test_skip_when_single_provider(self, settings_openrouter_only) -> None:
        from finagents.cli.picker import should_skip_picker

        assert should_skip_picker(settings_openrouter_only, provider_flag=None) is True

    def test_skip_when_provider_flag(self, settings_groq_only, monkeypatch) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        from finagents.cli.picker import should_skip_picker

        assert should_skip_picker(settings_groq_only, provider_flag="openrouter") is True

    def test_skip_when_env_set(self, settings_groq_only, monkeypatch) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        from finagents.cli.picker import should_skip_picker

        assert should_skip_picker(settings_groq_only, provider_flag=None) is True

    def test_skip_when_non_tty(self, settings_groq_only, monkeypatch) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        from finagents.cli.picker import should_skip_picker

        assert should_skip_picker(settings_groq_only, provider_flag=None) is True

    def test_no_skip_when_ambiguous_tty(self, settings_all_three, monkeypatch) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        from finagents.cli.picker import should_skip_picker

        assert should_skip_picker(settings_all_three, provider_flag=None) is False

    def test_lists_providers_full_order(self, settings_all_three) -> None:
        from finagents.cli.picker import list_configured_providers

        rows = list_configured_providers(settings_all_three)
        names = [r[0] for r in rows]
        assert names == ["openrouter", "cerebras", "groq"]

    def test_lists_providers_shows_distinct_models_per_provider(self, settings_all_three) -> None:
        from finagents.cli.picker import list_configured_providers
        from finagents.config import PROVIDER_DEFAULTS

        rows = list_configured_providers(settings_all_three)
        row_map = dict(rows)
        assert row_map["openrouter"] == PROVIDER_DEFAULTS["openrouter"]["deep"]
        assert row_map["cerebras"] == PROVIDER_DEFAULTS["cerebras"]["deep"]
        assert row_map["groq"] == PROVIDER_DEFAULTS["groq"]["deep"]
        models = {row_map[p] for p in ("openrouter", "cerebras", "groq")}
        assert len(models) == 3, (
            f"Expected provider-specific models, got {row_map!r} — picker "
            "regressed to the pre-Spec-1 'same model per row' bug."
        )

    def test_openrouter_excluded_without_key(self, monkeypatch) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        from finagents.cli.picker import list_configured_providers
        from finagents.config import Settings

        s = Settings(  # type: ignore[call-arg]
            openrouter_api_key=None,
            cerebras_api_key="cb",
            groq_api_key="gq",
        )
        names = [r[0] for r in list_configured_providers(s)]
        assert names == ["cerebras", "groq"]
