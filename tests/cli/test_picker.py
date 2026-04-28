from __future__ import annotations

import sys

import pytest


@pytest.fixture
def settings_groq_only(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from financialagents.config import Settings

    return Settings(groq_api_key="fake-groq", _env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def settings_no_groq(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from financialagents.config import Settings

    return Settings(groq_api_key=None, _env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def settings_all_four(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from financialagents.config import Settings

    return Settings(  # type: ignore[call-arg]
        openrouter_api_key="fake-or",
        cerebras_api_key="fake-cb",
        groq_api_key="fake-gq",
        _env_file=None,
    )


class TestProviderPicker:
    def test_lists_providers_with_groq(self, settings_groq_only) -> None:
        from financialagents.cli.picker import list_configured_providers

        rows = list_configured_providers(settings_groq_only)
        names = [r[0] for r in rows]
        assert "groq" in names
        assert "ollama" in names
        for _name, model in rows:
            assert model == settings_groq_only.deep_think_llm

    def test_lists_providers_ollama_only(self, settings_no_groq) -> None:
        from financialagents.cli.picker import list_configured_providers

        rows = list_configured_providers(settings_no_groq)
        assert rows == [("ollama", settings_no_groq.deep_think_llm)]

    def test_skip_when_single_provider(self, settings_no_groq) -> None:
        from financialagents.cli.picker import should_skip_picker

        assert should_skip_picker(settings_no_groq, provider_flag=None) is True

    def test_skip_when_provider_flag(self, settings_groq_only, monkeypatch) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        from financialagents.cli.picker import should_skip_picker

        assert should_skip_picker(settings_groq_only, provider_flag="ollama") is True

    def test_skip_when_env_set(self, settings_groq_only, monkeypatch) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        from financialagents.cli.picker import should_skip_picker

        assert should_skip_picker(settings_groq_only, provider_flag=None) is True

    def test_skip_when_non_tty(self, settings_groq_only, monkeypatch) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        from financialagents.cli.picker import should_skip_picker

        assert should_skip_picker(settings_groq_only, provider_flag=None) is True

    def test_no_skip_when_ambiguous_tty(self, settings_groq_only, monkeypatch) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        from financialagents.cli.picker import should_skip_picker

        assert should_skip_picker(settings_groq_only, provider_flag=None) is False

    def test_lists_providers_full_order(self, settings_all_four) -> None:
        from financialagents.cli.picker import list_configured_providers

        rows = list_configured_providers(settings_all_four)
        names = [r[0] for r in rows]
        assert names == ["openrouter", "cerebras", "groq", "ollama"]

    def test_openrouter_excluded_without_key(self, monkeypatch) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        from financialagents.cli.picker import list_configured_providers
        from financialagents.config import Settings

        s = Settings(  # type: ignore[call-arg]
            openrouter_api_key=None,
            cerebras_api_key="cb",
            groq_api_key="gq",
            _env_file=None,
        )
        names = [r[0] for r in list_configured_providers(s)]
        assert names == ["cerebras", "groq", "ollama"]
