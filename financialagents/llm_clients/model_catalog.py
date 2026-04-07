from __future__ import annotations

GROQ_MODELS: dict[str, str] = {
    "deep_think": "llama-3.3-70b-versatile",
    "quick_think": "llama-3.1-8b-instant",
}

OLLAMA_MODELS: dict[str, str] = {
    "deep_think": "gemma4:latest",
    "quick_think": "gemma4:latest",
}

_PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "groq": GROQ_MODELS,
    "ollama": OLLAMA_MODELS,
}


def get_default_model(provider: str, tier: str) -> str:
    if provider not in _PROVIDER_MODELS:
        raise ValueError(
            f"Unknown provider '{provider}'. Valid: {list(_PROVIDER_MODELS.keys())}"
        )
    models = _PROVIDER_MODELS[provider]
    if tier not in models:
        raise ValueError(f"Unknown tier '{tier}'. Valid: {list(models.keys())}")
    return models[tier]
