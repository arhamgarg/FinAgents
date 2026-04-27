from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestThemeConfig:
    def test_theme_toml_present_with_palette(self) -> None:
        config_path = REPO_ROOT / ".streamlit" / "config.toml"
        assert config_path.exists(), (
            f".streamlit/config.toml missing at {config_path}"
        )
        text = config_path.read_text(encoding="utf-8")
        assert 'base = "dark"' in text, "missing dark base directive"
        assert "#0F1419" in text, "missing backgroundColor token"
        assert "#1B2129" in text, "missing secondaryBackgroundColor token"
        assert "#E6EDF3" in text, "missing textColor token"
        assert "#30363D" in text, "missing borderColor token"
