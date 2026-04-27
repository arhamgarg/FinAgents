from __future__ import annotations

from pathlib import Path

import platformdirs

from financialagents.storage.paths import signals_db_path


class TestPathsResolution:
    def test_default_uses_platformdirs(self, clean_env, monkeypatch) -> None:
        monkeypatch.delenv("FA_DATA_DIR", raising=False)
        p = signals_db_path()
        expected = (
            Path(platformdirs.user_data_dir("financialagents", appauthor=False))
            / "signals.db"
        )
        assert p == expected

    def test_env_override(
        self, clean_env, monkeypatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("FA_DATA_DIR", str(tmp_path / "fa"))
        p = signals_db_path()
        assert p == tmp_path / "fa" / "signals.db"

    def test_tilde_expansion(self, clean_env, monkeypatch) -> None:
        monkeypatch.setenv("FA_DATA_DIR", "~/fa-test")
        p = signals_db_path()
        assert p == Path("~/fa-test").expanduser() / "signals.db"
