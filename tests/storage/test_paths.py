from __future__ import annotations

from pathlib import Path

import platformdirs

from finagents.storage.paths import signals_db_path


class TestPathsResolution:
    def test_default_uses_platformdirs(self, default_settings, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.delenv("FA_DATA_DIR", raising=False)
        monkeypatch.setattr(
            platformdirs,
            "user_data_dir",
            lambda appname, appauthor: str(tmp_path / appname),
        )
        p = signals_db_path()
        expected = Path(platformdirs.user_data_dir("finagents", appauthor=False)) / "signals.db"
        assert p == expected

    def test_default_migrates_legacy_data_dir(
        self, default_settings, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            platformdirs,
            "user_data_dir",
            lambda appname, appauthor: str(tmp_path / appname),
        )
        legacy_dir = tmp_path / "financialagents"
        legacy_dir.mkdir()
        (legacy_dir / "signals.db").write_bytes(b"signal history")
        (legacy_dir / "signals-backup.db").write_bytes(b"backup")

        path = signals_db_path()

        assert path == tmp_path / "finagents" / "signals.db"
        assert path.read_bytes() == b"signal history"
        assert (path.parent / "signals-backup.db").read_bytes() == b"backup"
        assert not legacy_dir.exists()

    def test_env_override(
        self,
        default_settings,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("FA_DATA_DIR", str(tmp_path / "fa"))
        p = signals_db_path()
        assert p == tmp_path / "fa" / "signals.db"

    def test_tilde_expansion(self, default_settings, monkeypatch) -> None:
        monkeypatch.setenv("FA_DATA_DIR", "~/fa-test")
        p = signals_db_path()
        assert p == Path("~/fa-test").expanduser() / "signals.db"
