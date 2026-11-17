from __future__ import annotations

from pathlib import Path

import platformdirs

from finagents.config import Settings

_APP_NAME = "finagents"
_LEGACY_APP_NAME = "financialagents"


def _migrate_legacy_data_dir(base: Path) -> None:
    legacy_base = Path(platformdirs.user_data_dir(_LEGACY_APP_NAME, appauthor=False))
    if not base.exists() and legacy_base.exists():
        legacy_base.rename(base)


def signals_db_path(settings: Settings) -> Path:
    if settings.fa_data_dir:
        base = Path(settings.fa_data_dir).expanduser()
    else:
        base = Path(platformdirs.user_data_dir(_APP_NAME, appauthor=False))
        _migrate_legacy_data_dir(base)
    return base / "signals.db"
