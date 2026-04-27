from __future__ import annotations

from pathlib import Path

import platformdirs

from financialagents.config import Settings


def signals_db_path(settings: Settings | None = None) -> Path:
    s = settings or Settings()
    if s.fa_data_dir:
        base = Path(s.fa_data_dir).expanduser()
    else:
        base = Path(
            platformdirs.user_data_dir("financialagents", appauthor=False)
        )
    return base / "signals.db"
