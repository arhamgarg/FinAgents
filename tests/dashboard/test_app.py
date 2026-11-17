from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip(
    "streamlit.testing.v1",
    reason="Streamlit AppTest required",
).AppTest


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = REPO_ROOT / "src" / "finagents" / "dashboard" / "app.py"


class TestDashboardApp:
    def test_navigation(self) -> None:
        at = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        assert not at.exception, f"AppTest raised: {[str(e.value) for e in at.exception]}"
