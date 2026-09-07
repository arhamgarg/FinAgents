from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def streamlit_app_test() -> Any:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    repo_root = Path(__file__).resolve().parents[2]
    app_path = repo_root / "finagents" / "dashboard" / "app.py"
    return AppTest.from_file(str(app_path))


@pytest.fixture
def event_queue() -> queue.Queue[Any]:
    return queue.Queue()
