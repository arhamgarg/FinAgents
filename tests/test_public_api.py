from __future__ import annotations

import re
import subprocess
import sys


class TestPublicAPI:
    def test_re_exports(self) -> None:
        import finagents

        assert "FinAgentsGraph" in finagents.__all__
        assert "__version__" in finagents.__all__
        from finagents import FinAgentsGraph

        assert FinAgentsGraph is not None

    def test_version_string(self) -> None:
        import finagents

        assert isinstance(finagents.__version__, str)
        assert re.match(r"^\d+\.\d+\.\d+", finagents.__version__), (
            f"version does not match X.Y.Z format: {finagents.__version__!r}"
        )

    def test_version_falls_back_when_metadata_missing(self) -> None:
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch
with patch('importlib.metadata.version', side_effect=PackageNotFoundError('finagents')):
    import finagents
    assert finagents.__version__ == '0.0.0+unknown'
""",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_no_heavy_top_level_imports(self) -> None:
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
import finagents
assert not {'streamlit', 'rich.live', 'aiosqlite'} & sys.modules.keys()
""",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
