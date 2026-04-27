from __future__ import annotations

import importlib
import re
import sys
from importlib.metadata import PackageNotFoundError

import pytest


class TestPublicAPI:
    def test_re_exports(self) -> None:
        import financialagents

        assert "FinancialAgentsGraph" in financialagents.__all__
        assert "__version__" in financialagents.__all__
        from financialagents import FinancialAgentsGraph

        assert FinancialAgentsGraph is not None

    def test_version_string(self) -> None:
        import financialagents

        assert isinstance(financialagents.__version__, str)
        assert re.match(r"^\d+\.\d+\.\d+", financialagents.__version__), (
            f"version does not match X.Y.Z format: {financialagents.__version__!r}"
        )

    def test_version_falls_back_when_metadata_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib.metadata as _m

        def raise_not_found(name: str):
            raise PackageNotFoundError(name)

        monkeypatch.setattr(_m, "version", raise_not_found)
        for mod in list(sys.modules):
            if mod == "financialagents" or mod.startswith("financialagents."):
                del sys.modules[mod]
        financialagents = importlib.import_module("financialagents")
        assert financialagents.__version__ == "0.0.0+unknown"

    def test_no_heavy_top_level_imports(self) -> None:
        for mod in list(sys.modules):
            if mod.startswith(("streamlit", "rich.live", "aiosqlite")):
                del sys.modules[mod]
        for mod in ("financialagents",):
            if mod in sys.modules:
                del sys.modules[mod]
        importlib.import_module("financialagents")
        assert "streamlit" not in sys.modules
        assert "rich.live" not in sys.modules
        assert "aiosqlite" not in sys.modules
