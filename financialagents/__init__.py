from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("financialagents")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


def __getattr__(name: str):
    if name == "FinancialAgentsGraph":
        from financialagents.graph.trading_graph import FinancialAgentsGraph

        return FinancialAgentsGraph
    raise AttributeError(f"module 'financialagents' has no attribute {name!r}")


__all__ = ["FinancialAgentsGraph", "__version__"]
