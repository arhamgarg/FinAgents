from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("finagents")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


def __getattr__(name: str):
    if name == "FinAgentsGraph":
        from finagents.graph.trading_graph import FinAgentsGraph

        return FinAgentsGraph
    raise AttributeError(f"module 'finagents' has no attribute {name!r}")


__all__ = ["FinAgentsGraph", "__version__"]
