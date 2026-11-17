from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finagents.graph.trading_graph import FinAgentsGraph

try:
    __version__ = _dist_version("finagents")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


def __getattr__(name: str) -> type[FinAgentsGraph]:
    if name == "FinAgentsGraph":
        from finagents.graph.trading_graph import FinAgentsGraph

        return FinAgentsGraph
    raise AttributeError(f"module 'finagents' has no attribute {name!r}")


__all__ = ["FinAgentsGraph", "__version__"]
