"""historical_system.indicators — registry + 104 indicator definitions.

Public API
----------
* :class:`Indicator` — base class each indicator subclasses.
* :func:`register` — decorator that adds an Indicator to the global registry.
* :func:`get` / :func:`all_names` / :func:`describe` — registry lookups.
* :func:`compute` — convenience for "compute indicator X on df with params Y".

Design notes
------------
Indicators are *pure* functions of (df, params). They don't know about
caching or backtesting — the cache layer in ``historical_system.data`` calls
them and stores the results.

Each indicator is defined in its own file under ``_defs/`` and imported here
to register itself. This keeps file diffs small when a single indicator is
tweaked, and makes it obvious which indicators exist (one ``ls _defs/``).

Formulas match TradingView/Alpaca chart output wherever there's ambiguity.
Where TradingView and textbook formulas differ, the docstring on the
indicator calls out which convention was chosen and why.
"""
from __future__ import annotations

from historical_system.indicators.base import (
    Indicator,
    IndicatorSpec,
    register,
    get,
    all_names,
    describe,
    compute,
    REGISTRY,
)

# Import every indicator definition so its @register decorator runs.
# The _defs package's __init__ does the bulk import.
from historical_system.indicators import _defs  # noqa: F401

__all__ = [
    "Indicator",
    "IndicatorSpec",
    "register",
    "get",
    "all_names",
    "describe",
    "compute",
    "REGISTRY",
]
