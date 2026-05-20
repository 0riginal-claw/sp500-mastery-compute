"""Base class, registry, and compute facade for indicators.

Every indicator subclasses :class:`Indicator` and declares:

* ``name`` — registry key (lowercase, underscore-separated).
* ``outputs`` — tuple of output column names the indicator produces.
* ``params`` — dict of default parameter values.
* ``deps`` — input columns it reads (``close``, ``high``, ``low``, etc.).
* ``compute(df, **params)`` — pure function; returns a ``dict[str, np.ndarray]``
  keyed by output column name, or a single numpy array when there's exactly
  one output.

The registry is a module-level ``dict`` populated by the :func:`register`
decorator at import time. See ``_defs/__init__.py`` for the import fan-out
that ensures every indicator module runs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
REGISTRY: dict[str, type["Indicator"]] = {}


class Indicator:
    """Abstract base — subclasses implement ``compute``.

    Subclasses set class-level attributes ``name``, ``outputs``, ``params``,
    and ``deps``. The metaclass-free registration via ``@register`` keeps
    subclass definitions plain Python.
    """

    name: ClassVar[str] = ""
    outputs: ClassVar[tuple[str, ...]] = ()
    params: ClassVar[dict[str, Any]] = {}
    deps: ClassVar[tuple[str, ...]] = ("close",)
    # Indicators that inherently look across symbols (e.g. A/D line).
    cross_symbol: ClassVar[bool] = False
    # Indicators that return histograms / non-time-series outputs.
    non_timeseries: ClassVar[bool] = False

    def compute(self, df: pd.DataFrame, **params: Any) -> dict[str, np.ndarray] | np.ndarray:
        raise NotImplementedError

    # --- convenience ----------------------------------------------------
    def __call__(self, df: pd.DataFrame, **params: Any) -> dict[str, np.ndarray] | np.ndarray:
        merged = {**self.params, **params}
        return self.compute(df, **merged)

    def params_hash(self, **params: Any) -> str:
        """Stable short hash for caching. Keys sorted, floats preserved."""
        merged = {**self.params, **params}
        blob = json.dumps(merged, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


@dataclass(frozen=True)
class IndicatorSpec:
    """A concrete request: (name, params). Used by strategy manifests."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def resolved(self) -> tuple[Indicator, dict[str, Any]]:
        ind = get(self.name)
        merged = {**ind.params, **self.params}
        return ind, merged

    def cache_key(self) -> str:
        ind = get(self.name)
        return f"{self.name}__{ind.params_hash(**self.params)}"


# ---------------------------------------------------------------------------
# Registration decorator
# ---------------------------------------------------------------------------
def register(cls: type[Indicator]) -> type[Indicator]:
    """Class decorator: add to REGISTRY keyed by ``cls.name``."""
    if not getattr(cls, "name", ""):
        raise ValueError(f"Indicator class {cls.__name__} must set 'name'")
    if cls.name in REGISTRY:
        existing = REGISTRY[cls.name]
        if existing is not cls:
            raise ValueError(
                f"Indicator name collision: '{cls.name}' already registered by "
                f"{existing.__module__}.{existing.__name__}, cannot re-register "
                f"with {cls.__module__}.{cls.__name__}"
            )
    REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> Indicator:
    """Fetch an indicator instance by name."""
    try:
        return REGISTRY[name]()
    except KeyError:
        raise KeyError(f"Unknown indicator: {name!r}. Known: {sorted(REGISTRY)[:10]}…")


def all_names() -> list[str]:
    return sorted(REGISTRY)


def describe(name: str) -> dict[str, Any]:
    cls = REGISTRY[name]
    return {
        "name": cls.name,
        "outputs": list(cls.outputs),
        "params": dict(cls.params),
        "deps": list(cls.deps),
        "cross_symbol": bool(cls.cross_symbol),
        "non_timeseries": bool(cls.non_timeseries),
        "doc": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else "",
    }


def compute(name: str, df: pd.DataFrame, **params: Any) -> dict[str, np.ndarray] | np.ndarray:
    """Convenience: ``compute('rsi', df, length=14)``."""
    ind = get(name)
    return ind(df, **params)


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
