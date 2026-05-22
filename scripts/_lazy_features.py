"""
Lazy feature-module shim for backtest_xgb_v10.

Refactor 2026-05-21: replaces ~56 inline `try: from X import Y / except: Y = None`
blocks at the top of backtest_xgb_v10.py.  Those blocks force ALL feature modules
to be parsed + imported at script import time, even when a given backtest only
calls a handful of them — this dominated the ~5s cold-start.

Usage (drop-in replacement):
  Old:
      try:
          from foo_features import compute_foo
      except ImportError:
          compute_foo = None
      ...
      if compute_foo is not None:
          x = compute_foo(df)

  New (in backtest_xgb_v10.py):
      from _lazy_features import compute_foo  # name resolved lazily on first attr access
      ...
      if compute_foo is not None:
          x = compute_foo(df)

Mechanism (PEP 562 module-level __getattr__):
  - On `from _lazy_features import compute_foo`, Python invokes
    __getattr__('compute_foo'); we infer the source module
    ('foo_features') from the conventional naming and try to import it.
  - If the import succeeds, we return the function and cache it on the module.
  - If the import fails (ImportError, ModuleNotFoundError, or any other
    exception during import — preserves the old try-except semantics),
    we return None and cache that too.

The naming convention used by every dir-glob auto-wired loader in v10 is
``compute_<modname>`` exported from module ``<modname>``.  This file encodes
that convention; any caller that does NOT follow it must add an explicit
entry to _OVERRIDES below.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Cache: name -> resolved object (callable or None).  None means "we tried and
# the import failed", so we don't retry repeatedly.
_RESOLVED: Dict[str, Any] = {}

# Override map for names that don't follow the `compute_<modname>` from
# `<modname>` convention.  Most simple try-import blocks DO follow it, so this
# map starts empty; populate as needed when refactoring more blocks.
#
# Format: {attr_name: (module_name, symbol_name_in_module)}
_OVERRIDES: Dict[str, tuple[str, str]] = {}


def _infer_module(name: str) -> Optional[tuple[str, str]]:
    """Infer (module_name, symbol_in_module) from the requested attribute name.

    Convention used throughout v10:
      attr "compute_foo_bar_features"  ->  module "foo_bar_features", symbol same as attr

    Returns None if the name doesn't match the convention and isn't overridden.
    """
    if name in _OVERRIDES:
        return _OVERRIDES[name]
    if name.startswith("compute_"):
        modname = name[len("compute_"):]
        # The symbol-in-module is identical to the attr name.
        return (modname, name)
    return None


def __getattr__(name: str) -> Any:
    """PEP 562 module-level __getattr__: lazy import on first access.

    Preserves the original ``try: from M import S / except ImportError: S = None``
    semantics: any failure during import yields None (cached).  We catch the
    broad Exception (not just ImportError) because some feature modules raise
    AttributeError / RuntimeError at import time when their own optional deps
    are missing, and the legacy code treated those identically to ImportError.
    """
    # Hit the cache first.
    if name in _RESOLVED:
        return _RESOLVED[name]

    # Sentinels that Python (or static tooling) may probe for.
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)

    inferred = _infer_module(name)
    if inferred is None:
        # Not a known/recognised compute_ shim; fall through to AttributeError
        # so callers see a real failure (rather than silently getting None for
        # a typo).
        raise AttributeError(f"_lazy_features has no attribute {name!r}")

    modname, symname = inferred
    try:
        mod = importlib.import_module(modname)
        obj = getattr(mod, symname, None)
        _RESOLVED[name] = obj
        return obj
    except Exception as e:  # noqa: BLE001 — match legacy try-except-ImportError surface
        logger.debug("_lazy_features: failed to import %s from %s: %s", symname, modname, e)
        _RESOLVED[name] = None
        return None
