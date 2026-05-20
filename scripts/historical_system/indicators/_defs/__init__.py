"""Import every indicator definition so its @register decorator runs.

Adding a new indicator is three steps:
  1. Create ``_defs/<your_indicator>.py`` with a class decorated by @register.
  2. Add ``from historical_system.indicators._defs import <your_indicator>``
     here (or rely on the glob below).
  3. That's it — the registry now knows about it.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

_HERE = Path(__file__).parent

# Discover every .py sibling (excluding __init__) and import it.
# This keeps the file list self-maintaining — adding _defs/foo.py is enough.
for mod_info in pkgutil.iter_modules([str(_HERE)]):
    if mod_info.name.startswith("_"):
        continue
    importlib.import_module(f"{__name__}.{mod_info.name}")
