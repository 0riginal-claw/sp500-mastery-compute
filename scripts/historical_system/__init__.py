"""historical_system - slim vendored shim (gabriel-stub 2026-05-20).

This is the LOCAL VENDORED COPY used by gabriel_indicators_features.py when
the Drive source-of-truth is unavailable (offline / sandbox / cloud runner).

The full upstream __init__.py at version_3 - Gabriel/.../src/historical_system
re-exports SimBrokerConfig / run_backtest / DataLoader / compute_metrics /
etc. at package import time, which forces walks into sub-packages we never
vendored (engine/, data/, metrics/, etc.).

Gabriel only needs historical_system.indicators.REGISTRY - the engine /
data-loader / metrics machinery is irrelevant for the feature wrapper. We
therefore skip the eager re-exports and let submodules import lazily.
"""
from __future__ import annotations
__version__ = "0.1.0-gabriel-stub"
__all__: list[str] = []
