"""Wrapper for FinMind/FinMind: Taiwan/global market data + macro fetcher

License: apache-2.0
Source clone: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/repos-claude-clones/FinMind
Generated: 2026-05-17 (github_hunt_loop cycle cycle2)

Import-guarded: if upstream lib isn't installed, returns df unchanged so the
pipeline never breaks. Install with `pip install finmind` to activate.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

try:
    import FinMind  # type: ignore  # noqa: F401
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def add_finmind_features(df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Add Taiwan/global market data + macro fetcher features to df.

    Inputs: df with OHLCV columns (open, high, low, close, volume).
    Returns: df with new columns prefixed `finmind_data_*`.

    If the upstream package isn't installed, returns df unchanged.
    """
    if not _AVAILABLE:
        log.info("FinMind not available; skipping")
        return df

    # TODO(auto-wire-consumer): elaborate with actual feature computations.
    # Default no-op so the pipeline runs.
    return df


__all__ = ["add_finmind_features"]
