"""Altimis/Scweet feature wrapper.

Source: https://github.com/Altimis/Scweet
License: MIT
Discovered: 2026-05-17 (longtail_01)

Causal-safe: producer emits past-only values; consumer applies .shift(1).
Graceful degradation: returns df with zero-filled feature columns on any error.

NOTE: This is a STUB wrapper. The repo is cloned and import path is wired,
but the feature extraction logic must be fleshed out in follow-up (human review).
The stub returns zero-filled features so downstream consumers stay stable.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

_CLONE_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/AI-Tools/repos-claude-clones/Scweet"
)
if _CLONE_ROOT.exists() and str(_CLONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLONE_ROOT))

FEATURES = [
    "scweet_signal_a",
    "scweet_signal_b",
]


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for c in FEATURES:
        df[c] = 0.0
    return df


def add_scweet_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add Scweet features.

    Inputs:
        df: must contain at minimum 'date','open','high','low','close','volume'
        ticker: ticker symbol (string)

    Output:
        df with FEATURES columns appended. Always returns df (graceful fail).
    """
    df = df.copy()
    try:
        # TODO: import the repo's primary feature function and call here.
        return _zero_fill(df)
    except Exception as e:  # pragma: no cover
        LOG.warning("add_scweet_features failed for %s: %s", ticker, e)
        return _zero_fill(df)
