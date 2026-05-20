"""Stub for historical_system.engine — created 2026-05-20 by autonomous_mode.

Provides SimBrokerConfig so gabriel_indicators_features.py imports succeed.
"""

from dataclasses import dataclass, field
from typing import Any

@dataclass
class SimBrokerConfig:
    """Stub SimBrokerConfig — replaces missing historical_system.engine import."""
    data_dir: str = ""
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    initial_capital: float = 100000.0
    commission_pct: float = 0.001
    slippage_bps: float = 5.0
    max_positions: int = 10
    universe: list = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, d: dict) -> "SimBrokerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
