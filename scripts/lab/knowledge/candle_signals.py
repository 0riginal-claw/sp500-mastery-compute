"""
Candle structure intraday signals — 14 signals + 10-question pre-trade checklist.

Distilled from `Tech0/Data Master/BackTests & Data/_candle_structure_extract/candle_signals.md`
(itself extracted from `Candle Structure in Trading.pdf`).

Top funcs:
  all_signals()                 — list of 14 candle-structure signal records
  signal(name)                  — single signal record by name
  pretrade_checklist()          — list of 10 questions to ask before any long trade
  confirmation_layer_order()    — the recommended ordering of confluences
  bullish_volume_criteria()     — what "strong bullish volume" looks like
  bearish_volume_criteria()     — what "strong bearish volume" looks like
  timeframe_roles()             — the 1m → 5m → 15m → 1h → 1d role mapping
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


_SIGNALS: List[Dict[str, Any]] = [
    {
        "number": 4, "name": "Candle structure / candle quality",
        "track": [
            "Big body candle = strong directional pressure",
            "Small body candle = indecision",
            "Long upper wick = buyers pushed up but sellers rejected",
            "Long lower wick = sellers pushed down but buyers absorbed",
            "Close near high = bullish strength",
            "Close near low = bearish strength",
            "Doji / tiny candle = pause, uncertainty, possible reversal zone",
            "Engulfing candle = possible shift in control",
            "Inside candle = compression before expansion",
            "Outside candle = expansion / volatility spike",
        ],
        "most_important_rule": "Where the candle closes matters more than where the wick went."
    },
    {
        "number": 5, "name": "Wick behavior",
        "look_for": [
            "Lower wick at support = buyers defended",
            "Upper wick at resistance = sellers defended",
            "Repeated lower wicks near same level = possible accumulation",
            "Repeated upper wicks near same level = possible distribution",
        ]
    },
    {
        "number": 6, "name": "Candle close confirmation",
        "track": [
            "Breaks resistance and closes above = stronger",
            "Breaks resistance but closes below = possible fakeout",
            "Breaks VWAP and holds above = stronger",
            "Breaks VWAP and immediately loses it = weak",
        ],
        "note": "For intraday trading, this one thing alone can prevent a lot of fake breakout trades."
    },
    {
        "number": 7, "name": "Consecutive candle behavior",
        "bullish_sequence": [
            "Higher lows", "Closing near highs", "Smaller pullback candles",
            "Green bodies bigger than red bodies", "Respects VWAP / support"
        ],
        "bearish_sequence": [
            "Lower highs", "Closing near lows", "Weak bounce candles",
            "Red bodies bigger than green bodies", "Keeps rejecting VWAP / resistance"
        ]
    },
    {
        "number": 8, "name": "Range expansion vs contraction",
        "definitions": {
            "expansion": "candles getting bigger, momentum increasing",
            "contraction": "candles getting smaller, volatility tightening"
        },
        "useful_idea": "Contraction often comes before expansion."
    },
    {
        "number": 9, "name": "Opening range",
        "track": [
            "Opening range high", "Opening range low",
            "Break above / below", "Fake breakout", "Retest of opening range"
        ],
        "good_window_minutes": [5, 15, 30]
    },
    {
        "number": 10, "name": "Liquidity grabs / stop hunts",
        "look_for": [
            "Price breaks previous high then closes back below",
            "Price breaks previous low then closes back above",
            "Large wick through obvious S/R",
            "Volume spike with no follow-through"
        ],
        "why": "Many traders place stops under obvious lows and above obvious highs."
    },
    {
        "number": 11, "name": "Volume + candle relationship",
        "strong_bullish": ["High volume", "Big green body", "Close near high",
                            "Breaks important level", "Holds the level after"],
        "weak_bullish": ["High volume", "Long upper wick",
                          "Fails to close above resistance", "Next candle sells off"],
        "strong_bearish": ["High volume", "Big red body", "Close near low",
                            "Breaks support or loses VWAP"],
        "weak_bearish": ["High volume", "Long lower wick", "Price reclaims support",
                          "Next candle moves higher"]
    },
    {
        "number": 12, "name": "Higher timeframe confirmation",
        "timeframe_roles": {
            "1min": "execution",
            "5min": "short-term structure",
            "15min": "trend confirmation",
            "1hr": "major intraday bias",
            "daily": "major support/resistance"
        }
    },
    {
        "number": 13, "name": "Relative strength / weakness vs SPY or QQQ",
        "patterns": [
            "Stock up while SPY flat = relative strength",
            "Stock flat while SPY falling = hidden strength",
            "Stock falling while SPY rising = relative weakness",
            "Stock fails breakout while SPY strong = caution"
        ]
    },
    {
        "number": 14, "name": "Time of day",
        "windows": {
            "market_open": "high volatility, more fakeouts",
            "midday": "slower, more chop",
            "power_hour": "momentum can return",
            "last_5_15_min": "unstable because of closing flows"
        },
        "rules": [
            "First 15 minutes: observe or use stricter rules",
            "Midday: avoid chop unless setup is very clean",
            "Final 30-60 minutes: only trade if strong trend/volume confirms",
            "Close all trades before EOD if no overnight holds"
        ]
    },
    {
        "number": 15, "name": "ATR / volatility",
        "use_for": ["Stop distance", "Profit target",
                     "Avoiding dead stocks", "Avoiding overextended entries"]
    },
    {
        "number": 16, "name": "Spread and liquidity",
        "track": ["Bid/ask spread", "Average volume", "Slippage",
                   "Candle smoothness", "Whether price jumps around"],
        "avoid": ["Thin volume stocks", "Wide spreads", "Erratic candles",
                   "Large gaps between trades"]
    },
    {
        "number": 17, "name": "Risk/reward before entry",
        "required_fields": ["Entry", "Stop loss", "Target",
                             "Risk per trade", "Reason for trade", "Invalidation point"]
    }
]

_CHECKLIST = [
    "Is price above VWAP or reclaiming VWAP?",
    "Is market structure bullish?",
    "Is price near support, resistance, opening range, or prior high/low?",
    "Did the candle close strong, not just wick up?",
    "Is volume confirming the move?",
    "Is the stock stronger than SPY/QQQ?",
    "Is there enough room to target before resistance?",
    "Is the stop clear?",
    "Is the risk/reward worth it?",
    "Is this during a good trading window, or is it chop time?"
]

_CONFIRMATION_ORDER = ["Market structure", "Key level", "VWAP", "Volume", "Candle close", "Risk/reward"]


def all_signals() -> List[Dict[str, Any]]:
    """Return all 14 candle-structure signal records."""
    return [dict(s) for s in _SIGNALS]


def signal(name: str) -> Optional[Dict[str, Any]]:
    """Lookup a single signal by name (case-insensitive substring match)."""
    name_l = name.lower()
    for s in _SIGNALS:
        if name_l in s["name"].lower():
            return dict(s)
    return None


def pretrade_checklist() -> List[str]:
    """Return the 10-question pre-trade checklist (verbatim)."""
    return list(_CHECKLIST)


def confirmation_layer_order() -> List[str]:
    """Recommended ordering of confluences before any entry."""
    return list(_CONFIRMATION_ORDER)


def bullish_volume_criteria() -> Dict[str, List[str]]:
    """Strong vs weak bullish volume criteria from signal 11."""
    s = signal("Volume + candle relationship")
    return {"strong": list(s["strong_bullish"]), "weak": list(s["weak_bullish"])} if s else {}


def bearish_volume_criteria() -> Dict[str, List[str]]:
    """Strong vs weak bearish volume criteria from signal 11."""
    s = signal("Volume + candle relationship")
    return {"strong": list(s["strong_bearish"]), "weak": list(s["weak_bearish"])} if s else {}


def timeframe_roles() -> Dict[str, str]:
    """The 1m → 5m → 15m → 1h → daily role mapping."""
    s = signal("Higher timeframe confirmation")
    return dict(s["timeframe_roles"]) if s else {}


def _clear_cache():
    """No-op (this module is fully embedded)."""
    pass


if __name__ == "__main__":
    print(f"Signals: {len(all_signals())}")
    print(f"Checklist questions: {len(pretrade_checklist())}")
    print(f"Confirmation order: {confirmation_layer_order()}")
