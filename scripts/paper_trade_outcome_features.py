"""
paper_trade_outcome_features.py — Live paper-trade outcomes as features.

Provides `add_paper_trade_outcome_features(df, ticker) -> df` that adds 7
.shift(1)-safe features to a daily OHLCV+features DataFrame, derived from
CLOSED paper trades emitted by `live_paper_trade.py` + ingest pipeline.

Features added (all rolling 30 calendar-day windows ending at each bar's date,
keyed by closed_at < bar_date so no in-flight position leaks):

  - paper_trade_win_rate_30d        (float in [0,1], 0 if no trades)
  - paper_trade_pf_30d              (float, profit factor = sum_wins/sum_losses;
                                     0 if no trades, 99.0 capped if no losses)
  - paper_trade_count_30d           (int, # closed trades in window)
  - paper_trade_last_outcome_sign   (+1/-1 for last closed trade; 0 if none in window)
  - paper_trade_avg_holding_days    (float, mean (closed_at - opened_at) in days)
  - paper_trade_signal_to_fill_lag_min (float, median minutes from signal generated_at
                                     to opened_at; 0 if missing)
  - paper_trade_in_drawdown_pct     (float, current ticker drawdown % inside the
                                     paper-trade running PnL series; <=0,
                                     0 if no drawdown / no trades)

Source data scanned each refresh:
  $SP/paper_trade/state/<DATE>_state.json   — has closed_trades[]
  $SP/paper_trade/daily/<DATE>/fills.csv    — backup source (optional)
  $SP/paper_trade/signals/<DATE>.json       — for generated_at (signal time)

Caching:
  Cached at $SP/cache/paper_trade_outcomes.parquet — one row per
  (ticker, trading_day) with cumulative rolling features. Cache regenerated
  any time the state/ directory mtime advances (i.e. once daily after ingest).

Graceful failure:
  - Missing paper_trade/ tree → returns df with 7 zero columns.
  - Empty / unparseable state JSON → that day skipped.
  - Idempotent: re-calling on already-augmented df is a no-op.

.shift(1)-safety:
  Each bar at date D sees only trades whose closed_at strictly < D (calendar
  day comparison). The cache keys per (ticker, day) so the lookup is a
  point-in-time merge — no in-flight position leakage.

Author: 2026-05-17 (paper-trade feedback wave)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WORK = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "AI-Tools/s&p500-ticker-mastery"
)
PT_DIR = WORK / "paper_trade"
STATE_DIR = PT_DIR / "state"
DAILY_DIR = PT_DIR / "daily"
SIGNALS_DIR = PT_DIR / "signals"
CACHE_DIR = WORK / "cache"
CACHE_PATH = CACHE_DIR / "paper_trade_outcomes.parquet"

PT_FEATURE_NAMES: list[str] = [
    "paper_trade_win_rate_30d",
    "paper_trade_pf_30d",
    "paper_trade_count_30d",
    "paper_trade_last_outcome_sign",
    "paper_trade_avg_holding_days",
    "paper_trade_signal_to_fill_lag_min",
    "paper_trade_in_drawdown_pct",
]

# Rolling window in calendar days
WINDOW_DAYS = 30
# PF cap when there are wins but no losses
_PF_NO_LOSS_CAP = 99.0


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _safe_load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except Exception as e:
        logger.debug("[pt_outcomes] skip %s: %s", p.name, e)
        return None


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    try:
        # tolerate trailing Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _scan_closed_trades() -> pd.DataFrame:
    """Walk paper_trade/state/*_state.json and return one row per closed trade.

    Columns:
      ticker, trade_date (date of closed_at), opened_at, closed_at,
      entry_price, exit_price, qty, pnl, signal_generated_at
    """
    rows: list[dict] = []
    if not STATE_DIR.exists():
        return pd.DataFrame()

    # cache signals_generated_at lookups per (date, ticker)
    sig_cache: dict[tuple[str, str], Optional[datetime]] = {}

    for state_path in sorted(STATE_DIR.glob("*_state.json")):
        state = _safe_load_json(state_path)
        if not state:
            continue
        date_str = state.get("date") or state_path.stem.split("_state")[0]

        # Load signals file for that date (once)
        for trade in state.get("closed_trades", []) or []:
            ticker = trade.get("ticker")
            if not ticker:
                continue

            opened_at = _parse_iso(trade.get("opened_at"))
            closed_at = _parse_iso(trade.get("closed_at"))
            entry = trade.get("entry_price")
            exit_ = trade.get("exit_price")
            pnl = trade.get("pnl")
            qty = trade.get("qty")

            # Resolve trade_date from closed_at if available, otherwise state's date
            if closed_at is not None:
                trade_date = closed_at.astimezone(timezone.utc).date().isoformat()
            else:
                trade_date = date_str

            # Generated_at lookup: load once per date
            key = (date_str, ticker)
            if key not in sig_cache:
                sig_path = SIGNALS_DIR / f"{date_str}.json"
                gen_at: Optional[datetime] = None
                if sig_path.exists():
                    sig_payload = _safe_load_json(sig_path)
                    if isinstance(sig_payload, list):
                        for s in sig_payload:
                            if s.get("ticker") == ticker:
                                gen_at = _parse_iso(s.get("generated_at"))
                                break
                sig_cache[key] = gen_at
            signal_generated_at = sig_cache[key]

            rows.append(
                {
                    "ticker": ticker,
                    "trade_date": trade_date,
                    "opened_at": opened_at,
                    "closed_at": closed_at,
                    "entry_price": float(entry) if entry is not None else np.nan,
                    "exit_price": float(exit_) if exit_ is not None else np.nan,
                    "qty": float(qty) if qty is not None else np.nan,
                    "pnl": float(pnl) if pnl is not None else np.nan,
                    "signal_generated_at": signal_generated_at,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.tz_localize(None)
    # Drop unusable rows
    df = df.dropna(subset=["pnl"]).copy()
    return df


# ---------------------------------------------------------------------------
# Per-ticker rolling feature build
# ---------------------------------------------------------------------------


def _build_rolling_features(trades_t: pd.DataFrame) -> pd.DataFrame:
    """Given closed trades for ONE ticker, produce per-trade-day rolling features.

    The output is one row per UNIQUE trade_date for this ticker with the 7
    features computed over the trailing WINDOW_DAYS ending at that date.

    Drawdown is computed from the cumulative-pnl series of this ticker's
    closed trades up to and including that date.
    """
    if trades_t.empty:
        return pd.DataFrame()

    df = trades_t.sort_values("closed_at").reset_index(drop=True).copy()

    # cumulative pnl + running peak for drawdown
    df["cum_pnl"] = df["pnl"].cumsum()
    df["running_peak"] = df["cum_pnl"].cummax()
    df["dd_pct"] = np.where(
        df["running_peak"] > 0,
        (df["cum_pnl"] - df["running_peak"]) / df["running_peak"] * 100.0,
        0.0,
    )
    # holding days
    if "opened_at" in df.columns and "closed_at" in df.columns:
        hold_secs = (df["closed_at"] - df["opened_at"]).dt.total_seconds()
        df["holding_days"] = hold_secs / 86400.0
    else:
        df["holding_days"] = np.nan
    # signal-to-fill lag in minutes
    if "signal_generated_at" in df.columns:
        lag_secs = (df["opened_at"] - df["signal_generated_at"]).dt.total_seconds()
        df["s2f_lag_min"] = lag_secs / 60.0
    else:
        df["s2f_lag_min"] = np.nan

    # Build per-day aggregate (one row per trade_date)
    out_rows: list[dict] = []
    unique_days = sorted(df["trade_date"].unique())
    for day in unique_days:
        # rolling window: trades closed in [day - WINDOW_DAYS, day]
        window_lo = day - pd.Timedelta(days=WINDOW_DAYS)
        mask = (df["trade_date"] > window_lo) & (df["trade_date"] <= day)
        w = df.loc[mask]
        # running drawdown is the LATEST dd through this day
        running = df.loc[df["trade_date"] <= day]
        if w.empty:
            continue

        wins = w.loc[w["pnl"] > 0, "pnl"]
        losses = w.loc[w["pnl"] < 0, "pnl"]
        n = len(w)
        win_rate = float((w["pnl"] > 0).sum()) / n if n > 0 else 0.0
        if len(losses) > 0:
            pf = float(wins.sum() / abs(losses.sum()))
        elif len(wins) > 0:
            pf = _PF_NO_LOSS_CAP
        else:
            pf = 0.0
        last_pnl = w.iloc[-1]["pnl"]
        last_sign = 1 if last_pnl > 0 else (-1 if last_pnl < 0 else 0)
        avg_hold = float(w["holding_days"].dropna().mean()) if w["holding_days"].notna().any() else 0.0
        med_lag = float(w["s2f_lag_min"].dropna().median()) if w["s2f_lag_min"].notna().any() else 0.0
        dd_now = float(running.iloc[-1]["dd_pct"]) if not running.empty else 0.0

        out_rows.append(
            {
                "trade_date": day,
                "paper_trade_win_rate_30d": win_rate,
                "paper_trade_pf_30d": pf,
                "paper_trade_count_30d": int(n),
                "paper_trade_last_outcome_sign": int(last_sign),
                "paper_trade_avg_holding_days": avg_hold,
                "paper_trade_signal_to_fill_lag_min": med_lag,
                "paper_trade_in_drawdown_pct": dd_now,
            }
        )

    if not out_rows:
        return pd.DataFrame()
    out = pd.DataFrame(out_rows).sort_values("trade_date").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def _cache_valid(cache_path: Path, source_dir: Path) -> bool:
    if not cache_path.exists() or not source_dir.exists():
        return False
    try:
        return cache_path.stat().st_mtime >= source_dir.stat().st_mtime
    except OSError:
        return False


def load_paper_trade_outcomes_table(force_refresh: bool = False) -> pd.DataFrame:
    """Return per-(ticker, trade_date) rolling-features DataFrame, cached.

    Returns empty DataFrame (with named columns) when no paper trades exist.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force_refresh and _cache_valid(CACHE_PATH, STATE_DIR):
        try:
            df = pd.read_parquet(CACHE_PATH)
            logger.debug("[pt_outcomes] cache HIT: %d rows", len(df))
            return df
        except Exception as e:
            logger.warning("[pt_outcomes] cache read failed: %s — regenerating", e)

    trades = _scan_closed_trades()
    if trades.empty:
        empty = pd.DataFrame(
            columns=["ticker", "trade_date", *PT_FEATURE_NAMES]
        )
        try:
            empty.to_parquet(CACHE_PATH, index=False)
        except Exception as e:
            logger.debug("[pt_outcomes] empty cache write skipped: %s", e)
        return empty

    blocks: list[pd.DataFrame] = []
    for ticker, grp in trades.groupby("ticker"):
        feats = _build_rolling_features(grp)
        if feats.empty:
            continue
        feats.insert(0, "ticker", ticker)
        blocks.append(feats)

    if not blocks:
        return pd.DataFrame(columns=["ticker", "trade_date", *PT_FEATURE_NAMES])

    out = pd.concat(blocks, ignore_index=True)
    try:
        out.to_parquet(CACHE_PATH, index=False)
        logger.info("[pt_outcomes] cache wrote %d rows -> %s", len(out), CACHE_PATH)
    except Exception as e:
        logger.warning("[pt_outcomes] cache write failed: %s", e)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in PT_FEATURE_NAMES:
        if col not in df.columns:
            if col in ("paper_trade_count_30d", "paper_trade_last_outcome_sign"):
                df[col] = 0
            else:
                df[col] = 0.0
    return df


def add_paper_trade_outcome_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add 7 paper-trade outcome features to df. Idempotent + .shift(1)-safe.

    Args:
        df: Daily DataFrame, expected DatetimeIndex (or column 'date').
        ticker: Symbol to look up in cached paper-trade outcomes.

    Returns:
        df with 7 new columns appended (zero-filled when no paper-trade data
        is available for this ticker yet, or when the paper_trade/ tree is
        missing entirely).
    """
    if df is None or len(df) == 0:
        return df

    # Idempotent guard: if all 7 present, no-op.
    present = [c for c in PT_FEATURE_NAMES if c in df.columns]
    if len(present) == len(PT_FEATURE_NAMES):
        logger.debug("[pt_outcomes] all 7 features already present — skip")
        return df

    # Bail-friendly: if paper_trade/ tree missing entirely, zero-fill.
    if not PT_DIR.exists():
        logger.info("[pt_outcomes] paper_trade/ missing — zero-filling 7 cols")
        return _zero_fill(df)

    try:
        outcomes = load_paper_trade_outcomes_table()
    except Exception as e:
        logger.warning("[pt_outcomes] table load failed (%s) — zeroing", e)
        return _zero_fill(df)

    if outcomes.empty:
        return _zero_fill(df)

    ticker_rows = outcomes[outcomes["ticker"] == ticker].copy()
    if ticker_rows.empty:
        return _zero_fill(df)

    ticker_rows = ticker_rows.sort_values("trade_date").reset_index(drop=True)
    ticker_rows["trade_date"] = pd.to_datetime(ticker_rows["trade_date"]).dt.tz_localize(None)

    # Build a Series indexed by bar date that picks the latest trade_date STRICTLY
    # before each bar date (this is the .shift(1) guard).
    if isinstance(df.index, pd.DatetimeIndex):
        bar_dates = df.index
    elif "date" in df.columns:
        bar_dates = pd.to_datetime(df["date"]).values
    else:
        # Fallback: cannot align by date → zero-fill
        logger.warning("[pt_outcomes] df has no DatetimeIndex or 'date' col — zeroing")
        return _zero_fill(df)

    bar_idx = pd.to_datetime(pd.Index(bar_dates))
    if bar_idx.tz is not None:
        bar_idx = bar_idx.tz_convert(None)

    # merge_asof requires sorted left + right by key
    left = pd.DataFrame({"bar_date": bar_idx}).reset_index(drop=True)
    left_sorted = left.sort_values("bar_date").reset_index()  # keep orig index
    # Right side trade_date must be < bar_date for strict shift(1) safety. Trick:
    # merge_asof direction='backward' with allow_exact_matches=False gives strict <.
    right = ticker_rows.rename(columns={"trade_date": "bar_date"})
    merged = pd.merge_asof(
        left_sorted,
        right,
        on="bar_date",
        direction="backward",
        allow_exact_matches=False,
    )
    # Restore original order
    merged = merged.sort_values("index").reset_index(drop=True)

    # Fill any pre-first-trade NaNs with zero defaults
    for col in PT_FEATURE_NAMES:
        if col not in merged.columns:
            merged[col] = 0
        if col in ("paper_trade_count_30d", "paper_trade_last_outcome_sign"):
            merged[col] = merged[col].fillna(0).astype(int)
        else:
            merged[col] = merged[col].fillna(0.0).astype(float)

    # Drop ticker column from merge if present
    if "ticker" in merged.columns:
        merged = merged.drop(columns=["ticker"])

    # Attach back to df by positional alignment (we preserved order)
    for col in PT_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = merged[col].values

    return df


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    # Build a synthetic 100-day daily frame to verify add_paper_trade_outcome_features
    idx = pd.date_range(end=datetime.utcnow().date(), periods=100, freq="B")
    demo = pd.DataFrame({"close": np.linspace(100, 110, len(idx))}, index=idx)
    out = add_paper_trade_outcome_features(demo, ticker)
    print(f"Input cols: 1  Output cols: {out.shape[1]}")
    added = [c for c in out.columns if c.startswith("paper_trade_")]
    print(f"Added {len(added)} paper-trade features: {added}")
    print(out[added].tail(3).to_string())
