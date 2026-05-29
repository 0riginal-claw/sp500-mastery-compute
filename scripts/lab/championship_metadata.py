"""championship_metadata.py — Per-ticker metadata enrichment (items 2-6 of the 23-item spec).

Fills the 5 metadata fields that the championship_search writer leaves as
`unknown_pending_metadata_backfill`:

    (2) GICS sector       — from sp500_constituents-detailed.csv (Wikipedia GICS Sector)
    (3) Market cap        — shares_outstanding * last_close
                            shares from a static manifest (publicly disclosed cover-page
                            shares from latest 10-Q); future tickers fall back to a
                            try-edgar-XBRL-then-NaN path
    (4) ADV (20-day)      — mean(volume[-20:]) from yfinance_5yr parquet cache
    (5) Realized vol      — annualized realized vol from daily log-returns (last 60d),
                            decile within the 502-universe distribution (1=low, 10=high)
    (6) Beta              — 60-day OLS regression of ticker returns on SPY returns

Caches the 502-universe vol distribution to `data/universe_vol_distribution.json`
so deciles are O(1) thereafter. Vol distribution refreshes when the cache is older
than `VOL_CACHE_TTL_DAYS` (default 14 days) — adequate for a 5y realized vol metric.

Public API:
  enrich_metadata(ticker) -> {sector, mcap, adv_20d, vol_decile, beta}
  build_universe_vol_distribution(force=False) -> dict
  load_sector_map() -> dict[ticker -> sector]
  shares_outstanding(ticker) -> Optional[int]   (static manifest + EDGAR XBRL fallback hook)
  market_cap(ticker) -> Optional[float]

Notes on Gabriel 1Day vs yfinance_5yr:
  The brief mentions `Gabriel/1Day/<T>/2026-05.parquet` but the brief was generic;
  championship_search ALREADY references `lab.indicator_hardening_runner.DRIVE_OHLC_DAILY`,
  which resolves to `AI-Tools/s&p500-ticker-mastery/cache/yfinance_5yr/<T>.parquet`
  (5y daily bars, columns: date, open, high, low, close, volume, ticker).  We use that
  same cache here so the enricher reads the SAME data the strategy backtester sees.
  Gabriel monthly slices have only ~19 days each — not enough for a 20-day ADV in a
  single file. yfinance_5yr is the consistent single-source-of-truth for the daily
  metadata path.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_BASE = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
)
SP500_MASTERY = DRIVE_BASE / "AI-Tools/s&p500-ticker-mastery"
DATA_DIR = SP500_MASTERY / "data"
CACHE_DIR = SP500_MASTERY / "cache"

DRIVE_OHLC_DAILY = CACHE_DIR / "yfinance_5yr"            # ticker → <T>.parquet, columns: date, open, high, low, close, volume, ticker
SP500_UNIVERSE_CSV = CACHE_DIR / "sp500_universe.csv"     # single-column 502-ticker list
SP500_DETAILED_CSV = (
    DRIVE_BASE / "AI-Tools/external-repos/trading-free-clones/strategy-impls/"
                 "arbitrage_research/Copula Approach/sp500_constituents-detailed.csv"
)  # canonical Wikipedia GICS Sector

VOL_DIST_CACHE = DATA_DIR / "universe_vol_distribution.json"
VOL_CACHE_TTL_DAYS = 14
RV_WINDOW_DAYS = 60      # realized vol window (5min-equivalent on daily: 60 day rolling)
BETA_WINDOW_DAYS = 60    # rolling beta window
ADV_WINDOW_DAYS = 20     # ADV (20-day average daily volume)
TRADING_DAYS_PER_YEAR = 252


# ─────────────────────────────────────────────────────────────────────────────
# Static shares_outstanding manifest (covers the 3 bootstrap tickers + room to grow)
# ─────────────────────────────────────────────────────────────────────────────
# Values sourced from latest 10-Q cover page (publicly available; SEC EDGAR).
# When `enrich_metadata` is called for a ticker not in this manifest the function
# attempts an EDGAR XBRL lookup (CommonStockSharesOutstanding); on miss it returns
# None, and the orchestrator gets `mcap=None` rather than a wrong number.
#
# Format: ticker → (shares_outstanding, as_of_date, source_form, source_url)
# To extend: add rows here OR populate `data/shares_outstanding_manifest.json`
# (which this module ALSO loads, deep-merged on top of the static dict).
_STATIC_SHARES_OUT: Dict[str, Tuple[int, str, str, str]] = {
    # AAPL — 10-Q Q2 FY2025 (filed 2025-05), cover page shows 14,946,948,000 shares
    "AAPL": (14_946_948_000, "2025-04-26", "10-Q",
             "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-Q"),
    # NVDA — 10-Q Q1 FY2026 (filed 2025-05), cover page shows 24,427,000,000 shares (post-split)
    "NVDA": (24_427_000_000, "2025-05-23", "10-Q",
             "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001045810&type=10-Q"),
    # A (Agilent Technologies) — 10-Q (filed 2025-03), cover page shows 284,300,000 shares
    "A": (284_300_000, "2025-02-26", "10-Q",
          "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001090872&type=10-Q"),
}


def _load_shares_manifest() -> Dict[str, int]:
    """Merge static dict with optional `data/shares_outstanding_manifest.json`.

    JSON file format: `{"TICKER": shares_outstanding_int, ...}`. JSON wins on conflict.
    """
    out: Dict[str, int] = {t: v[0] for t, v in _STATIC_SHARES_OUT.items()}
    extra = DATA_DIR / "shares_outstanding_manifest.json"
    if extra.exists():
        try:
            j = json.loads(extra.read_text())
            for k, v in j.items():
                if isinstance(v, (int, float)) and v > 0:
                    out[k.upper()] = int(v)
        except (OSError, json.JSONDecodeError):
            pass
    return out


def shares_outstanding(ticker: str) -> Optional[int]:
    """Return shares outstanding for `ticker` (int) or None if unknown.

    Lookup order:
      1. Static manifest (built-in 3 tickers + JSON override)
      2. EDGAR XBRL via `edgar_cache_loader` (fact: us-gaap:CommonStockSharesOutstanding)
         — best-effort; returns None on miss.
    """
    tk = ticker.upper()
    mfst = _load_shares_manifest()
    if tk in mfst:
        return mfst[tk]
    # Fallback: EDGAR XBRL — only attempted if loader is importable
    try:
        sys.path.insert(0, str(SP500_MASTERY / "scripts"))
        import edgar_cache_loader as _e  # type: ignore
        if hasattr(_e, "get_xbrl_fact"):
            val = _e.get_xbrl_fact(tk, "us-gaap:CommonStockSharesOutstanding")  # type: ignore[attr-defined]
            if val and isinstance(val, (int, float)) and val > 0:
                return int(val)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Sector lookup
# ─────────────────────────────────────────────────────────────────────────────
_SECTOR_CACHE: Optional[Dict[str, str]] = None


def load_sector_map() -> Dict[str, str]:
    """Load ticker → GICS Sector dict from sp500_constituents-detailed.csv.

    Returns the canonical S&P GICS Sector strings (e.g. "Information Technology",
    "Health Care"). Caches in-process. Falls back to {} if the source CSV is
    inaccessible (FUSE-blind, etc.).
    """
    global _SECTOR_CACHE
    if _SECTOR_CACHE is not None:
        return _SECTOR_CACHE
    out: Dict[str, str] = {}
    if not SP500_DETAILED_CSV.exists():
        _SECTOR_CACHE = out
        return out
    try:
        with SP500_DETAILED_CSV.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                sym = (row.get("Symbol") or "").strip().upper()
                sec = (row.get("GICS Sector") or "").strip()
                if sym and sec:
                    out[sym] = sec
    except OSError as e:
        print(f"[championship_metadata] sector csv read failed: {e}", flush=True)
    _SECTOR_CACHE = out
    return out


# ─────────────────────────────────────────────────────────────────────────────
# OHLC loader (yfinance_5yr parquet)
# ─────────────────────────────────────────────────────────────────────────────


def _load_daily(ticker: str):
    """Return pandas DataFrame with columns [date, open, high, low, close, volume]
    sorted ascending by date. None if not found or unreadable.
    """
    import pandas as pd
    p = DRIVE_OHLC_DAILY / f"{ticker.upper()}.parquet"
    if not p.exists():
        return None
    try:
        import pyarrow.parquet as pq
        df = pq.read_table(p).to_pandas()
    except Exception as e:
        print(f"[championship_metadata] couldn't read {p}: {e}", flush=True)
        return None
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker compute primitives
# ─────────────────────────────────────────────────────────────────────────────


def _adv_20d(df) -> Optional[float]:
    if df is None or "volume" not in df.columns or len(df) < ADV_WINDOW_DAYS:
        return None
    v = df["volume"].tail(ADV_WINDOW_DAYS).astype(float)
    if v.isna().all():
        return None
    return float(v.mean())


def _realized_vol_ann(df, window: int = RV_WINDOW_DAYS) -> Optional[float]:
    """Annualized realized vol from daily log-returns over the trailing `window` days."""
    if df is None or "close" not in df.columns or len(df) < window + 1:
        return None
    close = df["close"].astype(float).tail(window + 1).to_numpy()
    if not np.isfinite(close).all() or (close <= 0).any():
        return None
    log_ret = np.log(close[1:] / close[:-1])
    if not np.isfinite(log_ret).all() or log_ret.std(ddof=1) == 0:
        return None
    return float(log_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _beta_vs_spy(df_ticker, df_spy, window: int = BETA_WINDOW_DAYS) -> Optional[float]:
    """OLS beta of ticker daily returns on SPY daily returns over trailing `window`."""
    import pandas as pd
    if df_ticker is None or df_spy is None:
        return None
    if "close" not in df_ticker.columns or "close" not in df_spy.columns:
        return None
    # Align by date
    a = df_ticker[["date", "close"]].rename(columns={"close": "tk"}).copy()
    b = df_spy[["date", "close"]].rename(columns={"close": "spy"}).copy()
    m = a.merge(b, on="date", how="inner").tail(window + 1)
    if len(m) < window + 1:
        return None
    rt = np.log(m["tk"].to_numpy()[1:] / m["tk"].to_numpy()[:-1])
    rs = np.log(m["spy"].to_numpy()[1:] / m["spy"].to_numpy()[:-1])
    if not (np.isfinite(rt).all() and np.isfinite(rs).all()):
        return None
    var_s = rs.var(ddof=1)
    if var_s == 0:
        return None
    cov = np.cov(rt, rs, ddof=1)[0, 1]
    return float(cov / var_s)


# ─────────────────────────────────────────────────────────────────────────────
# Universe vol distribution → decile mapping
# ─────────────────────────────────────────────────────────────────────────────


def _load_universe() -> List[str]:
    if not SP500_UNIVERSE_CSV.exists():
        return []
    out: List[str] = []
    try:
        with SP500_UNIVERSE_CSV.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                t = (row.get("ticker") or "").strip().upper()
                if t:
                    out.append(t)
    except OSError:
        return []
    return out


def _cache_is_fresh(p: Path, ttl_days: int) -> bool:
    if not p.exists():
        return False
    try:
        age_sec = time.time() - p.stat().st_mtime
        return age_sec < ttl_days * 86400
    except OSError:
        return False


def build_universe_vol_distribution(force: bool = False) -> dict:
    """Compute realized vol for every ticker in sp500_universe.csv, cache to JSON.

    Output JSON layout:
        {
          "computed_at": "<UTC>",
          "window_days": 60,
          "n_tickers": 502,
          "n_with_vol": <int>,
          "vols": { "TICKER": <annualized_realized_vol>, ... },
          "deciles": [d10, d20, d30, d40, d50, d60, d70, d80, d90]  # 9 cut-points
        }
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not force and _cache_is_fresh(VOL_DIST_CACHE, VOL_CACHE_TTL_DAYS):
        try:
            return json.loads(VOL_DIST_CACHE.read_text())
        except (OSError, json.JSONDecodeError):
            pass

    tickers = _load_universe()
    vols: Dict[str, float] = {}
    n_missing = 0
    for tk in tickers:
        df = _load_daily(tk)
        rv = _realized_vol_ann(df)
        if rv is None:
            n_missing += 1
            continue
        vols[tk] = rv

    # Compute decile cut-points
    if vols:
        arr = np.array(sorted(vols.values()))
        deciles = [float(np.percentile(arr, p)) for p in (10, 20, 30, 40, 50, 60, 70, 80, 90)]
    else:
        deciles = []

    out = {
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_days": RV_WINDOW_DAYS,
        "n_tickers": len(tickers),
        "n_with_vol": len(vols),
        "n_missing": n_missing,
        "vols": vols,
        "deciles": deciles,
        "source": "yfinance_5yr daily close → log-return std × sqrt(252)",
    }
    tmp = VOL_DIST_CACHE.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(out, indent=2, default=str))
        tmp.replace(VOL_DIST_CACHE)
    except OSError as e:
        print(f"[championship_metadata] vol cache write failed: {e}", flush=True)
    return out


def _vol_decile_for(vol: Optional[float], dist: dict) -> Optional[int]:
    """Return 1..10 decile for `vol` within the universe distribution. 1 = lowest vol."""
    if vol is None or not dist.get("deciles"):
        return None
    cuts = dist["deciles"]  # 9 cut-points → 10 buckets
    bucket = 1
    for c in cuts:
        if vol > c:
            bucket += 1
        else:
            break
    # clamp 1..10
    return max(1, min(10, bucket))


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_mcap(mcap: Optional[float]) -> str:
    if mcap is None or not np.isfinite(mcap):
        return "unknown"
    if mcap >= 1e12:
        return f"${mcap/1e12:.2f}T"
    if mcap >= 1e9:
        return f"${mcap/1e9:.2f}B"
    if mcap >= 1e6:
        return f"${mcap/1e6:.2f}M"
    return f"${mcap:,.0f}"


def _fmt_adv(adv: Optional[float]) -> str:
    if adv is None or not np.isfinite(adv):
        return "unknown"
    if adv >= 1e9:
        return f"{adv/1e9:.2f}B shares"
    if adv >= 1e6:
        return f"{adv/1e6:.2f}M shares"
    if adv >= 1e3:
        return f"{adv/1e3:.2f}K shares"
    return f"{adv:,.0f} shares"


def _fmt_vol_decile(d: Optional[int]) -> str:
    if d is None:
        return "unknown"
    return f"D{d}/10"


def _fmt_beta(b: Optional[float]) -> str:
    if b is None or not np.isfinite(b):
        return "unknown"
    return f"{b:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# Public: enrich_metadata
# ─────────────────────────────────────────────────────────────────────────────


def enrich_metadata(
    ticker: str,
    vol_dist: Optional[dict] = None,
    formatted: bool = True,
) -> Dict[str, Any]:
    """Compute (sector, mcap, adv_20d, vol_decile, beta) for a single ticker.

    Args:
      ticker: e.g. "AAPL". Case-insensitive.
      vol_dist: pre-loaded universe vol distribution dict (avoid recompute when
                enriching many tickers). If None, will load/build from cache.
      formatted: when True (default), returns strings ready for the markdown writer.
                 When False, returns raw numeric values (useful for tests / posterior).

    Returns dict with keys:
      sector, mcap, adv_20d, vol_decile, beta
      plus debug fields: _raw_mcap, _raw_adv, _raw_vol_ann, _raw_beta, _shares_out
    """
    tk = ticker.upper()
    sectors = load_sector_map()
    sector = sectors.get(tk)

    df = _load_daily(tk)
    df_spy = _load_daily("SPY")

    # ADV 20d
    adv_raw = _adv_20d(df)

    # Market cap
    so = shares_outstanding(tk)
    last_close: Optional[float] = None
    if df is not None and "close" in df.columns and len(df) > 0:
        try:
            lc = df["close"].iloc[-1]
            if np.isfinite(lc):
                last_close = float(lc)
        except Exception:
            last_close = None
    mcap_raw: Optional[float] = None
    if so is not None and last_close is not None:
        mcap_raw = float(so) * float(last_close)

    # Realized vol → decile
    vol_raw = _realized_vol_ann(df)
    if vol_dist is None:
        vol_dist = build_universe_vol_distribution(force=False)
    decile = _vol_decile_for(vol_raw, vol_dist)

    # Beta
    beta_raw = _beta_vs_spy(df, df_spy)

    if not formatted:
        return {
            "sector": sector,
            "mcap": mcap_raw,
            "adv_20d": adv_raw,
            "vol_decile": decile,
            "beta": beta_raw,
            "_raw_vol_ann": vol_raw,
            "_shares_out": so,
            "_last_close": last_close,
        }

    return {
        "sector": sector or "unknown",
        "mcap": _fmt_mcap(mcap_raw),
        "adv_20d": _fmt_adv(adv_raw),
        "vol_decile": _fmt_vol_decile(decile),
        "beta": _fmt_beta(beta_raw),
        # Debug-only (orchestrator can drop these before writing)
        "_raw_mcap": mcap_raw,
        "_raw_adv": adv_raw,
        "_raw_vol_ann": vol_raw,
        "_raw_beta": beta_raw,
        "_shares_out": so,
        "_last_close": last_close,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backfill helper — patch existing CHAMPIONSHIP_FORMULA.md files in place
# ─────────────────────────────────────────────────────────────────────────────


_PLACEHOLDER = "unknown_pending_metadata_backfill"


def backfill_championship_file(path: Path, meta: Dict[str, str]) -> bool:
    """Replace items 2-6 in an existing CHAMPIONSHIP_FORMULA.md with `meta` values.

    Atomic write (tmp → replace). Returns True if file changed, False otherwise.
    """
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"[backfill] read failed {path}: {e}", flush=True)
        return False

    replacements = [
        (f"2. **Sector**: {_PLACEHOLDER}", f"2. **Sector**: {meta['sector']}"),
        (f"3. **Market Cap**: {_PLACEHOLDER}", f"3. **Market Cap**: {meta['mcap']}"),
        (f"4. **ADV (Avg Daily Volume)**: {_PLACEHOLDER}",
         f"4. **ADV (Avg Daily Volume)**: {meta['adv_20d']}"),
        (f"5. **Realized Vol decile**: {_PLACEHOLDER}",
         f"5. **Realized Vol decile**: {meta['vol_decile']}"),
        (f"6. **Beta**: {_PLACEHOLDER}", f"6. **Beta**: {meta['beta']}"),
    ]
    new_text = text
    n_changes = 0
    for old, new in replacements:
        if old in new_text:
            new_text = new_text.replace(old, new)
            n_changes += 1

    # Also rewrite the trailing note line so it accurately reflects the backfill
    note_old = (
        "- Metadata items 2-6 marked 'unknown_pending_metadata_backfill' will be "
        "filled by a separate metadata enrichment pass (sector/mcap/ADV "
        "sourced from sp500_universe.csv when accessible)."
    )
    note_new = (
        "- Metadata items 2-6 enriched by `lab.championship_metadata.enrich_metadata` "
        "(sector from S&P GICS, mcap = shares_outstanding × last_close, ADV from "
        "20-day yfinance volume, vol decile within 502-universe, beta vs SPY)."
    )
    if note_old in new_text:
        new_text = new_text.replace(note_old, note_new)

    if new_text == text:
        return False

    tmp = path.with_suffix(".md.tmp")
    try:
        tmp.write_text(new_text, encoding="utf-8")
        tmp.replace(path)
        print(f"[backfill] patched {n_changes} items in {path}", flush=True)
        return True
    except OSError as e:
        print(f"[backfill] write failed {path}: {e}", flush=True)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _championship_paths(ticker: str) -> List[Path]:
    """Same 3 mirror layout as championship_search.write_championship_file."""
    tk = ticker.upper()
    return [
        DRIVE_BASE / "Tech0/Data Master/universe/mastered" / tk / "CHAMPIONSHIP_FORMULA.md",
        Path("/Volumes/ZG-2TB/zg/championship_mirror") / tk / "CHAMPIONSHIP_FORMULA.md",
        DRIVE_BASE / "AI-Tools/research-lab/data_inventory/championships" / tk / "CHAMPIONSHIP_FORMULA.md",
    ]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["A", "AAPL", "NVDA"])
    ap.add_argument("--backfill", action="store_true",
                    help="Patch existing CHAMPIONSHIP_FORMULA.md files in place.")
    ap.add_argument("--rebuild-vol", action="store_true",
                    help="Force-rebuild the universe vol distribution cache.")
    args = ap.parse_args()

    print(f"[championship_metadata] building universe vol distribution (force={args.rebuild_vol})…",
          flush=True)
    t0 = time.time()
    vol_dist = build_universe_vol_distribution(force=args.rebuild_vol)
    print(f"  done in {time.time()-t0:.1f}s — n_with_vol={vol_dist.get('n_with_vol')}/{vol_dist.get('n_tickers')}, "
          f"cache={VOL_DIST_CACHE.stat().st_size if VOL_DIST_CACHE.exists() else 0} bytes",
          flush=True)
    if vol_dist.get("deciles"):
        print(f"  decile cut-points: {[round(d,3) for d in vol_dist['deciles']]}", flush=True)

    results = {}
    for tk in args.tickers:
        meta = enrich_metadata(tk, vol_dist=vol_dist, formatted=True)
        results[tk] = meta
        print(f"\n[{tk}] {meta['sector']} | mcap={meta['mcap']} | "
              f"ADV={meta['adv_20d']} | vol={meta['vol_decile']} | beta={meta['beta']}",
              flush=True)
        if args.backfill:
            for p in _championship_paths(tk):
                backfill_championship_file(p, meta)

    print("\n=== SUMMARY ===")
    print(json.dumps(
        {tk: {k: v for k, v in m.items() if not k.startswith("_")}
         for tk, m in results.items()},
        indent=2, default=str,
    ))


if __name__ == "__main__":
    sys.exit(main() or 0)
