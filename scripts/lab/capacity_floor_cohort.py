"""
Capacity-Floor Cohort Builder — Task #84 / Expansionist R3-R6

Builds the untested universe: liquid US stocks BELOW the HFT capacity floor
that are NOT in the S&P 500. The lab's structural-edge claim has only ever
been tested on S&P 500 large-caps; this cohort isolates the regime where
the edge is theoretically strongest (sub-HFT-capacity names quants can't
trade at scale).

Pipeline:
  A) Universe construction — Alpaca assets.get_all_assets() filtered to
     active tradable US equities on NYSE/NASDAQ, exclude S&P 500.
  B) Liquidity + market cap measurement — Alpaca snapshots for $-ADV
     proxy (latest trade price × latest day volume), yfinance for
     market cap (fast_info.marketCap).
  C) Filter to capacity-floor band — $-ADV in (10M, 30M), mcap in
     ($1B, $10B). Keep top-200 by $-ADV.
  D) Output a versioned CSV cohort under data/.

Run:
    python -m lab.capacity_floor_cohort

Output:
    data/capacity_floor_cohort_2026-05-29.csv

Auth: macOS Keychain (service: alpaca-paper-api-key, alpaca-paper-secret-key)
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
DRIVE_BASE = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
LAB_BASE = f"{DRIVE_BASE}/AI-Tools/s&p500-ticker-mastery"
DATA_DIR = Path(f"{LAB_BASE}/data")
SCRATCH = Path("/Volumes/ZG-2TB/zg/tmp/champ_003c")
STATUS_PATH = SCRATCH / "status.md"

COHORT_CSV = DATA_DIR / "capacity_floor_cohort_2026-05-29.csv"
CANDIDATES_CACHE = SCRATCH / "candidates_cache.json"
SNAPSHOTS_CACHE = SCRATCH / "snapshots_cache.json"
MCAP_CACHE = SCRATCH / "mcap_cache.json"

# ----------------------------------------------------------------------------
# Auth (Keychain)
# ----------------------------------------------------------------------------
def _keychain_get(service: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
        return None
    except Exception:
        return None


def _load_env_file(path: str) -> Dict[str, str]:
    """Parse a shell-style .env file (KEY=VALUE per line, may have 'export ' prefix
    and trailing comments). Quotes are stripped. Lines containing only `export KEY`
    (no `=`) are skipped."""
    env: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip()
                # strip surrounding quotes
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                # strip trailing comment if value isn't quoted
                if "#" in v and not v.startswith('"') and not v.startswith("'"):
                    v = v.split("#", 1)[0].strip()
                env[k] = v
    except FileNotFoundError:
        pass
    return env


def alpaca_credentials() -> tuple[str, str]:
    """Resolve paper API credentials. Env vars first, then ~/.config/auto_signup/alpaca.env,
    then macOS Keychain."""
    # 1) env
    key = os.environ.get("ALPACA_PAPER_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_PAPER_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")

    # 2) shell env file (this Mac's canonical Alpaca credentials)
    if not (key and secret):
        env_path = os.path.expanduser("~/.config/auto_signup/alpaca.env")
        loaded = _load_env_file(env_path)
        key = key or loaded.get("ALPACA_PAPER_API_KEY") or loaded.get("APCA_API_KEY_ID") or loaded.get("ALPACA_API_KEY")
        secret = secret or loaded.get("ALPACA_PAPER_SECRET_KEY") or loaded.get("APCA_API_SECRET_KEY") or loaded.get("ALPACA_SECRET_KEY")

    # 3) Keychain (legacy fallback)
    if not key:
        key = _keychain_get("alpaca-paper-api-key")
    if not secret:
        secret = _keychain_get("alpaca-paper-secret-key")

    if not key or not secret:
        raise RuntimeError(
            "Alpaca credentials not found: env vars / ~/.config/auto_signup/alpaca.env / Keychain all empty."
        )
    return key, secret


# ----------------------------------------------------------------------------
# S&P 500 universe (to exclude)
# ----------------------------------------------------------------------------
def sp500_set() -> set[str]:
    """Load the 502 mastered tickers (S&P 500-like universe) to exclude."""
    sys.path.insert(0, f"{LAB_BASE}/scripts")
    from lab.knowledge import inventory  # type: ignore
    return set(inventory.mastered())


# ----------------------------------------------------------------------------
# A) Universe construction
# ----------------------------------------------------------------------------
def fetch_candidate_universe() -> List[Dict]:
    """
    Hit Alpaca assets endpoint, filter to active+tradable US equities on
    NYSE/NASDAQ, exclude S&P 500 mastered set.
    """
    if CANDIDATES_CACHE.exists():
        try:
            cached = json.loads(CANDIDATES_CACHE.read_text())
            if isinstance(cached, list) and cached:
                print(f"[A] using cached candidate universe: {len(cached)} tickers")
                return cached
        except Exception:
            pass

    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus

    key, secret = alpaca_credentials()
    client = TradingClient(api_key=key, secret_key=secret, paper=True)

    req = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,
        status=AssetStatus.ACTIVE,
    )
    assets = client.get_all_assets(req)

    sp500 = sp500_set()
    valid_exchanges = {"NYSE", "NASDAQ", "ARCA"}
    candidates: List[Dict] = []
    for a in assets:
        sym = a.symbol
        if "." in sym or "/" in sym or "+" in sym:  # warrants, prefs, units
            continue
        if sym in sp500:
            continue
        exch = a.exchange.value if hasattr(a.exchange, "value") else str(a.exchange)
        if exch not in valid_exchanges:
            continue
        if not a.tradable:
            continue
        candidates.append({
            "ticker": sym,
            "exchange": exch,
            "name": getattr(a, "name", "") or "",
            "shortable": bool(getattr(a, "shortable", False)),
            "marginable": bool(getattr(a, "marginable", False)),
            "fractionable": bool(getattr(a, "fractionable", False)),
        })

    print(f"[A] candidate universe (post-Alpaca filter): {len(candidates)} tickers")
    CANDIDATES_CACHE.write_text(json.dumps(candidates))
    return candidates


# ----------------------------------------------------------------------------
# B) Liquidity (snapshots)
# ----------------------------------------------------------------------------
def fetch_snapshots(symbols: List[str]) -> Dict[str, Dict]:
    """
    Alpaca multi-snapshot endpoint — returns latest trade + bar (with volume)
    per symbol. We compute $-ADV from latest_daily_bar.volume * price.
    Batched 200 symbols/request.
    """
    if SNAPSHOTS_CACHE.exists():
        try:
            cached = json.loads(SNAPSHOTS_CACHE.read_text())
            if isinstance(cached, dict) and cached:
                print(f"[B] using cached snapshots: {len(cached)} tickers")
                return cached
        except Exception:
            pass

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockSnapshotRequest

    key, secret = alpaca_credentials()
    client = StockHistoricalDataClient(api_key=key, secret_key=secret)

    snapshots: Dict[str, Dict] = {}
    BATCH = 200
    n_batches = (len(symbols) + BATCH - 1) // BATCH
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i + BATCH]
        batch_idx = i // BATCH + 1
        try:
            req = StockSnapshotRequest(symbol_or_symbols=batch)
            result = client.get_stock_snapshot(req)
        except Exception as e:
            print(f"  [B] batch {batch_idx}/{n_batches} failed: {e}")
            continue

        for sym, snap in result.items():
            try:
                # daily bar gives the trading day's volume; use prior_daily if today is empty/early
                day_bar = snap.daily_bar or snap.previous_daily_bar
                if day_bar is None:
                    continue
                price = float(day_bar.close) if day_bar.close else 0.0
                volume = float(day_bar.volume) if day_bar.volume else 0.0
                snapshots[sym] = {
                    "price": price,
                    "volume": volume,
                    "dollar_volume": price * volume,
                    "high": float(day_bar.high) if day_bar.high else 0.0,
                    "low": float(day_bar.low) if day_bar.low else 0.0,
                    "timestamp": day_bar.timestamp.isoformat() if day_bar.timestamp else None,
                }
            except Exception:
                continue
        print(f"  [B] batch {batch_idx}/{n_batches} done (running total {len(snapshots)})")

    print(f"[B] snapshots fetched: {len(snapshots)} / {len(symbols)} tickers")
    SNAPSHOTS_CACHE.write_text(json.dumps(snapshots))
    return snapshots


# ----------------------------------------------------------------------------
# B2) Market cap (yfinance)
# ----------------------------------------------------------------------------
def fetch_market_cap_one(ticker: str) -> Optional[Dict]:
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        mcap = info.get("marketCap") if isinstance(info, dict) else getattr(info, "market_cap", None)
        if mcap is None or not mcap:
            return None
        # try to capture sector lazily — too slow for 2000 tickers, do later for filtered set
        return {"ticker": ticker, "market_cap": float(mcap)}
    except Exception:
        return None


def fetch_market_caps(symbols: List[str], max_workers: int = 8) -> Dict[str, float]:
    """Parallel yfinance fast_info lookups. Cached."""
    cache: Dict[str, float] = {}
    if MCAP_CACHE.exists():
        try:
            cache = json.loads(MCAP_CACHE.read_text())
            if isinstance(cache, dict):
                print(f"[B2] loaded mcap cache: {len(cache)} entries")
        except Exception:
            cache = {}

    to_fetch = [s for s in symbols if s not in cache]
    if not to_fetch:
        return cache

    print(f"[B2] fetching market caps for {len(to_fetch)} new tickers...")

    completed = 0
    save_every = 50

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_market_cap_one, s): s for s in to_fetch}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                result = fut.result()
                if result:
                    cache[sym] = result["market_cap"]
            except Exception:
                pass
            completed += 1
            if completed % save_every == 0:
                MCAP_CACHE.write_text(json.dumps(cache))
                print(f"  [B2] progress: {completed}/{len(to_fetch)} (cache size {len(cache)})")

    MCAP_CACHE.write_text(json.dumps(cache))
    print(f"[B2] mcap final: {len(cache)} resolved")
    return cache


# ----------------------------------------------------------------------------
# C) Filter to capacity-floor band
# ----------------------------------------------------------------------------
ADV_MIN = 10_000_000.0    # $10M
ADV_MAX = 30_000_000.0    # $30M
MCAP_MIN = 1_000_000_000.0   # $1B
MCAP_MAX = 10_000_000_000.0  # $10B
TARGET_COHORT = 200


def filter_capacity_floor(
    candidates: List[Dict],
    snapshots: Dict[str, Dict],
    mcaps: Dict[str, float],
) -> List[Dict]:
    rows: List[Dict] = []
    for c in candidates:
        sym = c["ticker"]
        s = snapshots.get(sym)
        m = mcaps.get(sym)
        if not s or m is None:
            continue
        adv = s["dollar_volume"]
        price = s["price"]
        if not (ADV_MIN < adv < ADV_MAX):
            continue
        if not (MCAP_MIN < m < MCAP_MAX):
            continue
        if price < 5.0:   # exclude penny stocks
            continue
        rows.append({
            "ticker": sym,
            "exchange": c["exchange"],
            "name": c["name"],
            "market_cap": round(m, 2),
            "adv_dollars": round(adv, 2),
            "last_price": round(price, 4),
            "shortable": int(c["shortable"]),
            "marginable": int(c["marginable"]),
        })

    rows.sort(key=lambda r: r["adv_dollars"], reverse=True)
    print(f"[C] post-band filter: {len(rows)} tickers (band: ADV ${ADV_MIN/1e6:.0f}M-${ADV_MAX/1e6:.0f}M, MCAP ${MCAP_MIN/1e9:.0f}B-${MCAP_MAX/1e9:.0f}B)")
    return rows[:TARGET_COHORT]


# ----------------------------------------------------------------------------
# Sector enrichment (only for filtered set — much cheaper)
# ----------------------------------------------------------------------------
def enrich_sectors(rows: List[Dict]) -> List[Dict]:
    import yfinance as yf
    print(f"[C2] enriching sectors for {len(rows)} tickers...")

    def _sector_for(ticker: str) -> str:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            return info.get("sector", "") or info.get("industry", "") or ""
        except Exception:
            return ""

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_sector_for, r["ticker"]): r for r in rows}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                r["sector"] = fut.result()
            except Exception:
                r["sector"] = ""
    return rows


# ----------------------------------------------------------------------------
# D) Write cohort CSV
# ----------------------------------------------------------------------------
def write_cohort_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No cohort rows — cannot write empty CSV.")
    cols = ["ticker", "exchange", "sector", "market_cap", "adv_dollars", "last_price",
            "shortable", "marginable", "name"]
    # Make sure all keys present
    for r in rows:
        r.setdefault("sector", "")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"[D] wrote cohort CSV: {path} ({len(rows)} rows)")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def build_cohort() -> List[Dict]:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    # A
    candidates = fetch_candidate_universe()
    candidate_count = len(candidates)

    # B
    syms = [c["ticker"] for c in candidates]
    snapshots = fetch_snapshots(syms)

    # B2 — gate to symbols that passed snapshot (no point in mcap-looking up no-volume names)
    syms_with_vol = [s for s, v in snapshots.items() if v["dollar_volume"] >= ADV_MIN]
    print(f"[B+] post-volume gate: {len(syms_with_vol)} tickers above ${ADV_MIN/1e6:.0f}M $-ADV")
    mcaps = fetch_market_caps(syms_with_vol)

    # C
    cohort = filter_capacity_floor(candidates, snapshots, mcaps)

    # C2 — sector enrich
    cohort = enrich_sectors(cohort)

    # D
    write_cohort_csv(cohort, COHORT_CSV)

    finished = datetime.now(timezone.utc).isoformat()
    STATUS_PATH.write_text(
        f"# capacity_floor_cohort status\n\n"
        f"- started: {started}\n"
        f"- finished: {finished}\n"
        f"- candidate_universe: {candidate_count}\n"
        f"- snapshots_resolved: {len(snapshots)}\n"
        f"- post_volume_gate: {len(syms_with_vol)}\n"
        f"- mcaps_resolved: {len(mcaps)}\n"
        f"- cohort_final: {len(cohort)}\n"
        f"- band_adv: ({ADV_MIN/1e6:.0f}M, {ADV_MAX/1e6:.0f}M)\n"
        f"- band_mcap: ({MCAP_MIN/1e9:.0f}B, {MCAP_MAX/1e9:.0f}B)\n"
        f"- cohort_csv: {COHORT_CSV}\n"
    )

    return cohort


if __name__ == "__main__":
    cohort = build_cohort()
    print()
    print(f"DONE — {len(cohort)} cohort tickers.")
    print("Sample 5:")
    for r in cohort[:5]:
        print(f"  {r['ticker']:6s}  sector={r.get('sector','?'):20s}  "
              f"mcap=${r['market_cap']/1e9:.2f}B  adv=${r['adv_dollars']/1e6:.1f}M  "
              f"px=${r['last_price']:.2f}")
