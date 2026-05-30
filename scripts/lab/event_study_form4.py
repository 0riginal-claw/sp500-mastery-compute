"""CHAMP-003a: Within-ticker Form 4 cluster ≥ p95 event study.

R6 First-Principles "simplest experiment that resolves the question in 24h":
does `form4_insider_cluster_score` (within-ticker top 5%) carry informational
signal — abnormal post-event return separable from zero — independent of any
strategy mechanics?

Pre-registration: AI-Tools/reports/champ_003a_event_study_pre_registration_2026-05-29.md
(sha256 = 8e36bb319b7751e67bf46f32a30ee6a04c5f4d4f31bf38256a0502bbd4821649)

Outputs (UTC-stamped run dir under
    AI-Tools/s&p500-ticker-mastery/data/event_studies/form4_cluster_p95/<utc>/):
  events.csv         — event-level rows (ticker, event_date, score, ar_1d, ar_5d, ar_20d, sector)
  per_ticker.csv     — per-ticker aggregate stats
  per_sector.csv     — per-GICS-sector aggregate stats
  meta.json          — run parameters + coverage + sha of pre-reg

Statistical tests are in the companion results-writer (analyze function below).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import multiprocessing as mp
import os
import sqlite3
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants — locked at pre-registration
# ---------------------------------------------------------------------------

DRIVE_BASE = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
MASTERY_ROOT = f"{DRIVE_BASE}/AI-Tools/s&p500-ticker-mastery"
OHLC_DIR = f"{MASTERY_ROOT}/cache/yfinance_5yr"
EDGAR_DB = "/Volumes/ZG-2TB/zg/edgar_state/index/edgar.db"
OUT_BASE = f"{MASTERY_ROOT}/data/event_studies/form4_cluster_p95"
TMP_BASE = "/Volumes/ZG-2TB/zg/tmp/champ_003a"
PRE_REG_PATH = f"{DRIVE_BASE}/AI-Tools/reports/champ_003a_event_study_pre_registration_2026-05-29.md"

LOOKBACK_D = 5            # form4 cluster window
HORIZONS = (1, 5, 20)     # trading-day forward horizons
BETA_LOOKBACK = 60        # bars for OLS beta estimation
BETA_MIN_BARS = 30        # min valid pre-event returns; else fallback beta=1
PCTILE = 0.95             # within-ticker p95 threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("champ_003a")


# ---------------------------------------------------------------------------
# Form 4 P-buy data — direct SQL (more reliable than altdata path)
# ---------------------------------------------------------------------------

def fetch_form4_pbuys(ticker: str) -> pd.DataFrame:
    """Return DataFrame of open-market PURCHASE events for a ticker.

    Columns: filed_at (Timestamp), insider_cik (str), insider_name (str),
             shares (float), value (float).
    Filters: code='P', direction='A' (acquisition), is_derivative=0.
    Empty DataFrame if no rows or DB unreachable.
    """
    sql = (
        "SELECT filed_at, insider_cik, insider_name, shares, value "
        "FROM form4_transactions "
        "WHERE issuer_ticker = ? "
        "AND code = 'P' AND direction = 'A' AND is_derivative = 0 "
        "ORDER BY filed_at"
    )
    try:
        with sqlite3.connect(f"file:{EDGAR_DB}?mode=ro", uri=True, timeout=10.0) as con:
            rows = con.execute(sql, (ticker.upper(),)).fetchall()
    except Exception as e:
        logger.warning("form4 fetch failed for %s: %s", ticker, e)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["filed_at", "insider_cik", "insider_name", "shares", "value"])
    df["filed_at"] = pd.to_datetime(df["filed_at"], errors="coerce")
    df = df.dropna(subset=["filed_at"]).sort_values("filed_at").reset_index(drop=True)
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    return df


def compute_cluster_score(bar_dates: pd.DatetimeIndex, pbuys: pd.DataFrame,
                          lookback_d: int = LOOKBACK_D) -> pd.Series:
    """For each bar date t, cluster_score = n_distinct_insiders *
    log(1 + total_value_bought) over P-buys with filed_at in (t - lookback_d, t].

    Vectorized via searchsorted + cumsum (same trick as altdata module).
    Distinct-insider count is per-bar (small inner loop, bounded by window).
    """
    out = pd.Series(0.0, index=bar_dates, dtype="float64")
    if pbuys.empty:
        return out

    event_times = pbuys["filed_at"].values.astype("datetime64[ns]")
    event_values = pbuys["value"].to_numpy(dtype="float64")
    insider_codes, _ = pd.factorize(pbuys["insider_cik"].astype(str), sort=False)
    insider_codes = insider_codes.astype(np.int64)

    bar_arr = bar_dates.values.astype("datetime64[ns]")
    window_ns = np.timedelta64(int(lookback_d), "D")
    upper = np.searchsorted(event_times, bar_arr, side="right")
    lower = np.searchsorted(event_times, bar_arr - window_ns, side="right")

    cum_val = np.concatenate(([0.0], np.cumsum(event_values)))
    sum_window = cum_val[upper] - cum_val[lower]

    n_insider = np.zeros(len(bar_arr), dtype="float64")
    nonempty = np.flatnonzero(upper > lower)
    for i in nonempty:
        lo, hi = lower[i], upper[i]
        n_insider[i] = float(len(np.unique(insider_codes[lo:hi])))

    score = n_insider * np.log1p(np.clip(sum_window, 0.0, None))
    return pd.Series(score, index=bar_dates, dtype="float64")


# ---------------------------------------------------------------------------
# Price + benchmark loaders
# ---------------------------------------------------------------------------

_SPY_CACHE: Dict[str, pd.DataFrame] = {}


def _load_ohlc(ticker: str) -> Optional[pd.DataFrame]:
    path = f"{OHLC_DIR}/{ticker}.parquet"
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.warning("ohlc read failed for %s: %s", ticker, e)
        return None
    if df.empty or "close" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _load_spy() -> Optional[pd.DataFrame]:
    if "SPY" in _SPY_CACHE:
        return _SPY_CACHE["SPY"]
    df = _load_ohlc("SPY")
    if df is not None:
        _SPY_CACHE["SPY"] = df
    return df


# ---------------------------------------------------------------------------
# Per-ticker event extraction
# ---------------------------------------------------------------------------

def extract_events_one(ticker: str, sector_map: Dict[str, str],
                       spy_close: pd.Series) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (events_list, ticker_meta)."""
    meta: Dict[str, Any] = {
        "ticker": ticker, "ohlc_bars": 0, "pbuys": 0, "events_above_p95": 0,
        "sector": sector_map.get(ticker, "Unknown"), "status": "ok",
    }

    ohlc = _load_ohlc(ticker)
    if ohlc is None or ohlc.empty:
        meta["status"] = "no_ohlc"
        return [], meta
    meta["ohlc_bars"] = len(ohlc)

    pbuys = fetch_form4_pbuys(ticker)
    meta["pbuys"] = len(pbuys)
    if pbuys.empty:
        meta["status"] = "no_pbuys"
        return [], meta

    bar_idx = ohlc.index
    score = compute_cluster_score(bar_idx, pbuys, lookback_d=LOOKBACK_D)
    positive = score[score > 0]
    if positive.empty:
        meta["status"] = "no_positive_score"
        return [], meta

    threshold = float(np.quantile(positive.values, PCTILE))
    # Edge case: if all positive scores equal, threshold = max, so >= still picks all
    event_mask = (score >= threshold) & (score > 0)
    event_dates = bar_idx[event_mask]
    if len(event_dates) == 0:
        meta["status"] = "no_events_above_threshold"
        return [], meta

    # Pre-compute daily returns for ticker + SPY
    close = ohlc["close"].astype(float)
    r_t = close.pct_change()
    spy_close_aligned = spy_close.reindex(close.index).ffill()
    r_spy = spy_close_aligned.pct_change()

    events: List[Dict[str, Any]] = []
    bar_positions = {d: i for i, d in enumerate(close.index)}
    closes = close.values
    spy_vals = spy_close_aligned.values
    rt_arr = r_t.values
    rspy_arr = r_spy.values
    n = len(closes)

    for ev_date in event_dates:
        i = bar_positions.get(ev_date)
        if i is None:
            continue

        # Beta estimation: prior BETA_LOOKBACK bars of joint daily returns
        lo = max(0, i - BETA_LOOKBACK)
        rt_win = rt_arr[lo:i]
        rspy_win = rspy_arr[lo:i]
        # mask NaN
        mask = ~(np.isnan(rt_win) | np.isnan(rspy_win))
        n_valid = int(mask.sum())
        beta_fallback = False
        if n_valid < BETA_MIN_BARS:
            beta = 1.0
            beta_fallback = True
        else:
            rt_w = rt_win[mask]
            rs_w = rspy_win[mask]
            var_s = float(np.var(rs_w))
            if var_s <= 0:
                beta = 1.0
                beta_fallback = True
            else:
                cov = float(np.mean((rt_w - rt_w.mean()) * (rs_w - rs_w.mean())))
                beta = cov / var_s
                if not np.isfinite(beta):
                    beta = 1.0
                    beta_fallback = True

        row: Dict[str, Any] = {
            "ticker": ticker,
            "event_date": ev_date.strftime("%Y-%m-%d"),
            "score": float(score.loc[ev_date]),
            "score_pctile_within_ticker": float((positive < score.loc[ev_date]).mean()),
            "threshold_p95": threshold,
            "sector": meta["sector"],
            "beta_60d": float(beta),
            "beta_fallback": beta_fallback,
        }

        for h in HORIZONS:
            fwd_i = i + h
            if fwd_i >= n:
                row[f"ar_{h}d"] = np.nan
                row[f"r_t_{h}d"] = np.nan
                row[f"r_spy_{h}d"] = np.nan
                continue
            c0 = closes[i]
            ch = closes[fwd_i]
            s0 = spy_vals[i]
            sh = spy_vals[fwd_i]
            if not (np.isfinite(c0) and np.isfinite(ch) and np.isfinite(s0) and np.isfinite(sh)
                    and c0 > 0 and s0 > 0):
                row[f"ar_{h}d"] = np.nan
                row[f"r_t_{h}d"] = np.nan
                row[f"r_spy_{h}d"] = np.nan
                continue
            r_t_h = ch / c0 - 1.0
            r_spy_h = sh / s0 - 1.0
            ar_h = r_t_h - beta * r_spy_h
            row[f"ar_{h}d"] = float(ar_h)
            row[f"r_t_{h}d"] = float(r_t_h)
            row[f"r_spy_{h}d"] = float(r_spy_h)

        events.append(row)

    meta["events_above_p95"] = len(events)
    return events, meta


# ---------------------------------------------------------------------------
# Sector map — yfinance Ticker.get_info (cached)
# ---------------------------------------------------------------------------

def build_sector_map(tickers: List[str], cache_path: str) -> Dict[str, str]:
    """Fetch GICS sector per ticker via yfinance, cached to JSON."""
    cache: Dict[str, str] = {}
    if os.path.exists(cache_path):
        try:
            cache = json.loads(Path(cache_path).read_text())
        except Exception:
            cache = {}
    missing = [t for t in tickers if t not in cache]
    if not missing:
        return cache

    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available; sector map empty")
        return cache

    for t in missing:
        try:
            info = yf.Ticker(t).get_info()
            cache[t] = info.get("sector") or "Unknown"
        except Exception as e:
            logger.debug("sector lookup failed for %s: %s", t, e)
            cache[t] = "Unknown"
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cache_path).write_text(json.dumps(cache, indent=2, sort_keys=True))
    return cache


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

# Globals set in worker init
_W_SECTOR_MAP: Dict[str, str] = {}
_W_SPY_CLOSE: Optional[pd.Series] = None


def _worker_init(sector_map: Dict[str, str]):
    global _W_SECTOR_MAP, _W_SPY_CLOSE
    _W_SECTOR_MAP = sector_map
    spy = _load_spy()
    _W_SPY_CLOSE = spy["close"].astype(float) if spy is not None else None


def _worker_one(ticker: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if _W_SPY_CLOSE is None:
        return [], {"ticker": ticker, "status": "no_spy", "pbuys": 0,
                    "events_above_p95": 0, "sector": "Unknown", "ohlc_bars": 0}
    try:
        return extract_events_one(ticker, _W_SECTOR_MAP, _W_SPY_CLOSE)
    except Exception as e:
        logger.warning("worker failed for %s: %s", ticker, e)
        return [], {"ticker": ticker, "status": f"error: {e}", "pbuys": 0,
                    "events_above_p95": 0, "sector": "Unknown", "ohlc_bars": 0}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def get_universe() -> List[str]:
    sys.path.insert(0, f"{MASTERY_ROOT}/scripts")
    from lab.knowledge.inventory import mastered
    return list(mastered())


def run_study(tickers: Optional[List[str]] = None, n_workers: int = 4,
              smoke_only: bool = False) -> Path:
    utc = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(OUT_BASE) / utc
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(TMP_BASE).mkdir(parents=True, exist_ok=True)

    if tickers is None:
        tickers = get_universe()
    if smoke_only:
        tickers = ["AAPL"]

    logger.info("CHAMP-003a starting: %d tickers, out_dir=%s", len(tickers), out_dir)

    sector_cache = f"{MASTERY_ROOT}/cache/sector_map_yf.json"
    sector_map = build_sector_map(tickers, sector_cache)

    # Warmup SPY in parent process
    spy = _load_spy()
    if spy is None:
        raise SystemExit("SPY parquet missing — cannot compute abnormal returns")

    all_events: List[Dict[str, Any]] = []
    all_meta: List[Dict[str, Any]] = []
    t0 = time.time()
    completed = 0

    if n_workers <= 1:
        _worker_init(sector_map)
        for t in tickers:
            evs, meta = _worker_one(t)
            all_events.extend(evs)
            all_meta.append(meta)
            completed += 1
            if completed % 25 == 0:
                logger.info("  %d / %d done in %.1fs", completed, len(tickers), time.time() - t0)
    else:
        with mp.Pool(processes=n_workers, initializer=_worker_init,
                     initargs=(sector_map,)) as pool:
            for evs, meta in pool.imap_unordered(_worker_one, tickers, chunksize=4):
                all_events.extend(evs)
                all_meta.append(meta)
                completed += 1
                if completed % 25 == 0:
                    logger.info("  %d / %d done in %.1fs", completed, len(tickers), time.time() - t0)

    logger.info("All workers done in %.1fs; %d events from %d tickers",
                time.time() - t0, len(all_events), len(all_meta))

    # Save events.csv
    if all_events:
        ev_df = pd.DataFrame(all_events)
    else:
        ev_df = pd.DataFrame(columns=["ticker", "event_date", "score", "sector",
                                       "ar_1d", "ar_5d", "ar_20d"])
    ev_df.to_csv(out_dir / "events.csv", index=False)
    logger.info("wrote %s (%d rows)", out_dir / "events.csv", len(ev_df))

    # Per-ticker meta
    meta_df = pd.DataFrame(all_meta)
    meta_df.to_csv(out_dir / "ticker_meta.csv", index=False)

    # Per-ticker stats
    if not ev_df.empty:
        per_ticker = (
            ev_df.groupby("ticker")
            .agg(
                n_events=("ticker", "size"),
                mean_ar_1d=("ar_1d", "mean"),
                mean_ar_5d=("ar_5d", "mean"),
                mean_ar_20d=("ar_20d", "mean"),
                std_ar_5d=("ar_5d", "std"),
                sector=("sector", "first"),
            )
            .reset_index()
        )
        # t-stats per ticker
        per_ticker["t_ar_5d"] = per_ticker.apply(
            lambda r: (r["mean_ar_5d"] / (r["std_ar_5d"] / np.sqrt(r["n_events"])))
            if r["n_events"] > 1 and r["std_ar_5d"] and r["std_ar_5d"] > 0
            else np.nan, axis=1,
        )
        per_ticker.to_csv(out_dir / "per_ticker.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "per_ticker.csv", index=False)

    # Per-sector stats
    if not ev_df.empty:
        per_sector = (
            ev_df.groupby("sector")
            .agg(
                n_events=("sector", "size"),
                n_tickers=("ticker", "nunique"),
                mean_ar_1d=("ar_1d", "mean"),
                mean_ar_5d=("ar_5d", "mean"),
                mean_ar_20d=("ar_20d", "mean"),
                std_ar_5d=("ar_5d", "std"),
            )
            .reset_index()
        )
        per_sector["t_ar_5d"] = per_sector.apply(
            lambda r: (r["mean_ar_5d"] / (r["std_ar_5d"] / np.sqrt(r["n_events"])))
            if r["n_events"] > 1 and r["std_ar_5d"] and r["std_ar_5d"] > 0
            else np.nan, axis=1,
        )
        per_sector.to_csv(out_dir / "per_sector.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "per_sector.csv", index=False)

    # Meta JSON
    pre_reg_sha = hashlib.sha256(Path(PRE_REG_PATH).read_bytes()).hexdigest()
    meta_json = {
        "utc_start": utc,
        "n_tickers_attempted": len(tickers),
        "n_tickers_with_events": int((meta_df["events_above_p95"] > 0).sum()) if not meta_df.empty else 0,
        "n_tickers_with_pbuys": int((meta_df["pbuys"] > 0).sum()) if not meta_df.empty else 0,
        "n_events_total": len(ev_df),
        "lookback_d": LOOKBACK_D,
        "horizons": list(HORIZONS),
        "beta_lookback": BETA_LOOKBACK,
        "beta_min_bars": BETA_MIN_BARS,
        "pctile_threshold": PCTILE,
        "pre_reg_path": PRE_REG_PATH,
        "pre_reg_sha256": pre_reg_sha,
        "edgar_db": EDGAR_DB,
        "ohlc_dir": OHLC_DIR,
        "elapsed_s": round(time.time() - t0, 2),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta_json, indent=2))
    logger.info("meta: %s", json.dumps(meta_json, indent=2))

    return out_dir


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def analyze(run_dir: Path) -> Dict[str, Any]:
    """Run the pre-registered statistical tests against events.csv in run_dir.

    Returns a dict suitable for serialization to JSON / report.
    """
    ev = pd.read_csv(run_dir / "events.csv")
    if ev.empty:
        return {"status": "INSUFFICIENT_DATA", "reason": "no events", "n_events": 0}

    res: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "n_events_total": len(ev),
        "n_tickers_with_events": int(ev["ticker"].nunique()),
    }

    # ----- Aggregate test (cluster-robust by sector) -----
    ar5 = ev["ar_5d"].dropna()
    n_ar5 = len(ar5)
    res["n_events_with_ar5"] = n_ar5
    if n_ar5 < 30:
        res["aggregate"] = {"status": "INSUFFICIENT", "n": n_ar5}
    else:
        mean_ar5 = float(ar5.mean())
        # Standard SE (pooled)
        std_ar5 = float(ar5.std(ddof=1))
        se_pooled = std_ar5 / np.sqrt(n_ar5)
        t_pooled = mean_ar5 / se_pooled if se_pooled > 0 else float("nan")

        # Cluster-robust SE: clusters = sectors
        sector_means = ev.dropna(subset=["ar_5d"]).groupby("sector")["ar_5d"]
        # Per-cluster aggregates: mean and n
        per_cluster = sector_means.agg(["mean", "count", "sum", lambda x: float(np.var(x, ddof=0)) * len(x)])
        per_cluster.columns = ["mean", "n", "sum_x", "var_n"]
        # CR1 cluster-robust SE estimator for mean:
        # Var_cluster(mean) = (1/N^2) * sum_g (sum_g (x - mean))^2
        # Equivalent: residuals per obs r_i = x_i - mean; cluster sum S_g = sum_g r_i;
        # Var = (G/(G-1)) * (N/(N-1)) * sum_g S_g^2 / N^2  (CR1 adjustment)
        N = n_ar5
        merged = ev.dropna(subset=["ar_5d"]).copy()
        merged["resid"] = merged["ar_5d"] - mean_ar5
        cluster_S = merged.groupby("sector")["resid"].sum()
        G = int(len(cluster_S))
        if G < 2:
            t_cluster = float("nan")
            se_cluster = float("nan")
            p_cluster = float("nan")
        else:
            sum_S_sq = float((cluster_S ** 2).sum())
            cr1_adj = (G / (G - 1)) * (N / (N - 1)) if N > 1 else 1.0
            var_cluster = cr1_adj * sum_S_sq / (N * N)
            se_cluster = float(np.sqrt(var_cluster))
            t_cluster = mean_ar5 / se_cluster if se_cluster > 0 else float("nan")
            # p-value via t-distribution with df = G - 1
            try:
                from scipy.stats import t as t_dist
                p_cluster = float(2 * (1 - t_dist.cdf(abs(t_cluster), df=G - 1)))
            except ImportError:
                # normal approximation fallback
                from math import erf, sqrt
                p_cluster = float(2 * (1 - 0.5 * (1 + erf(abs(t_cluster) / sqrt(2)))))

        res["aggregate"] = {
            "n": n_ar5,
            "mean_ar_5d": mean_ar5,
            "std_ar_5d": std_ar5,
            "se_pooled": float(se_pooled),
            "t_pooled": float(t_pooled),
            "se_cluster_robust_by_sector": float(se_cluster),
            "t_cluster_robust": float(t_cluster),
            "p_two_sided_cluster": float(p_cluster),
            "n_sector_clusters": G,
        }

        # ar_1d and ar_20d (pooled t only, secondary)
        for h in (1, 20):
            arh = ev[f"ar_{h}d"].dropna()
            if len(arh) >= 30:
                m = float(arh.mean())
                s = float(arh.std(ddof=1))
                se = s / np.sqrt(len(arh)) if len(arh) > 1 else float("nan")
                t = m / se if se and se > 0 else float("nan")
                res[f"aggregate_ar_{h}d"] = {"n": int(len(arh)), "mean": m, "se_pooled": se, "t_pooled": t}

    # ----- Per-ticker Bonferroni -----
    per_ticker = pd.read_csv(run_dir / "per_ticker.csv")
    if not per_ticker.empty:
        tested = per_ticker[per_ticker["n_events"] >= 5].copy()
        N_tested = int(len(tested))
        res["per_ticker_n_tested"] = N_tested
        if N_tested > 0:
            alpha = 0.05 / N_tested
            try:
                from scipy.stats import t as t_dist
                # two-sided p from t_ar_5d
                tested["p_two_sided"] = tested.apply(
                    lambda r: float(2 * (1 - t_dist.cdf(abs(r["t_ar_5d"]), df=max(r["n_events"] - 1, 1))))
                    if pd.notna(r["t_ar_5d"]) else float("nan"), axis=1)
            except ImportError:
                from math import erf, sqrt
                tested["p_two_sided"] = tested.apply(
                    lambda r: float(2 * (1 - 0.5 * (1 + erf(abs(r["t_ar_5d"]) / sqrt(2)))))
                    if pd.notna(r["t_ar_5d"]) else float("nan"), axis=1)
            tested["bonferroni_alpha"] = alpha
            tested["survives"] = (
                (tested["p_two_sided"] < alpha) & (tested["mean_ar_5d"] > 0)
            )
            tested.to_csv(run_dir / "per_ticker_bonferroni.csv", index=False)
            survivors = tested[tested["survives"]].copy()
            res["per_ticker_bonferroni_alpha"] = float(alpha)
            res["per_ticker_survivors_n"] = int(len(survivors))
            res["per_ticker_survivors"] = survivors[["ticker", "n_events", "mean_ar_5d",
                                                       "t_ar_5d", "p_two_sided", "sector"]].to_dict(orient="records")

    # ----- Per-sector subset -----
    per_sector = pd.read_csv(run_dir / "per_sector.csv")
    if not per_sector.empty:
        sec_pass = per_sector[(per_sector["t_ar_5d"] > 2.5) & (per_sector["mean_ar_5d"] > 0)].copy()
        res["per_sector_winners_t_gt_2_5"] = sec_pass.to_dict(orient="records")

    # ----- Subset: top decile (within-ticker p99+) -----
    if "score_pctile_within_ticker" in ev.columns:
        top_dec = ev[ev["score_pctile_within_ticker"] >= 0.99].dropna(subset=["ar_5d"])
        mid_dec = ev[(ev["score_pctile_within_ticker"] >= 0.95)
                     & (ev["score_pctile_within_ticker"] < 0.99)].dropna(subset=["ar_5d"])
        for name, grp in [("p99_plus", top_dec), ("p95_to_p99", mid_dec)]:
            if len(grp) >= 30:
                m = float(grp["ar_5d"].mean())
                s = float(grp["ar_5d"].std(ddof=1))
                se = s / np.sqrt(len(grp))
                t = m / se if se > 0 else float("nan")
                res[f"subset_score_{name}"] = {
                    "n": int(len(grp)), "mean_ar_5d": m, "se_pooled": float(se),
                    "t_pooled": float(t),
                }

    # ----- Subset: many events vs few -----
    if not per_ticker.empty:
        many = per_ticker[per_ticker["n_events"] >= 20]["ticker"].tolist()
        few = per_ticker[per_ticker["n_events"] < 20]["ticker"].tolist()
        for name, t_list in [("ge20_events", many), ("lt20_events", few)]:
            grp = ev[ev["ticker"].isin(t_list)].dropna(subset=["ar_5d"])
            if len(grp) >= 30:
                m = float(grp["ar_5d"].mean())
                s = float(grp["ar_5d"].std(ddof=1))
                se = s / np.sqrt(len(grp))
                t = m / se if se > 0 else float("nan")
                res[f"subset_event_count_{name}"] = {
                    "n": int(len(grp)), "mean_ar_5d": m, "se_pooled": float(se),
                    "t_pooled": float(t),
                }

    # ----- Verdict (mechanical, locked criteria) -----
    res["verdict"] = render_verdict(res)
    (run_dir / "analysis.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def render_verdict(res: Dict[str, Any]) -> Dict[str, Any]:
    """Apply pre-reg pass/fail rules. Return {outcome, rationale}."""
    n = res.get("n_events_total", 0)
    n_with_ar5 = res.get("n_events_with_ar5", 0)
    n_tickers_ev = res.get("n_tickers_with_events", 0)
    n_tickers_tested = res.get("per_ticker_n_tested", 0)

    if n_with_ar5 < 100 or n_tickers_tested < 20:
        return {
            "outcome": "D_INCONCLUSIVE_INSUFFICIENT_DATA",
            "rationale": (
                f"n_events_with_ar5={n_with_ar5} (<100) or n_tickers_with>=5_events={n_tickers_tested} (<20). "
                f"Data is too sparse for the pre-registered tests to be reliable. "
                f"This dominates A/B/C per pre-reg §6."
            ),
        }

    agg = res.get("aggregate", {})
    t_cluster = agg.get("t_cluster_robust", float("nan"))
    p_cluster = agg.get("p_two_sided_cluster", float("nan"))
    mean_ar = agg.get("mean_ar_5d", float("nan"))
    survivors = res.get("per_ticker_survivors_n", 0)
    sector_winners = res.get("per_sector_winners_t_gt_2_5", []) or []
    subset_score = res.get("subset_score_p99_plus", {})
    subset_count = res.get("subset_event_count_ge20_events", {})
    subset_winners = []
    for s in (subset_score, subset_count):
        if s and isinstance(s, dict) and s.get("t_pooled", 0) > 2.5 and s.get("mean_ar_5d", 0) > 0:
            subset_winners.append(s)

    # A. Universal
    try:
        if (not np.isnan(t_cluster) and t_cluster > 2.5 and not np.isnan(p_cluster)
                and p_cluster < 0.05 and not np.isnan(mean_ar) and mean_ar > 0
                and survivors >= 3):
            return {
                "outcome": "A_SIGNAL_CONFIRMED_UNIVERSAL",
                "rationale": (
                    f"Aggregate t_cluster={t_cluster:.2f} (>2.5), p={p_cluster:.4f} (<0.05), "
                    f"mean_ar_5d={mean_ar:+.4f} (>0), Bonferroni survivors={survivors} (>=3)."
                ),
            }
    except Exception:
        pass

    # B. Capacity-floor subset
    if survivors >= 1 or sector_winners or subset_winners:
        return {
            "outcome": "B_CAPACITY_FLOOR_CANDIDATE",
            "rationale": (
                f"Aggregate is insufficient OR Bonferroni survivors={survivors}, "
                f"sector_winners(t>2.5)={len(sector_winners)}, subset_winners={len(subset_winners)}. "
                f"At least one subset clears t>2.5 → propose CHAMP-003b on that subset."
            ),
            "survivors_n": survivors,
            "sector_winners": sector_winners,
            "subset_winners": subset_winners,
        }

    # C. Empirically absent
    try:
        if not np.isnan(t_cluster) and abs(t_cluster) < 2.0 and survivors == 0:
            return {
                "outcome": "C_SIGNAL_EMPIRICALLY_ABSENT",
                "rationale": (
                    f"Aggregate |t_cluster|={abs(t_cluster):.2f} (<2.0), zero Bonferroni survivors, "
                    f"zero sector winners, zero subset winners → form4_cluster within-ticker p95 "
                    f"signal not present at S&P 500 large-cap timeframe."
                ),
            }
    except Exception:
        pass

    return {
        "outcome": "B_CAPACITY_FLOOR_CANDIDATE_OR_AMBIGUOUS",
        "rationale": (
            f"Aggregate t_cluster={t_cluster}, survivors={survivors}, but does not cleanly hit A or C; "
            f"reporting as ambiguous capacity-floor candidate pending review."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Run AAPL only (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--analyze-only", type=str, default=None,
                    help="Skip event extraction; analyze given run_dir UTC stamp")
    args = ap.parse_args()

    if args.analyze_only:
        run_dir = Path(OUT_BASE) / args.analyze_only
        if not run_dir.exists():
            logger.error("run_dir not found: %s", run_dir)
            return 1
        res = analyze(run_dir)
        print(json.dumps(res, indent=2, default=str))
        return 0

    run_dir = run_study(n_workers=args.workers, smoke_only=args.smoke)
    res = analyze(run_dir)
    print(json.dumps(res.get("verdict", {}), indent=2))
    print(f"\nFull analysis: {run_dir / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
