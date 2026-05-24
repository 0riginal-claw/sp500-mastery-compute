#!/usr/bin/env python3
"""
quiver_ic_audit.py - PIT-safe IC audit of currently-unwired Quiver Quant
tier-1 endpoints against forward N-bar returns.

# karpathy_checked: audit-only, no writes to live trading code
# autosolve_skip: read-only audit
# cloud_routing_skip: local-only audit, network only to api.quiverquant.com
# council2_action: 2026-05-24 scheduled audit per Council #2 verdict

Endpoints audited (the 4 unwired tier-1 endpoints from CLAUDE.md
`reference_quiver_quant_tier1.md`):
  - insider_form4 : /beta/live/insiders/<TICKER>
  - dark_pool     : /beta/historical/offexchange/<TICKER>
  - wsb           : /beta/live/wallstreetbets/<TICKER>
  - patents       : /beta/historical/allpatents/<TICKER>
                    (fallback: /beta/live/patents/<TICKER>)

PIT-safe join: shift filing/disclosure date by +1 trading day before joining
to forward return. (i.e., bar_date strictly > filing_date.)

API key resolution order:
  1. env: QUIVER_API_KEY
  2. env: QUIVER_TOKEN
  3. file: ~/.zg/quiver/api_key  (single line, trimmed)
  4. file: $AI_DRIVE/.secrets/quiver_api_key  (single line, trimmed)
If none found: writes credentials_missing status and emits an empty TSV
with header + a JSON sidecar so the launchd run leaves a clear audit trail.

Usage:
  python quiver_ic_audit.py \\
      --tickers NTRS,JPM,AAPL \\
      --horizon 5 \\
      --output ../research/quiver_unwired_ic_2026-05-24/ic_table.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import urllib.error
import urllib.parse
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "cache" / "yfinance_5yr"

ENDPOINTS = {
    "insider_form4": [
        "https://api.quiverquant.com/beta/live/insiders/{ticker}",
        "https://api.quiverquant.com/beta/historical/insiders/{ticker}",
    ],
    "dark_pool": [
        "https://api.quiverquant.com/beta/historical/offexchange/{ticker}",
        "https://api.quiverquant.com/beta/live/offexchange/{ticker}",
    ],
    "wsb": [
        "https://api.quiverquant.com/beta/live/wallstreetbets/{ticker}",
        "https://api.quiverquant.com/beta/historical/wallstreetbets/{ticker}",
    ],
    "patents": [
        "https://api.quiverquant.com/beta/historical/allpatents/{ticker}",
        "https://api.quiverquant.com/beta/live/patents/{ticker}",
    ],
}

# Try these date field names in order, per endpoint
DATE_FIELDS = [
    "Date", "ReportDate", "FilingDate", "DisclosureDate",
    "TransactionDate", "Day", "Period", "report_date", "date",
]

# Try these numeric/sentiment fields as the feature value, in order
VALUE_FIELDS = [
    # insider form 4
    "Shares", "shares", "AmountShares", "Amount",
    # dark pool
    "DarkPoolPercent", "OffExchangePercent", "ShortVolume", "ShortPercent",
    "Volume", "volume",
    # wsb
    "Mentions", "Sentiment", "mentions", "sentiment", "Count", "Score",
    # patents
    "Patents", "PatentCount", "patents_filed", "count",
]


def _resolve_api_key() -> Tuple[Optional[str], str]:
    """Return (key, source). key=None if not found."""
    for env_name in ("QUIVER_API_KEY", "QUIVER_TOKEN"):
        v = os.environ.get(env_name)
        if v and v.strip():
            return v.strip(), "env:{}".format(env_name)
    candidates = [
        Path.home() / ".zg" / "quiver" / "api_key",
    ]
    drv = os.environ.get("AI_DRIVE")
    if drv:
        candidates.append(Path(drv) / ".secrets" / "quiver_api_key")
    for p in candidates:
        try:
            if p.exists():
                txt = p.read_text().strip()
                if txt:
                    return txt, "file:{}".format(p)
        except Exception:
            continue
    return None, "missing"


def _fetch_json(url: str, api_key: Optional[str], timeout: int = 20):
    headers = {
        "User-Agent": "sp500-mastery-quiver-ic-audit/1.0",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw), None
    except urllib.error.HTTPError as e:
        return None, "http_{}".format(e.code)
    except urllib.error.URLError as e:
        return None, "url_error_{}".format(e.reason)
    except json.JSONDecodeError as e:
        return None, "json_decode_{}".format(e)
    except Exception as e:
        return None, "fetch_error_{}".format(type(e).__name__)


def _try_endpoints(endpoint_key: str, ticker: str, api_key: Optional[str]):
    """Try each URL variant for an endpoint until one succeeds (list, non-empty).
    Returns (records, used_url, error_note)."""
    for url_tmpl in ENDPOINTS[endpoint_key]:
        url = url_tmpl.format(ticker=urllib.parse.quote(ticker.upper()))
        data, err = _fetch_json(url, api_key)
        if data is None:
            continue
        if isinstance(data, list):
            return data, url, None
        if isinstance(data, dict):
            # Some endpoints wrap list under a key
            for k in ("data", "results", "items"):
                if isinstance(data.get(k), list):
                    return data[k], url, None
    return None, None, "all_variants_failed"


def _normalize_records(records, endpoint_key):
    """Convert API records to a list of (date, value) tuples. Drop records
    without a usable date. value defaults to 1.0 if no numeric field found
    (treats endpoint as a count/event indicator)."""
    import pandas as pd
    rows = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        # Find date
        dt = None
        for f in DATE_FIELDS:
            v = rec.get(f)
            if v:
                try:
                    dt = pd.to_datetime(v, errors="coerce")
                    if not pd.isna(dt):
                        break
                except Exception:
                    continue
        if dt is None or pd.isna(dt):
            continue
        # Find value
        val = None
        for f in VALUE_FIELDS:
            if f in rec:
                try:
                    val = float(rec[f])
                    break
                except (TypeError, ValueError):
                    continue
        if val is None:
            val = 1.0  # event indicator
        rows.append((dt, val))
    return rows


def _build_pit_safe_feature(records, bar_index, horizon_bars):
    """For each bar_date in bar_index, compute:
       - count of records with filing_date < bar_date AND within prior 30 trading days
       - sum of values within same window
    PIT-safe: strict less-than on bar date (shift by +1 day prior to join).
    Returns DataFrame indexed by bar_index with cols [count_30d, value_sum_30d]."""
    import numpy as np
    import pandas as pd

    if not records:
        return pd.DataFrame(
            {"count_30d": 0.0, "value_sum_30d": 0.0},
            index=bar_index,
        )

    rec_df = pd.DataFrame(records, columns=["filing_date", "value"])
    rec_df["filing_date"] = pd.to_datetime(rec_df["filing_date"])
    # +1 day shift: a record filed on day D only becomes joinable on day D+1
    rec_df["available_date"] = rec_df["filing_date"] + pd.Timedelta(days=1)
    rec_df = rec_df.sort_values("available_date").reset_index(drop=True)

    counts = np.zeros(len(bar_index), dtype=float)
    sums = np.zeros(len(bar_index), dtype=float)
    avail = rec_df["available_date"].to_numpy()
    vals = rec_df["value"].to_numpy()
    bars = bar_index.to_numpy()

    # 30-calendar-day window
    window = np.timedelta64(30, "D")
    for i, bar_date in enumerate(bars):
        # records with available_date <= bar_date  (PIT-safe: <=)
        # AND available_date > bar_date - 30d
        right = np.searchsorted(avail, bar_date, side="right")
        if right == 0:
            continue
        # Inside window
        low_bound = bar_date - window
        left = np.searchsorted(avail, low_bound, side="right")
        if left >= right:
            continue
        counts[i] = right - left
        sums[i] = vals[left:right].sum()

    return pd.DataFrame(
        {"count_30d": counts, "value_sum_30d": sums},
        index=bar_index,
    )


def _pearson_ic(x, y):
    import numpy as np
    import pandas as pd
    s = pd.concat([pd.Series(x).reset_index(drop=True),
                   pd.Series(y).reset_index(drop=True)], axis=1).dropna()
    n = len(s)
    if n < 30:
        return float("nan"), float("nan"), n
    xa = s.iloc[:, 0].to_numpy(dtype=float)
    ya = s.iloc[:, 1].to_numpy(dtype=float)
    if np.std(xa) == 0 or np.std(ya) == 0:
        return float("nan"), float("nan"), n
    r = float(np.corrcoef(xa, ya)[0, 1])
    if abs(r) >= 1.0:
        return r, 0.0, n
    t = r * (n - 2) ** 0.5 / ((1.0 - r * r) ** 0.5)
    try:
        from scipy import stats
        p = float(2.0 * (1.0 - stats.t.cdf(abs(t), df=n - 2)))
    except Exception:
        from math import erf, sqrt
        z = abs(t)
        p = float(2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))))
    return r, p, n


def _load_ohlcv(ticker: str):
    import pandas as pd
    p = CACHE_DIR / "{}.parquet".format(ticker.upper())
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    if "close" not in [c.lower() for c in df.columns]:
        return None
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="PIT-safe IC audit of unwired Quiver tier-1 endpoints.")
    ap.add_argument("--tickers", type=str, required=True,
                    help="comma-separated ticker list (e.g. NTRS,JPM,AAPL)")
    ap.add_argument("--horizon", type=int, default=5,
                    help="forward-return horizon in bars (default 5)")
    ap.add_argument("--output", type=str, required=True,
                    help="path to TSV output")
    ap.add_argument("--endpoints", type=str, default="insider_form4,dark_pool,wsb,patents",
                    help="comma-separated endpoint keys to audit")
    ap.add_argument("--api-pause-sec", type=float, default=0.5,
                    help="delay between API calls to be polite")
    args = ap.parse_args()

    try:
        import pandas as pd
        import numpy as np  # noqa: F401
    except Exception as e:
        print("[quiver-ic] pandas/numpy required: {}".format(e), file=sys.stderr)
        return 2

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key, key_source = _resolve_api_key()
    summary = {
        "run_started_utc": datetime.utcnow().isoformat() + "Z",
        "tickers": [t.strip().upper() for t in args.tickers.split(",") if t.strip()],
        "horizon_bars": int(args.horizon),
        "endpoints": [e.strip() for e in args.endpoints.split(",") if e.strip()],
        "api_key_source": key_source,
        "api_key_present": api_key is not None,
        "output_tsv": str(output_path),
    }

    if api_key is None:
        # Write a header-only TSV + status JSON. Do NOT punt to user — leave
        # an auditable trail so when launchd runs Sun 9am ET, the gap is logged.
        cols = ["endpoint", "ticker", "feature", "IC", "p_value", "n_obs",
                "n_filings", "status", "url_used"]
        empty_df = pd.DataFrame(columns=cols)
        empty_df.to_csv(output_path, sep="\t", index=False)
        summary["status"] = "credentials_missing"
        summary["note"] = ("No QUIVER_API_KEY found in env or "
                           "~/.zg/quiver/api_key or $AI_DRIVE/.secrets/quiver_api_key. "
                           "Skipped all endpoint fetches. To complete the audit, "
                           "place the tier-1 API key at one of those locations "
                           "and re-run.")
        Path(str(output_path) + ".summary.json").write_text(json.dumps(summary, indent=2))
        print("[quiver-ic] credentials_missing — wrote empty TSV + summary", file=sys.stderr)
        return 0

    tickers = summary["tickers"]
    endpoints = summary["endpoints"]

    # Preload OHLCV + forward returns
    ohlcv = {}
    skipped = []
    for t in tickers:
        df = _load_ohlcv(t)
        if df is None:
            skipped.append(t)
            continue
        # Normalize close col
        close_col = [c for c in df.columns if c.lower() == "close"][0]
        df = df.rename(columns={close_col: "close"})
        df["fwd_ret"] = df["close"].shift(-args.horizon) / df["close"] - 1.0
        ohlcv[t] = df
    if not ohlcv:
        summary["status"] = "no_ticker_data"
        summary["tickers_skipped"] = skipped
        Path(str(output_path) + ".summary.json").write_text(json.dumps(summary, indent=2))
        cols = ["endpoint", "ticker", "feature", "IC", "p_value", "n_obs",
                "n_filings", "status", "url_used"]
        pd.DataFrame(columns=cols).to_csv(output_path, sep="\t", index=False)
        print("[quiver-ic] no ticker OHLCV available; aborting", file=sys.stderr)
        return 3

    rows = []
    for endpoint_key in endpoints:
        if endpoint_key not in ENDPOINTS:
            rows.append({
                "endpoint": endpoint_key, "ticker": "*", "feature": "*",
                "IC": float("nan"), "p_value": float("nan"), "n_obs": 0,
                "n_filings": 0, "status": "unknown_endpoint", "url_used": "",
            })
            continue
        for t in tickers:
            if t not in ohlcv:
                continue
            time.sleep(args.api_pause_sec)
            records, url_used, fetch_err = _try_endpoints(endpoint_key, t, api_key)
            if records is None:
                rows.append({
                    "endpoint": endpoint_key, "ticker": t, "feature": "(count_30d/value_sum_30d)",
                    "IC": float("nan"), "p_value": float("nan"), "n_obs": 0,
                    "n_filings": 0,
                    "status": "fetch_failed_{}".format(fetch_err or "unknown"),
                    "url_used": "",
                })
                continue

            normalized = _normalize_records(records, endpoint_key)
            n_filings = len(normalized)
            feat_df = _build_pit_safe_feature(
                normalized, ohlcv[t].index, args.horizon
            )
            fwd = ohlcv[t]["fwd_ret"]
            for feat_col in ["count_30d", "value_sum_30d"]:
                ic, pv, nobs = _pearson_ic(feat_df[feat_col], fwd)
                rows.append({
                    "endpoint": endpoint_key,
                    "ticker": t,
                    "feature": feat_col,
                    "IC": ic,
                    "p_value": pv,
                    "n_obs": nobs,
                    "n_filings": n_filings,
                    "status": "ok",
                    "url_used": url_used or "",
                })

    pd.DataFrame(rows, columns=[
        "endpoint", "ticker", "feature", "IC", "p_value", "n_obs",
        "n_filings", "status", "url_used"
    ]).to_csv(output_path, sep="\t", index=False)
    summary["status"] = "complete"
    summary["rows_written"] = len(rows)
    summary["tickers_skipped"] = skipped
    Path(str(output_path) + ".summary.json").write_text(json.dumps(summary, indent=2))
    print("[quiver-ic] wrote {} rows -> {}".format(len(rows), output_path))
    print("[quiver-ic] summary -> {}.summary.json".format(output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
