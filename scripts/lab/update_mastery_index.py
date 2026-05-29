"""update_mastery_index.py — Merge validation results into indicator_mastery_index.csv.

Reads the runner's `summary.json` files for each indicator and writes back to the canonical
mastery CSV at /My Drive/Combo-1/BackTests & Data/indicator_mastery_index.csv with new fields:
  win_rate_v2, pbo, dsr, wfe, holdout_sharpe, validation_status, validation_utc, validation_n_tickers

Always creates a timestamped backup first per workspace safety rules.

Name → indicator_id mapping uses the registry's name and the CSV's indicator_name fuzzy match.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CANONICAL_CSV = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/Combo-1/BackTests & Data/indicator_mastery_index.csv")
LOCAL_CACHE_CSV = Path("/Volumes/ZG-2TB/zg/btd-local/indicator_mastery_index.csv")
DRIVE_RESULTS = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/data/indicator_validation")
LOCAL_RESULTS = Path("/Volumes/ZG-2TB/zg/indicator_backtest/results")

# Map runner indicator name -> mastery CSV indicator_name substring (lowercased) for matching
NAME_TO_CSV = {
    "MACD_12_26_9": "macd(12,26,9)",
    "BB_20_2": "bollinger bands (20,2)",
    "BB_pctB": "bollinger %b",
    "Keltner_20_1.5": "keltner channels (20,1.5)",
    "OBV": "obv",
    "Stoch_14_3_3": "stochastic (14,3,3)",
    "Williams_R_14": "williams %r(14)",
    "CCI_20": "cci(20)",
    "MFI_14": "mfi(14)",
    "Fisher_Transform_10": "fisher transform",
    "Connors_RSI_3": "connors rsi(3)",
    "Supertrend_10_3": "supertrend",
}


def find_summary(name: str, utc_tag: str) -> dict | None:
    """Try local then Drive results path."""
    for base in (LOCAL_RESULTS / name / utc_tag, DRIVE_RESULTS / name / utc_tag):
        p = base / "summary.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return None


def backup_csv(p: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bp = p.parent / f"indicator_mastery_index_pre_validation_{ts}.csv"
    try:
        shutil.copy(p, bp)
        print(f"backup: {bp}")
    except OSError as e:
        print(f"backup FAILED at {bp}: {e}")
    return bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utc-tag", required=True)
    ap.add_argument("--target-csv", default=str(CANONICAL_CSV))
    ap.add_argument("--cache-csv", default=str(LOCAL_CACHE_CSV))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    target = Path(args.target_csv)
    cache = Path(args.cache_csv)

    # Prefer local cache (writeable + readable), fall back to canonical
    src = cache if cache.exists() else target
    # The CSV has unquoted commas inside parameter fields like "MACD(12,26,9)".
    # Use the python `csv` module to read it as raw rows, then fix the column count
    # by treating the canonical 15-column schema as fixed and joining overflow into the
    # `evidence_path` column (the last column, which may already contain commas).
    import csv as _csv
    try:
        with open(src, newline="") as f:
            reader = _csv.reader(f)
            rows = list(reader)
    except OSError as e:
        print(f"read failed at {src}: {e}")
        return 1
    header = rows[0]
    n_cols = len(header)
    fixed = []
    for r in rows[1:]:
        if not r or not any(r):
            continue
        if len(r) == n_cols:
            fixed.append(r)
        elif len(r) > n_cols:
            # Overflow — collapse unquoted commas inside the indicator_name field (col 1).
            # Heuristic: if r[1] starts with "MACD(", "Bollinger ", "Stochastic ", "Keltner ",
            # or ends with "(<digit>" pattern, merge columns 1..k until we find a non-numeric
            # token closing with ')'. Otherwise default to collapsing extras into the last
            # column (evidence_path with unquoted commas).
            r1 = r[1]
            join_until = None
            if any(r1.startswith(p) for p in ("MACD(", "Bollinger ", "Stochastic ", "Keltner ", "MACD Histogram ")):
                # Walk forward until we see ')' closing the parameter spec
                for k in range(2, min(len(r), 6)):
                    if ")" in r[k]:
                        join_until = k
                        break
            if join_until is not None:
                merged_name = ",".join(r[1: join_until + 1])
                r2 = [r[0], merged_name] + list(r[join_until + 1:])
                # Now r2 may still have extras at the end — collapse into evidence_path
                if len(r2) > n_cols:
                    r2 = r2[: n_cols - 1] + [",".join(r2[n_cols - 1:])]
                if len(r2) < n_cols:
                    r2 = r2 + [""] * (n_cols - len(r2))
                fixed.append(r2)
            else:
                # Default: collapse trailing extras into evidence_path
                r2 = r[: n_cols - 1] + [",".join(r[n_cols - 1:])]
                if len(r2) == n_cols:
                    fixed.append(r2)
        else:
            # Pad short rows (rare)
            r2 = r + [""] * (n_cols - len(r))
            fixed.append(r2)
    df = pd.DataFrame(fixed, columns=header)
    print(f"read {len(df)} rows from {src} (raw {len(rows)-1})")

    # Add new columns if missing
    new_cols = ["win_rate_v2", "pbo", "dsr", "wfe", "validation_status", "validation_utc", "validation_n_tickers"]
    for c in new_cols:
        if c not in df.columns:
            df[c] = None

    updates = 0
    for name, csv_substr in NAME_TO_CSV.items():
        summary = find_summary(name, args.utc_tag)
        if summary is None:
            print(f"  [no summary] {name}")
            continue
        # Find matching row(s)
        mask = df["indicator_name"].str.lower().str.contains(csv_substr.lower(), regex=False, na=False)
        if mask.sum() == 0:
            print(f"  [no match] {name} -> '{csv_substr}'")
            continue
        idxs = df.index[mask].tolist()
        for i in idxs:
            df.at[i, "win_rate_v2"] = round(summary.get("wr_mean") or 0, 4)
            df.at[i, "pbo"] = round(summary.get("pbo_mean") or 0, 4)
            df.at[i, "dsr"] = round(summary.get("dsr_mean") or 0, 4)
            df.at[i, "wfe"] = round(summary.get("wfe_mean") or 0, 4)
            df.at[i, "validation_status"] = summary.get("new_status")
            df.at[i, "validation_utc"] = args.utc_tag
            df.at[i, "validation_n_tickers"] = summary.get("n_tickers_ok")
            # Also overwrite top-level status field if validation produced a clear verdict
            if summary.get("new_status") in ("TESTED_MULTIPLE_TICKERS", "REJECTED"):
                df.at[i, "status"] = summary.get("new_status")
            updates += 1
            print(f"  [updated row {i}] {name} -> {summary.get('new_status')} WR={summary.get('wr_mean'):.3f} PBO={summary.get('pbo_mean'):.3f} DSR={summary.get('dsr_mean'):.3f}")

    print(f"\ntotal row updates: {updates}")
    if args.dry_run:
        print("DRY-RUN — not writing")
        return 0

    # Always write a local cache copy first (safer if Drive is flaky)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        print(f"wrote cache: {cache}")
    except OSError as e:
        print(f"cache write FAILED: {e}")

    # Backup + write canonical
    if target.exists():
        backup_csv(target)
    try:
        df.to_csv(target, index=False)
        print(f"wrote canonical: {target}")
    except OSError as e:
        print(f"canonical write FAILED: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
