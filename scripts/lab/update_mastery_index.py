"""update_mastery_index.py — Merge validation results into indicator_mastery_index.csv.

Two outputs:
1. `indicator_validation_results.csv` — STANDALONE table of validation outputs (safe, idempotent).
2. `indicator_mastery_index.csv` — original file with appended validation columns. To avoid
   damaging the unquoted-comma cells in source columns, we operate at the *line* level:
   - Read original file as raw text lines
   - For each line, regex-match the row id and indicator name pattern
   - Append validation fields by string-concatenation (after the original line's trailing fields)

This is non-destructive: original columns are untouched even if their unquoted commas would
confuse a standard pandas csv parser.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

CANONICAL_CSV = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/Combo-1/BackTests & Data/indicator_mastery_index.csv")
LOCAL_CACHE_CSV = Path("/Volumes/ZG-2TB/zg/btd-local/indicator_mastery_index.csv")
DRIVE_RESULTS = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/data/indicator_validation")
LOCAL_RESULTS = Path("/Volumes/ZG-2TB/zg/indicator_backtest/results")
STANDALONE_LOCAL = Path("/Volumes/ZG-2TB/zg/btd-local/indicator_validation_results.csv")
STANDALONE_DRIVE = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/data/indicator_validation_results.csv")

# Map runner indicator name -> indicator_id in mastery CSV (1-indexed)
NAME_TO_ID = {
    "MACD_12_26_9": "11",
    "BB_20_2": "13",
    "BB_pctB": "14",
    "Keltner_20_1.5": "15",
    "OBV": "19",
    "Stoch_14_3_3": "20",
    "Williams_R_14": "21",
    "CCI_20": "22",
    "MFI_14": "23",
    "Fisher_Transform_10": "25",
    "Connors_RSI_3": "34",
    "Supertrend_10_3": "28",
}


def find_summary(name: str, utc_tag: str) -> dict | None:
    for base in (LOCAL_RESULTS / name / utc_tag, DRIVE_RESULTS / name / utc_tag):
        p = base / "summary.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return None


def write_standalone(summaries: dict[str, dict], utc_tag: str, paths: list[Path]):
    """Write a clean validation-results CSV (separate file, no risk to mastery CSV)."""
    rows = []
    for name, s in summaries.items():
        rows.append({
            "indicator_id": NAME_TO_ID.get(name, ""),
            "indicator_name_runner": name,
            "validation_utc": utc_tag,
            "timeframe": "1d",
            "n_tickers_attempted": s.get("n_tickers_attempted"),
            "n_tickers_ok": s.get("n_tickers_ok"),
            "n_trades_total": s.get("n_trades_total"),
            "wr_mean": round(s.get("wr_mean") or 0, 4),
            "wfe_mean": round(s.get("wfe_mean") or 0, 4),
            "pbo_mean": round(s.get("pbo_mean") or 0, 4),
            "dsr_mean": round(s.get("dsr_mean") or 0, 4),
            "new_status": s.get("new_status"),
            "elapsed_sec": round(s.get("elapsed_sec") or 0, 1),
        })
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for p in paths:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            print(f"  wrote standalone: {p}")
        except OSError as e:
            print(f"  write FAILED {p}: {e}")


def append_to_mastery(target: Path, summaries: dict[str, dict], utc_tag: str):
    """Non-destructive: read original file as text lines, append validation columns to matched
    rows, write to a new copy. Safer than parsing the broken CSV.
    """
    try:
        text = target.read_text()
    except OSError as e:
        print(f"read failed at {target}: {e}")
        return None
    lines = text.splitlines()
    if not lines:
        return None
    header = lines[0]
    new_cols = "win_rate_v2,pbo_v2,dsr_v2,wfe_v2,validation_status,validation_utc,validation_n_tickers"
    if "validation_utc" not in header:
        header = header + "," + new_cols

    out_lines = [header]
    updates = 0
    for line in lines[1:]:
        if not line.strip():
            out_lines.append(line)
            continue
        # First column is indicator_id followed by comma
        m = re.match(r"^(\d+),", line)
        rid = m.group(1) if m else None
        matched_name = None
        for name, idx in NAME_TO_ID.items():
            if rid == idx and name in summaries:
                matched_name = name
                break
        if matched_name is None:
            out_lines.append(line)
            continue
        s = summaries[matched_name]
        append = f',{round(s.get("wr_mean") or 0, 4)},{round(s.get("pbo_mean") or 0, 4)},{round(s.get("dsr_mean") or 0, 4)},{round(s.get("wfe_mean") or 0, 4)},{s.get("new_status")},{utc_tag},{s.get("n_tickers_ok")}'
        out_lines.append(line + append)
        updates += 1
        print(f"  [appended row id={rid}] {matched_name} -> {s.get('new_status')} WR={s.get('wr_mean'):.3f}")

    print(f"  total rows appended: {updates}")
    return "\n".join(out_lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utc-tag", required=True)
    ap.add_argument("--target-csv", default=str(CANONICAL_CSV))
    ap.add_argument("--cache-csv", default=str(LOCAL_CACHE_CSV))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-drive", action="store_true", help="skip Drive writes (Drive FUSE flaky)")
    args = ap.parse_args()

    # Collect all summaries that exist
    summaries: dict[str, dict] = {}
    for name in NAME_TO_ID:
        s = find_summary(name, args.utc_tag)
        if s is not None:
            summaries[name] = s
        else:
            print(f"  [no summary] {name}")
    print(f"\ncollected {len(summaries)} summaries\n")
    if not summaries:
        return 0

    # Standalone CSV
    print("== standalone validation results CSV ==")
    paths = [STANDALONE_LOCAL]
    if not args.no_drive:
        paths.append(STANDALONE_DRIVE)
    if not args.dry_run:
        write_standalone(summaries, args.utc_tag, paths)
    else:
        print("  DRY-RUN — not writing standalone")

    # Append to mastery (local cache first, then canonical)
    print("\n== appending to mastery_index ==")
    # Local cache:
    text = append_to_mastery(Path(args.cache_csv), summaries, args.utc_tag)
    if text is not None and not args.dry_run:
        try:
            Path(args.cache_csv).write_text(text)
            print(f"  wrote local cache: {args.cache_csv}")
        except OSError as e:
            print(f"  cache write FAILED: {e}")

    # Canonical:
    if not args.no_drive:
        text = append_to_mastery(Path(args.target_csv), summaries, args.utc_tag)
        if text is not None and not args.dry_run:
            # Backup first
            try:
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                bp = Path(args.target_csv).parent / f"indicator_mastery_index_pre_validation_{ts}.csv"
                shutil.copy(args.target_csv, bp)
                print(f"  backup: {bp}")
            except OSError as e:
                print(f"  backup FAILED: {e}")
            try:
                Path(args.target_csv).write_text(text)
                print(f"  wrote canonical: {args.target_csv}")
            except OSError as e:
                print(f"  canonical write FAILED: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
