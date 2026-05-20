"""
gabriel_priors_harvest.py — One-shot harvester that builds a unified per-ticker
priors parquet from BOTH formats found under
  /My Drive/version_3 - Gabriel/Mastered Tickers - Gabriel/:

  (A) Deep mastered folders (27 tickers, e.g. AAPL/, AMZN/) with:
        analysis_report.md  — overall PF/WR/N + regime + monthly perf
        config.yaml         — champion winner: signal/wr/pf/n/stop/target

  (B) Single-file mastered stubs (496 tickers, e.g. A.md, ABBV.md) with:
        ## Formula: <name>
        - Weights / Threshold / Exit (ATR stop/target)
        - Walk-Forward Results (period, total trades, win rate, total PnL)

Output:
  cache/gabriel_priors_full.parquet
    columns: ticker, source ("folder"|"stub"|"missing"),
             pf, wr (fraction), n_trades, regime_breakdown_score,
             monthly_perf_consistency, formula, threshold, atr_stop,
             atr_target, total_pnl_pct, avg_per_trade_pct

Idempotent: re-running rebuilds the parquet from scratch (cheap once
the OS has the Drive entries cached). Safe with .shift(1) — these
scalars summarise a SEPARATE pre-live backtest.

Author: 2026-05-17, Gabriel Full-Coverage wave.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gabriel_harvest")

ROOT = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "version_3 - Gabriel/Mastered Tickers - Gabriel"
)
OUT = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "AI-Tools/s&p500-ticker-mastery/cache/gabriel_priors_full.parquet"
)

NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def _safe_float(s):
    if s is None:
        return None
    m = NUM_RE.search(str(s))
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


# ---------- Parser for deep-folder analysis_report.md ----------
def parse_folder_analysis(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log.debug("read %s failed: %s", path, e)
        return out

    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) >= 2:
                key = cells[0].lower()
                val = cells[1]
                if key == "total trades" and "n_trades" not in out:
                    n = _safe_float(val)
                    if n is not None:
                        out["n_trades"] = int(n)
                elif key == "win rate" and "wr" not in out:
                    wr = _safe_float(val)
                    if wr is not None:
                        out["wr"] = wr / 100.0 if wr > 1.0 else wr
                elif key == "profit factor" and "pf" not in out:
                    pf = _safe_float(val)
                    if pf is not None:
                        out["pf"] = pf

    # Regime breakdown (uptrend / sideways / downtrend) – avg per-formula WR per regime
    for regime in ("uptrend", "sideways", "downtrend"):
        pat = re.compile(rf"### Regime: {regime}\s*(.*?)(?=\n###|\Z)", re.S | re.I)
        m = pat.search(text)
        wrs = []
        if m:
            for line in m.group(1).splitlines():
                if line.startswith("|"):
                    cells = [c.strip() for c in line.split("|") if c.strip()]
                    if (
                        len(cells) >= 3
                        and cells[0].lower() != "formula"
                        and "---" not in cells[0]
                    ):
                        v = _safe_float(cells[2])
                        if v is not None and 0 <= v <= 100:
                            wrs.append(v / 100.0)
        if wrs:
            out[f"regime_wr_{regime}"] = float(np.mean(wrs))

    # Monthly performance
    m2 = re.search(r"##\s*\d*\.?\s*Monthly Performance\s*(.*)", text, re.S | re.I)
    if m2:
        monthly = []
        for line in m2.group(1).splitlines():
            if line.startswith("|"):
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if (
                    len(cells) >= 4
                    and cells[0].lower() != "month"
                    and "---" not in cells[0]
                ):
                    rv = _safe_float(cells[3])
                    if rv is not None:
                        monthly.append(rv)
        if monthly:
            out["monthly_total_ret"] = monthly
    return out


# ---------- Parser for single-file stub <TICKER>.md ----------
STUB_FORMULA_RE = re.compile(r"##\s*Formula:\s*([^\s\n]+)", re.I)
STUB_THRESHOLD_RE = re.compile(r"\*\*Threshold\*\*:\s*([-+]?\d*\.?\d+)", re.I)
STUB_EXIT_RE = re.compile(
    r"\*\*Exit\*\*:.*?([-+]?\d*\.?\d+)\s*ATR\s*stop.*?([-+]?\d*\.?\d+)\s*ATR\s*target",
    re.I,
)
STUB_TRADES_RE = re.compile(r"\*\*Total trades\*\*:\s*([-+]?\d+)", re.I)
STUB_WR_RE = re.compile(r"\*\*Win rate\*\*:\s*([-+]?\d*\.?\d+)\s*%", re.I)
STUB_PNL_RE = re.compile(r"\*\*Total PnL\*\*:\s*([-+]?\d*\.?\d+)\s*%", re.I)
STUB_AVG_RE = re.compile(r"\*\*Avg per trade\*\*:\s*([-+]?\d*\.?\d+)\s*%", re.I)


def parse_stub_md(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log.debug("read %s failed: %s", path, e)
        return out

    if (m := STUB_FORMULA_RE.search(text)):
        out["formula"] = m.group(1).strip()
    if (m := STUB_THRESHOLD_RE.search(text)):
        v = _safe_float(m.group(1))
        if v is not None:
            out["threshold"] = v
    if (m := STUB_EXIT_RE.search(text)):
        stop = _safe_float(m.group(1))
        tgt = _safe_float(m.group(2))
        if stop is not None:
            out["atr_stop"] = stop
        if tgt is not None:
            out["atr_target"] = tgt
    if (m := STUB_TRADES_RE.search(text)):
        v = _safe_float(m.group(1))
        if v is not None:
            out["n_trades"] = int(v)
    if (m := STUB_WR_RE.search(text)):
        v = _safe_float(m.group(1))
        if v is not None:
            out["wr"] = v / 100.0
    if (m := STUB_PNL_RE.search(text)):
        v = _safe_float(m.group(1))
        if v is not None:
            out["total_pnl_pct"] = v
    if (m := STUB_AVG_RE.search(text)):
        v = _safe_float(m.group(1))
        if v is not None:
            out["avg_per_trade_pct"] = v

    # Synthetic PF from WR (Kelly-style): if no trade-level CSV available,
    # approximate PF = (WR * avg_win) / ((1-WR) * avg_loss). With ATR-based
    # symmetric R, avg_win/avg_loss ~ atr_target/atr_stop. Conservative.
    wr = out.get("wr")
    tgt = out.get("atr_target")
    stop = out.get("atr_stop")
    if wr is not None and tgt is not None and stop is not None and stop > 0 and wr < 1.0:
        out["pf"] = (wr * tgt) / ((1 - wr) * stop)
    return out


# ---------- Scoring helpers ----------
def compute_regime_score(parsed: dict) -> float:
    regs = [
        parsed.get("regime_wr_uptrend"),
        parsed.get("regime_wr_sideways"),
        parsed.get("regime_wr_downtrend"),
    ]
    rs = [r for r in regs if r is not None]
    if len(rs) == 3:
        return float(np.mean(rs) - 0.5 * np.std(rs))
    # If only the stub gives a global WR, approximate robustness from WR alone
    wr = parsed.get("wr")
    if wr is not None:
        # treat high-WR as more robust; scale down so it sits in roughly [0, 1]
        return float(wr * 0.85)
    return 0.0


def compute_consistency(parsed: dict) -> float:
    monthly = parsed.get("monthly_total_ret", [])
    if monthly and len(monthly) >= 2:
        m_mean = float(np.mean(monthly))
        m_std = float(np.std(monthly))
        if abs(m_mean) > 1e-6:
            cv = m_std / m_mean
            return float(np.clip(-cv, -3.0, 3.0))
    return 0.0


# ---------- Driver ----------
def harvest(resume: bool = True) -> pd.DataFrame:
    """Harvest Gabriel priors into a single parquet.

    resume=True: read existing OUT parquet, skip tickers already covered.
                 Saves progress every 25 stubs so a Drive stall is recoverable.
    """
    if not ROOT.exists():
        raise FileNotFoundError(f"Gabriel root missing: {ROOT}")
    t0 = time.time()

    # Enumerate ONCE (Drive: avoid repeated listdir calls)
    entries = sorted(os.listdir(ROOT))
    log.info("entries=%d under %s", len(entries), ROOT.name)

    folders = []  # ticker dirnames
    stubs = []  # (ticker, file)
    for e in entries:
        full = ROOT / e
        if full.is_dir():
            folders.append(e)
        elif e.endswith(".md") and not e.startswith("_"):
            stubs.append(e[:-3])  # drop .md

    log.info("folders=%d stubs=%d", len(folders), len(stubs))

    # Resume support: load existing parquet, skip tickers already harvested.
    rows = []
    already_done: set = set()
    if resume and OUT.exists():
        try:
            existing = pd.read_parquet(OUT)
            rows = existing.to_dict("records")
            already_done = set(existing["ticker"].astype(str))
            log.info("resume: loaded %d existing rows", len(rows))
        except Exception as e:
            log.warning("resume load failed (%s) — starting fresh", e)
            rows = []
            already_done = set()
    # 1) Parse folder-based deep masteries
    for i, t in enumerate(folders, 1):
        if t in already_done:
            continue
        if i % 5 == 0:
            log.info("folder %d/%d %s", i, len(folders), t)
        parsed = parse_folder_analysis(ROOT / t / "analysis_report.md")
        if not parsed:
            continue
        rows.append(
            {
                "ticker": t,
                "source": "folder",
                "pf": float(parsed.get("pf", 0.0)),
                "wr": float(parsed.get("wr", 0.0)),
                "n_trades": int(parsed.get("n_trades", 0)),
                "regime_breakdown_score": compute_regime_score(parsed),
                "monthly_perf_consistency": compute_consistency(parsed),
                "formula": "",
                "threshold": 0.0,
                "atr_stop": 0.0,
                "atr_target": 0.0,
                "total_pnl_pct": 0.0,
                "avg_per_trade_pct": 0.0,
            }
        )
    log.info("folder parse done: %d rows", len(rows))

    # 2) Parse single-file stubs (with incremental save every 25 to survive Drive stalls)
    n_stub_ok = 0
    for i, t in enumerate(stubs, 1):
        if t in already_done:
            continue
        if i % 10 == 0:
            log.info("stub %d/%d %s elapsed=%.1fs", i, len(stubs), t, time.time() - t0)
        parsed = parse_stub_md(ROOT / f"{t}.md")
        if not parsed:
            continue
        rows.append(
            {
                "ticker": t,
                "source": "stub",
                "pf": float(parsed.get("pf", 0.0)),
                "wr": float(parsed.get("wr", 0.0)),
                "n_trades": int(parsed.get("n_trades", 0)),
                "regime_breakdown_score": compute_regime_score(parsed),
                "monthly_perf_consistency": compute_consistency(parsed),
                "formula": parsed.get("formula", ""),
                "threshold": float(parsed.get("threshold", 0.0)),
                "atr_stop": float(parsed.get("atr_stop", 0.0)),
                "atr_target": float(parsed.get("atr_target", 0.0)),
                "total_pnl_pct": float(parsed.get("total_pnl_pct", 0.0)),
                "avg_per_trade_pct": float(parsed.get("avg_per_trade_pct", 0.0)),
            }
        )
        n_stub_ok += 1
        # Incremental checkpoint every 50 stubs to /tmp first (Drive-write is slow).
        # The final save to Drive happens once at the end.
        if n_stub_ok % 50 == 0:
            try:
                tmp_df = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
                tmp_path = Path("/tmp/gabriel_priors_full.parquet")
                tmp_df.to_parquet(tmp_path, index=False)
                log.info("local checkpoint: wrote %d rows -> %s", len(tmp_df), tmp_path)
            except Exception as e:
                log.warning("checkpoint save failed: %s", e)
    log.info("stub parse done: %d/%d OK", n_stub_ok, len(stubs))

    df = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    log.info("FINAL rows=%d distinct_tickers=%d elapsed=%.1fs",
             len(df), df["ticker"].nunique(), time.time() - t0)
    return df


def main():
    import shutil

    df = harvest()
    # Write to /tmp first (fast), then copy to Drive in one shot.
    tmp_out = Path("/tmp/gabriel_priors_full.parquet")
    df.to_parquet(tmp_out, index=False)
    log.info("wrote staging %s (%d rows)", tmp_out, len(df))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tmp_out, OUT)
    log.info("copied to Drive: %s (%d rows)", OUT, len(df))
    # Quick coverage report
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(df["source"].value_counts().to_string())
    nz = (df[["pf", "wr", "n_trades"]] != 0).any(axis=1).sum()
    print(f"Rows with any non-zero pf/wr/n: {nz}")
    print(df.head(8).to_string())


if __name__ == "__main__":
    main()
