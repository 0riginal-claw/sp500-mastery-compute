#!/usr/bin/env python3
"""memory_auto_save_daemon.py

Auto-aggregates mastery results + pacing transitions + daemon health + sweep
artifacts + paper-trade P&L into the user-memory directory so cold-start
helpers don't rediscover state.

Run modes:
  - Hourly (LaunchAgent StartInterval=3600)
  - Nightly (LaunchAgent StartCalendarInterval hour=23 minute=55)

Outputs:
  AI-Tools/s&p500-ticker-mastery/cache/per_ticker_best.parquet
  <MEM>/project_per_ticker_best.md
  <MEM>/project_pacing_history.md
  <MEM>/project_daemon_health.md
  <MEM>/project_paper_trade_pnl.md
  <MEM>/MEMORY.md  (4 new index lines, idempotent)

Safety:
  - Backs up MEMORY.md before each edit (timestamped, kept in <MEM>/backups/)
  - Refuses to overwrite a memory file whose frontmatter metadata.type is
    anything other than 'auto_generated' (so curated files are protected).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ----------------------------- paths -----------------------------------------
AI_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools"
)
SP = AI_ROOT / "s&p500-ticker-mastery"
MEM = AI_ROOT / (
    "ClaudeCode/config/projects/"
    "-Users-orginal-Library-CloudStorage-GoogleDrive-zachgladstone-gmail-com-"
    "My-Drive-AI-Tools-ClaudeCode-projects/memory"
)
MEMORY_INDEX = MEM / "MEMORY.md"
BACKUP_DIR = MEM / "backups"
CACHE_DIR = SP / "cache"
LOG_DIR = SP / "logs"

MASTERY_DIR = SP / "mastery_files"
PACING_STATE = SP / "dashboard" / "pacing_state.json"
PACING_HISTORY = SP / "dashboard" / "pacing_history.jsonl"
SWEEP_DIR = SP / "sweep_artifacts"
PAPER_DAILY = SP / "paper_trade" / "daily"
PAPER_STATE = SP / "paper_trade" / "state"

# Files we own. Any other metadata.type means manually curated → skip.
OWNED_FILES = [
    "project_per_ticker_best.md",
    "project_pacing_history.md",
    "project_daemon_health.md",
    "project_paper_trade_pnl.md",
]
INDEX_LINES = {
    "project_per_ticker_best.md": (
        "- [Per-ticker best params](project_per_ticker_best.md) — "
        "Auto-aggregated mastery results (PF/DD/WR/n + params) for all "
        "mastered S&P 500 tickers."
    ),
    "project_pacing_history.md": (
        "- [Pacing regime history](project_pacing_history.md) — "
        "Auto-aggregated regime transitions (under/on/over/emergency) with "
        "timestamps + cost%."
    ),
    "project_daemon_health.md": (
        "- [Daemon health snapshot](project_daemon_health.md) — "
        "Auto-aggregated `launchctl` snapshot for `com.zg.*` daemons, last "
        "7-day stats."
    ),
    "project_paper_trade_pnl.md": (
        "- [Paper-trade P&L](project_paper_trade_pnl.md) — "
        "Auto-aggregated daily P&L summary since paper-trade launch."
    ),
}
AUTO_BANNER_BEGIN = "<!-- BEGIN AUTO-AGGREGATED INDEX -->"
AUTO_BANNER_END = "<!-- END AUTO-AGGREGATED INDEX -->"

# ----------------------------- logging ---------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "memory_auto_save.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("memory_auto_save")


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


# ----------------------------- frontmatter -----------------------------------
FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def read_frontmatter_type(path: Path) -> str | None:
    """Return the metadata.type field of an existing memory file, or None."""
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = FM_RE.match(text)
    if not m:
        return None
    block = m.group(1)
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("type:"):
            return line.split(":", 1)[1].strip()
    return None


def safe_write_memory(path: Path, body: str, frontmatter_extra: dict | None = None) -> bool:
    """Write a memory file only if it doesn't exist or is auto_generated.

    Returns True if written, False if skipped (curated file protected).
    """
    existing_type = read_frontmatter_type(path)
    if existing_type is not None and existing_type != "auto_generated":
        log.warning(
            "SKIP %s (frontmatter type=%s — curated, not overwriting)",
            path.name, existing_type,
        )
        return False

    fm = {
        "name": path.stem.replace("_", "-"),
        "description": "Auto-aggregated by memory_auto_save_daemon.py — do not edit by hand.",
        "metadata": {
            "node_type": "memory",
            "type": "auto_generated",
            "generated_by": "memory_auto_save_daemon.py",
            "generated_at": now_iso(),
        },
    }
    if frontmatter_extra:
        fm["metadata"].update(frontmatter_extra)

    fm_lines = ["---"]
    fm_lines.append(f"name: {fm['name']}")
    fm_lines.append(f"description: {fm['description']}")
    fm_lines.append("metadata:")
    for k, v in fm["metadata"].items():
        fm_lines.append(f"  {k}: {v}")
    fm_lines.append("---\n")
    out = "\n".join(fm_lines) + "\n" + body.rstrip() + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out, encoding="utf-8")
    log.info("WROTE %s (%d bytes)", path.name, len(out))
    return True


# ----------------------------- 1a. mastery files -----------------------------
def parse_mastery_file(path: Path) -> dict:
    """Heuristic parser for the various mastery-file formats (ML / XGB / FAILED).

    Returns dict with: ticker, version, status, pf, dd, wr, n, ret, comp_score,
    params (json str or ""), mtime.
    """
    name = path.stem  # e.g. AAPL_ML_mastered or AAPL_XGB_v10_mythos_mastered
    is_failed = name.endswith("_FAILED")
    stem = name.replace("_mastered", "").replace("_FAILED", "")
    parts = stem.split("_", 1)
    ticker = parts[0]
    version = parts[1] if len(parts) > 1 else "ML"

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = ""

    def find_num(patterns: list[str]) -> float | None:
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
        return None

    pf = find_num([
        r"profit_factor[\"']?\s*:\s*([0-9.+-eE]+)",
        r"PF:?\s*([0-9.]+)",
        r"Profit factor\s*\|\s*([0-9.]+)",
    ])
    dd = find_num([
        r"max_drawdown_pct[\"']?\s*:\s*(-?[0-9.+-eE]+)",
        r"DD:?\s*(-?[0-9.]+%?)",
        r"Max drawdown[^|]*\|\s*(-?[0-9.]+)",
    ])
    wr = find_num([
        r"win_rate[\"']?\s*:\s*([0-9.+-eE]+)",
        r"WR:?\s*([0-9.]+)",
        r"Win rate\s*\|\s*([0-9.]+)",
    ])
    n = find_num([
        r"n_trades[\"']?\s*:\s*([0-9]+)",
        r"n trades?:?\s*([0-9]+)",
        r"n_trades\s*\|\s*([0-9]+)",
    ])
    ret = find_num([
        r"total_return_pct[\"']?\s*:\s*(-?[0-9.+-eE]+)",
        r"RET:?\s*(-?[0-9.]+%?)",
        r"Total return[^|]*\|\s*(-?[0-9.]+)",
    ])

    # composite score: pf - 5*|dd| (heuristic, just for ranking)
    comp = None
    if pf is not None and dd is not None:
        comp = pf - 5.0 * abs(dd)

    # rough params: extract any best_params JSON-ish blob (XGB sweeps)
    params = ""
    m = re.search(r"best_params[\"']?\s*:\s*(\{[^{}]*\})", text)
    if m:
        params = m.group(1).strip()

    return {
        "ticker": ticker,
        "version": version,
        "status": "FAILED" if is_failed else "MASTERED",
        "pf": pf,
        "dd": dd,
        "wr": wr,
        "n": int(n) if n is not None else None,
        "ret": ret,
        "comp_score": comp,
        "params": params,
        "mtime": _dt.datetime.fromtimestamp(
            path.stat().st_mtime, tz=_dt.timezone.utc
        ).isoformat(timespec="seconds"),
    }


def build_per_ticker_best() -> pd.DataFrame:
    rows: list[dict] = []
    if not MASTERY_DIR.exists():
        log.warning("mastery_files dir missing: %s", MASTERY_DIR)
        return pd.DataFrame(rows)
    for p in sorted(MASTERY_DIR.glob("*_mastered.md")):
        rows.append(parse_mastery_file(p))
    for p in sorted(MASTERY_DIR.glob("*_FAILED.md")):
        rows.append(parse_mastery_file(p))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # pick best version per ticker by comp_score (mastered preferred)
    df["_rank"] = df.apply(
        lambda r: (0 if r["status"] == "MASTERED" else 1, -(r["comp_score"] or -1e9)),
        axis=1,
    )
    df_sorted = df.sort_values(["ticker", "_rank"]).drop(columns="_rank")
    df_best = df_sorted.groupby("ticker", as_index=False).first()
    return df_best.sort_values("ticker").reset_index(drop=True)


def write_per_ticker_best() -> bool:
    df = build_per_ticker_best()
    if df.empty:
        body = "_No mastery files found in `mastery_files/`._\n"
    else:
        # persist parquet
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(CACHE_DIR / "per_ticker_best.parquet", index=False)
        except Exception as e:  # pragma: no cover
            log.warning("parquet write failed: %s", e)

        n_master = int((df["status"] == "MASTERED").sum())
        n_fail = int((df["status"] == "FAILED").sum())

        lines = [
            f"# Per-ticker best results (auto)",
            "",
            f"_Generated {now_iso()} — {len(df)} tickers ({n_master} MASTERED, {n_fail} FAILED). "
            f"Parquet: `s&p500-ticker-mastery/cache/per_ticker_best.parquet`._",
            "",
            "| Ticker | Version | Status | PF | DD | WR | n | RET | Comp |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in df.iterrows():
            def fmt(x, suffix=""):
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return "-"
                if isinstance(x, float):
                    return f"{x:.3f}{suffix}"
                return str(x)
            lines.append(
                f"| {r['ticker']} | {r['version']} | {r['status']} | "
                f"{fmt(r['pf'])} | {fmt(r['dd'])} | {fmt(r['wr'])} | "
                f"{fmt(r['n'])} | {fmt(r['ret'])} | {fmt(r['comp_score'])} |"
            )

        # sweep artifacts (optuna best_params per ticker)
        sweep_rows = []
        if SWEEP_DIR.exists():
            for td in sorted(SWEEP_DIR.iterdir()):
                if not td.is_dir():
                    continue
                summary = td / "sweep_summary.json"
                bp = td / "best_params.json"
                if bp.exists():
                    try:
                        data = json.loads(bp.read_text())
                        sweep_rows.append((td.name, json.dumps(data)[:200]))
                    except Exception:
                        pass
                elif summary.exists():
                    try:
                        data = json.loads(summary.read_text())
                        # extract first fold best_params
                        fr = data.get("fold_results") or []
                        bp_data = fr[0].get("best_params") if fr else {}
                        sweep_rows.append((td.name, json.dumps(bp_data)[:200]))
                    except Exception:
                        pass
        if sweep_rows:
            lines += ["", "## Optuna sweep best_params (first fold)", ""]
            lines += ["| Ticker | best_params |", "|---|---|"]
            for tk, p in sweep_rows:
                lines.append(f"| {tk} | `{p}` |")

        body = "\n".join(lines)
    return safe_write_memory(MEM / "project_per_ticker_best.md", body)


# ----------------------------- 1b. pacing history ----------------------------
def write_pacing_history() -> bool:
    transitions: list[dict] = []
    last_regime: str | None = None
    total_lines = 0
    if PACING_HISTORY.exists():
        try:
            with PACING_HISTORY.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    total_lines += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    reg = row.get("regime")
                    if reg != last_regime:
                        transitions.append({
                            "ts": row.get("ts", ""),
                            "from": last_regime or "(start)",
                            "to": reg,
                            "pace_ratio": row.get("pace_ratio"),
                            "week_used_pct": row.get("week_used_pct"),
                            "week_elapsed_pct": row.get("week_elapsed_pct"),
                            "model_default": row.get("recommended_model_default"),
                        })
                        last_regime = reg
        except Exception as e:  # pragma: no cover
            log.warning("pacing_history read failed: %s", e)

    current = {}
    if PACING_STATE.exists():
        try:
            current = json.loads(PACING_STATE.read_text())
        except Exception:
            pass

    lines = [
        "# Pacing regime history (auto)",
        "",
        f"_Generated {now_iso()} — {len(transitions)} transitions across "
        f"{total_lines} pacing snapshots._",
        "",
    ]
    if current:
        lines += [
            "## Current state",
            "",
            f"- ts: {current.get('ts')}",
            f"- regime: **{current.get('regime')}**",
            f"- week_used_pct: {current.get('week_used_pct')}",
            f"- week_elapsed_pct: {current.get('week_elapsed_pct')}",
            f"- pace_ratio: {current.get('pace_ratio')}",
            f"- model_default: {current.get('recommended_model_default')}",
            f"- reset_at: {current.get('reset_at')}",
            "",
        ]
    if transitions:
        lines += [
            "## Regime transitions",
            "",
            "| ts | from | to | pace_ratio | week_used% | week_elapsed% | model_default |",
            "|---|---|---|---|---|---|---|",
        ]
        # only last 200 to keep file bounded
        for t in transitions[-200:]:
            lines.append(
                f"| {t['ts']} | {t['from']} | {t['to']} | "
                f"{t['pace_ratio']} | {t['week_used_pct']} | "
                f"{t['week_elapsed_pct']} | {t['model_default']} |"
            )
    else:
        lines.append("_No transitions found in pacing_history.jsonl._")
    return safe_write_memory(MEM / "project_pacing_history.md", "\n".join(lines))


# ----------------------------- 1c. daemon health -----------------------------
def write_daemon_health() -> bool:
    rows: list[tuple[str, str, str]] = []
    try:
        out = subprocess.check_output(["launchctl", "list"], text=True, timeout=15)
        for line in out.splitlines():
            if "com.zg" not in line:
                continue
            # launchctl list columns: PID Status Label
            parts = line.split(None, 2)
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
            else:
                rows.append(("?", "?", line.strip()))
    except Exception as e:  # pragma: no cover
        log.warning("launchctl failed: %s", e)

    # Optional log-tail stats for known daemons
    tail_stats: list[tuple[str, str]] = []
    if LOG_DIR.exists():
        cutoff = _dt.datetime.now() - _dt.timedelta(days=7)
        for lf in sorted(LOG_DIR.glob("*.log")):
            try:
                size = lf.stat().st_size
                mtime = _dt.datetime.fromtimestamp(lf.stat().st_mtime)
                if mtime < cutoff:
                    continue
                tail_stats.append((lf.name, f"{size} bytes · last {mtime.isoformat(timespec='seconds')}"))
            except Exception:
                continue

    lines = [
        "# Daemon health snapshot (auto)",
        "",
        f"_Generated {now_iso()}_",
        "",
        "## `launchctl list | grep com.zg`",
        "",
        "| PID | Status | Label |",
        "|---|---|---|",
    ]
    if not rows:
        lines.append("| - | - | _(no com.zg daemons registered)_ |")
    else:
        for pid, status, label in rows:
            lines.append(f"| {pid} | {status} | {label} |")
    lines += [
        "",
        "## Recent log activity (last 7 days)",
        "",
    ]
    if tail_stats:
        lines.append("| log | stat |")
        lines.append("|---|---|")
        for n, s in tail_stats:
            lines.append(f"| {n} | {s} |")
    else:
        lines.append("_No logs in the last 7 days under `s&p500-ticker-mastery/logs/`._")
    return safe_write_memory(MEM / "project_daemon_health.md", "\n".join(lines))


# ----------------------------- 1d. paper-trade P&L ---------------------------
def _safe_read_parquet(p: Path):
    try:
        return pd.read_parquet(p)
    except Exception as e:  # pragma: no cover
        log.warning("parquet read failed %s: %s", p, e)
        return None


def write_paper_trade_pnl() -> bool:
    daily_rows: list[dict] = []
    if PAPER_DAILY.exists():
        for f in sorted(PAPER_DAILY.glob("*.parquet")):
            df = _safe_read_parquet(f)
            if df is None or df.empty:
                continue
            row = {"file": f.name, "rows": int(len(df))}
            # try common P&L column names
            for col in ("pnl", "realized_pnl", "daily_pnl", "total_pnl"):
                if col in df.columns:
                    try:
                        row[col] = float(df[col].sum())
                    except Exception:
                        pass
            daily_rows.append(row)

    state_rows: list[dict] = []
    if PAPER_STATE.exists():
        for f in sorted(PAPER_STATE.glob("*_state.json")):
            try:
                data = json.loads(f.read_text())
                state_rows.append({
                    "date": data.get("date", f.stem),
                    "mode": data.get("mode", "?"),
                    "halted": data.get("halted", False),
                    "realized_pnl": data.get("realized_pnl", 0.0),
                    "open_positions": len(data.get("positions") or {}),
                    "closed_trades": len(data.get("closed_trades") or []),
                })
            except Exception:
                continue

    lines = [
        "# Paper-trade P&L (auto)",
        "",
        f"_Generated {now_iso()}_",
        "",
    ]
    if state_rows:
        total = sum(r["realized_pnl"] for r in state_rows)
        lines += [
            f"**Cumulative realized P&L:** ${total:,.2f} across {len(state_rows)} sessions.",
            "",
            "## Daily state files (`paper_trade/state/*_state.json`)",
            "",
            "| date | mode | halted | realized_pnl | open | closed |",
            "|---|---|---|---|---|---|",
        ]
        for r in state_rows:
            lines.append(
                f"| {r['date']} | {r['mode']} | {r['halted']} | "
                f"${r['realized_pnl']:.2f} | {r['open_positions']} | {r['closed_trades']} |"
            )
    else:
        lines.append("_No state files in `paper_trade/state/`._")

    lines += ["", "## Daily parquets (`paper_trade/daily/*.parquet`)", ""]
    if daily_rows:
        lines.append("| file | rows | known columns |")
        lines.append("|---|---|---|")
        for r in daily_rows:
            extras = {k: v for k, v in r.items() if k not in ("file", "rows")}
            lines.append(f"| {r['file']} | {r['rows']} | `{json.dumps(extras)}` |")
    else:
        lines.append("_No daily parquets yet._")

    return safe_write_memory(MEM / "project_paper_trade_pnl.md", "\n".join(lines))


# ----------------------------- 2. MEMORY.md index ----------------------------
def update_memory_index() -> bool:
    if not MEMORY_INDEX.exists():
        log.warning("MEMORY.md missing: %s", MEMORY_INDEX)
        return False

    # backup
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"MEMORY.{now_stamp()}.md.bak"
    shutil.copy2(MEMORY_INDEX, backup_path)
    log.info("Backed up MEMORY.md → %s", backup_path.name)

    text = MEMORY_INDEX.read_text(encoding="utf-8")

    auto_block = (
        f"{AUTO_BANNER_BEGIN}\n"
        + "\n".join(INDEX_LINES[f] for f in OWNED_FILES)
        + f"\n{AUTO_BANNER_END}\n"
    )

    if AUTO_BANNER_BEGIN in text and AUTO_BANNER_END in text:
        new_text = re.sub(
            re.escape(AUTO_BANNER_BEGIN) + r".*?" + re.escape(AUTO_BANNER_END) + r"\n?",
            auto_block,
            text,
            flags=re.DOTALL,
        )
    else:
        # append to end (preserve any trailing newline)
        if not text.endswith("\n"):
            text += "\n"
        new_text = text + "\n" + auto_block

    if new_text == text:
        log.info("MEMORY.md already up-to-date — no change.")
        return False
    MEMORY_INDEX.write_text(new_text, encoding="utf-8")
    log.info("MEMORY.md updated with auto-aggregated index block.")
    return True


# ----------------------------- main ------------------------------------------
def _write_heartbeat(status: str = "running") -> None:
    """Atomic heartbeat write (six-fail-fix F7 — 2026-05-20)."""
    try:
        import tempfile
        hb_dir = AI_ROOT / "state" / "memory_auto_save"
        hb_dir.mkdir(parents=True, exist_ok=True)
        hb = hb_dir / "heartbeat.json"
        import time as _t
        import json as _j
        payload = _j.dumps({"ts": int(_t.time()), "pid": os.getpid(), "status": status})
        with tempfile.NamedTemporaryFile(dir=str(hb_dir), delete=False, mode="w") as tmp:
            tmp.write(payload)
            tmp_path = tmp.name
        os.replace(tmp_path, hb)
    except Exception:
        pass


def main() -> int:
    log.info("memory_auto_save start (pid=%d)", os.getpid())
    _write_heartbeat("start")
    results = {
        "per_ticker_best": False,
        "pacing_history": False,
        "daemon_health": False,
        "paper_trade_pnl": False,
        "memory_index": False,
    }
    try:
        results["per_ticker_best"] = write_per_ticker_best()
    except Exception as e:
        log.exception("per_ticker_best failed: %s", e)
    try:
        results["pacing_history"] = write_pacing_history()
    except Exception as e:
        log.exception("pacing_history failed: %s", e)
    try:
        results["daemon_health"] = write_daemon_health()
    except Exception as e:
        log.exception("daemon_health failed: %s", e)
    try:
        results["paper_trade_pnl"] = write_paper_trade_pnl()
    except Exception as e:
        log.exception("paper_trade_pnl failed: %s", e)
    try:
        results["memory_index"] = update_memory_index()
    except Exception as e:
        log.exception("memory_index failed: %s", e)
    log.info("memory_auto_save done — %s", results)
    _write_heartbeat("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
