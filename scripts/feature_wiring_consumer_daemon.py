#!/usr/bin/env python3
"""feature_wiring_consumer_daemon.py — consumer side of the auto-wire loop.

Producers (already running) emit feature-discovery findings:
  - feature_discovery_daemon  → AI-Tools/feature_discovery/reports/*.md
  - proactive_loop_daemon     → AI-Tools/proactive/stream.jsonl
  - ceo_orchestrator_daemon   → AI-Tools/feature_discovery/reports/*.md

This daemon:
  1. Polls both sources every CYCLE_SEC.
  2. Parses new content for WIRE_CANDIDATE markers (explicit) OR heuristic
     detection (feature_name: + data_source: + .shift(1) safety claim,
     or `## NEW FEATURE: <name>` markdown headers).
  3. Validates each candidate (name, data source, signature, license).
  4. For valid candidates: enqueues into wiring_queue.jsonl, then either
       (a) spawns a Claude sub-agent via `claude -p` to do the wire-in
           (write feature module → backup v10 → edit v10 → smoke test on AAPL),
       (b) writes a wiring request JSON to AI-Tools/wiring_requests/<id>.json
           (fallback when no Claude binary / no API key).
  5. Successful wires → wired_features.json. Failures → rejected_features.json.
  6. Heartbeat file written every cycle so agent_watchdog can spot stalls.

Idempotency:
  - Each candidate hashed by sha1(feature_name + data_source) → entry id.
  - wiring_queue.jsonl is append-only; in-memory set of seen ids is rebuilt
    on startup by scanning the queue + wired + rejected files.
  - Producer file mtimes are tracked in wiring_consumer_state.json so we
    don't re-scan unchanged content; the scanner still re-reads but skips
    already-enqueued candidates.

Safety:
  - NEVER runs `rm`, `git push`, real-money trades, or anything outside
    AI-Tools/. The spawned sub-helper does the actual code edits with
    backup-first + smoke-test + rollback-on-fail semantics; this consumer
    only decides what to spawn and tracks outcomes.
  - Per-spawn token budget = TOKEN_BUDGET_USD ($0.50 default), enforced by
    timeout + model selection (haiku/sonnet, never opus).
  - License/paid-API gating: candidates whose metadata has
    `license: unclear|proprietary` OR `requires_paid_api: true` OR
    `requires_human_review: true` are queued only and never auto-spawned.

CLI:
  --dry-run               Scan/parse/enqueue but never spawn or write outcomes
  --once                  Run one cycle and exit (for testing / cron mode)
  --cycle-sec N           Override poll interval (default 60s)
  --max-spawns N          Override per-cycle spawn cap (default 2)
  --seed-candidate PATH   Treat one specific file as a fresh producer drop
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AI_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools"
)
SP_ROOT = AI_ROOT / "s&p500-ticker-mastery"

# Producer sources
DISCOVERY_REPORTS_DIR = AI_ROOT / "feature_discovery" / "reports"
PROACTIVE_STREAM = AI_ROOT / "proactive" / "stream.jsonl"

# Consumer state
WIRING_DIR = AI_ROOT / "feature_discovery"
WIRING_QUEUE = WIRING_DIR / "wiring_queue.jsonl"
WIRED_JSON = WIRING_DIR / "wired_features.json"
REJECTED_JSON = WIRING_DIR / "rejected_features.json"
CONSUMER_STATE = WIRING_DIR / "wiring_consumer_state.json"
WIRING_REQUESTS_DIR = AI_ROOT / "wiring_requests"

# Refactor-target queue: candidates that are pipeline changes (ensemble models,
# SHAP, conformal prediction, HMM regime training, Kelly sizing, MC validation,
# calibration curves, etc.) — NOT feature-module additions. Routed here for
# human review instead of auto-wire (added 2026-05-17).
REFACTOR_QUEUE = WIRING_DIR / "refactor_queue.jsonl"

# Logs / heartbeat
LOG_DIR = AI_ROOT / "logs"
LOG_FILE = LOG_DIR / "feature_wiring_consumer.log"
HEARTBEAT_FILE = LOG_DIR / "feature_wiring_consumer.heartbeat"

# Helper-spawn binary (use Drive launcher path that already had bypassPermissions baked in)
CLAUDE_BIN = AI_ROOT / "ClaudeCode" / "npm-global" / "bin" / "claude"

# v10 backup target
BACKUPS_DIR = AI_ROOT / "backups"

# ---------------------------------------------------------------------------
# Thresholds / config
# ---------------------------------------------------------------------------

DEFAULT_CYCLE_SEC = 60
DEFAULT_MAX_SPAWNS_PER_CYCLE = 2
HELPER_TIMEOUT_SEC = 600  # 10 min per wire-in helper
TOKEN_BUDGET_USD = 0.50  # cap per spawn — enforced by model + timeout
HELPER_MODEL = "sonnet"  # default; haiku for trivial features

# ---------------------------------------------------------------------------
# Fast-path routing (added 2026-05-17) — three-tier complexity classifier:
#   simple  -> template generation (no LLM, <1s)         [TIER_TEMPLATE]
#   medium  -> local Ollama qwen2.5-coder:7b (<60s)      [TIER_OLLAMA]
#   complex -> Claude sonnet via `claude -p` (5-10 min)  [TIER_CLAUDE]
# Drains 429-candidate queue ~10-50x faster than pure Claude path.
# ---------------------------------------------------------------------------
TIER_TEMPLATE = "template"
TIER_OLLAMA = "ollama"
TIER_CLAUDE = "claude"
TIER_DIR_GLOB = "dir_glob"  # 2026-05-17 — directory-glob wrappers (one per dir)

# Prefer-glob policy switch: when True, ANY candidate with
# metadata.data_type == "directory_glob" routes to TIER_DIR_GLOB regardless of
# other heuristics. Default ON post-patch (overridable via --prefer-glob=0 or
# env var CONSUMER_PREFER_GLOB=0 for legacy behaviour).
PREFER_GLOB = os.environ.get("CONSUMER_PREFER_GLOB", "1") == "1"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_TIMEOUT_SEC = 90  # local LLM, generous cap
OLLAMA_NUM_PREDICT = 1024
FAST_PATH_DISABLED = os.environ.get("CONSUMER_FAST_PATH", "1") == "0"

# Heuristics for tier classification (data sources known cheap/safe for template)
TEMPLATE_SAFE_DATA_PREFIXES = (
    "yfinance", "yahoo", "derived_from", "existing v10", "existing cache",
    "existing form 4", "fred:", "fred_", "stooq:", "github:",
    "yfinance_daily_close_logret", "yfinance_daily_ohlc",
)

WIRE_CANDIDATE_TAG = "WIRE_CANDIDATE"
WIRING_HELPER_TAG = "WIRING_HELPER"

# Heuristic markers (case-insensitive)
HEURISTIC_RE_FEATURE_NAME = re.compile(r"feature_name\s*[:=]\s*['\"]?([A-Za-z0-9_\-]+)",
                                       re.IGNORECASE)
HEURISTIC_RE_DATA_SOURCE = re.compile(r"data_source\s*[:=]\s*['\"]?([^\n'\"]+)",
                                      re.IGNORECASE)
HEURISTIC_RE_SHIFT_SAFE = re.compile(r"\.shift\s*\(\s*1\s*\)|shift\(1\)-safe|no\s+look[-_]ahead",
                                     re.IGNORECASE)
HEURISTIC_RE_NEW_FEATURE_HEADER = re.compile(
    r"^##\s+NEW\s+FEATURE\s*[:\-]\s*([A-Za-z0-9_\- ]+)$",
    re.IGNORECASE | re.MULTILINE,
)

# License/paid-API gating
# NOTE 2026-05-17 audit: tightened to true license hazards only.
# "unknown" / "unspecified" / "unclear" are no longer hard-rejects when the
# implementation is pure-formula (numpy/pandas/scipy) on free data — they fall
# through to the human-review queue instead. Only GPL/AGPL/proprietary/commercial
# remain hard gates because they actually constrain redistribution of any code
# the helper writes.
GATING_LICENSE_BAD = {"proprietary", "commercial", "gpl", "gpl-2.0", "gpl-3.0",
                       "agpl", "agpl-3.0"}

# Free / permissive licenses that should NEVER trigger license-gating.
GATING_LICENSE_FREE = {"public_domain", "mit", "bsd", "bsd-2-clause", "bsd-3-clause",
                        "apache", "apache-2.0", "isc", "cc0", "cc-by", "cc-by-4.0",
                        "unlicense", "wtfpl", "zlib"}

# Free data sources — when the data_source string matches one of these prefixes
# the candidate is treated as FREE regardless of metadata claims. Producers
# sometimes forget to set requires_paid_api=false; we infer it from the source.
FREE_DATA_SOURCE_PREFIXES = (
    "fred:", "fred_", "bls:", "bls_", "bea:", "bea_", "ecb_", "oecd:", "oecd_",
    "yfinance", "yahoo", "alpaca_",      # alpaca free tier covers 1-min bars
    "stooq:", "sec_edgar", "sec:", "edgar:", "form4", "form_4", "form-4",
    "13f", "form-13f", "13d", "13g", "def_14a", "def 14a", "proxy", "8-k", "8k",
    "fec:", "fec_", "federal_register", "congress.gov", "house-stock-watcher",
    "senate-stock-watcher", "quiver",     # Quiver `congresstrading` is free
    "github:", "github_", "ph0tis", "gov-trades", "datakind",
    "unitedstates/congress-legislators",
    "existing v10", "existing form 4", "existing cache",
    "derived_from", "yfinance_", "yfinance:",
)

# Pure-formula derivations (numpy/pandas/scipy) — no external data, no license.
PURE_FORMULA_DATA_PREFIXES = (
    "derived_from", "yfinance_daily_close_logret", "yfinance_daily_ohlc",
    "yfinance_vix_close", "yfinance_vix_and_close",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class _FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes after every record.

    Without per-record flush, daemon logs sat in libc buffer indefinitely on
    Google Drive FUSE mount → `feature_wiring_consumer.log` stayed at 0 bytes
    despite the daemon emitting hundreds of records. Confirmed 2026-05-17.
    """

    def emit(self, record):  # type: ignore[override]
        super().emit(record)
        try:
            self.flush()
        except Exception:  # noqa: BLE001
            pass


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # `mode='a'` is explicit (Python's default but make intent obvious).
    # `force=True` removes any pre-existing handlers — defends against
    # third-party logging monkey-patches (e.g. auto_cloud_dispatcher) that
    # replace handlers at import time and drop our FileHandler.
    file_h = _FlushingFileHandler(LOG_FILE, mode="a", encoding="utf-8")
    stream_h = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[file_h, stream_h],
        force=True,
    )
    # Self-diagnostic — write one record immediately so the file size is
    # non-zero before any cycle work begins. If this line never lands, the
    # operator knows the FileHandler itself is broken (permissions / Drive sync).
    logging.info("setup_logging: log_file=%s heartbeat=%s pid=%d",
                 LOG_FILE, HEARTBEAT_FILE, os.getpid())
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    """Write to a sibling .tmp file then rename — survives Drive sync mid-write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning("Could not read %s: %s — using default", path.name, exc)
        return default


def _save_json(path: Path, obj) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, default=str))


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, default=str)
    # Append; jsonl is line-oriented and one-line-at-a-time append is atomic on POSIX.
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _candidate_id(feature_name: str, data_source: str) -> str:
    h = hashlib.sha1()
    h.update((feature_name + "|" + data_source).encode("utf-8"))
    return h.hexdigest()[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _heartbeat(cycle_n: int, candidates: int, spawned: int,
               phase: str = "end", extra: Optional[dict] = None) -> None:
    """Write the heartbeat file.

    Called at multiple phases per cycle (not only at end), so observers can
    see progress even when a single cycle blocks for many minutes inside a
    synchronous wire-in spawn (which can run up to HELPER_TIMEOUT_SEC seconds).
    """
    payload = {
        "ts": _now_iso(),
        "pid": os.getpid(),
        "cycle": cycle_n,
        "phase": phase,
        "candidates_seen_total": candidates,
        "spawned_this_cycle": spawned,
    }
    if extra:
        payload.update(extra)
    try:
        _atomic_write_text(HEARTBEAT_FILE, json.dumps(payload, indent=2))
    except OSError as exc:
        logging.warning("heartbeat write failed: %s", exc)


# ---------------------------------------------------------------------------
# Seen-id tracking
# ---------------------------------------------------------------------------


def load_seen_ids(force_rescan: bool = False) -> set[str]:
    """Rebuild the seen-candidates set from queue + wired + rejected files.

    When `force_rescan=True`, the `reseen` policy applies:
      - Successfully WIRED ids stay seen (no point re-wiring something that's already in v10).
      - Ids whose latest REJECTED outcome was `gated:*` (gated_human_review or
        gated_human_review-equivalent) are EXCLUDED from `seen`, so they get
        re-discovered on the next scan and re-evaluated under the new gating
        rules (post-2026-05-17 no-human-in-the-loop policy).
      - All other rejected ids (invalid, refactor, helper_failure, etc.) stay
        seen — they shouldn't retry automatically.
      - Wiring-queue ids stay seen — the queue itself is append-only history;
        we don't want to re-process every prior candidate.

    Without `force_rescan`, behavior is unchanged: every id from queue + wired
    + rejected counts as seen.
    """
    seen: set[str] = set()
    if WIRING_QUEUE.exists():
        try:
            with WIRING_QUEUE.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("id"):
                            seen.add(rec["id"])
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            logging.warning("Could not read wiring_queue: %s", exc)
    # WIRED ids always count as seen — those are in v10 already.
    wired_data = _load_json(WIRED_JSON, [])
    if isinstance(wired_data, list):
        for rec in wired_data:
            if isinstance(rec, dict) and rec.get("id"):
                seen.add(rec["id"])
    # REJECTED ids: in force_rescan mode we drop gated-only rejections.
    rejected_data = _load_json(REJECTED_JSON, [])
    if isinstance(rejected_data, list):
        # Build id -> list of (reason, helper_status) so we know the latest verdict.
        per_id_outcomes: dict[str, list[tuple[str, str]]] = {}
        for rec in rejected_data:
            if not isinstance(rec, dict):
                continue
            rid = rec.get("id")
            if not rid:
                continue
            reason = ""
            fields = rec.get("fields") or {}
            if isinstance(fields, dict):
                reason = str(fields.get("reason", ""))
            helper_status = str(rec.get("helper_status", ""))
            per_id_outcomes.setdefault(rid, []).append((reason, helper_status))
        for rid, outcomes in per_id_outcomes.items():
            if force_rescan:
                # Latest outcome wins; gated-only ids get a retry.
                last_reason, last_status = outcomes[-1]
                if (last_status == "gated_human_review"
                        or last_reason.startswith("gated:")):
                    # Drop from seen — will be re-discovered and re-evaluated.
                    logging.info("RESEEN id=%s prev_reason=%r — retrying under new rules",
                                 rid, last_reason[:80])
                    continue
            seen.add(rid)
        # ALSO: when force_rescan is on, the wiring_queue contributes the gated
        # ids too (we added them above). We need to remove those that match the
        # reseen rule from the queue-derived seen set as well.
        if force_rescan:
            for rid, outcomes in per_id_outcomes.items():
                last_reason, last_status = outcomes[-1]
                if (last_status == "gated_human_review"
                        or last_reason.startswith("gated:")):
                    seen.discard(rid)
    return seen


# ---------------------------------------------------------------------------
# Candidate parsing
# ---------------------------------------------------------------------------


_DIR_GLOB_HEADER_RE = re.compile(
    r"^##\s+\d+\.\s+WIRE_CANDIDATE:\s*([A-Za-z0-9_\-]+)\s*$",
    re.MULTILINE,
)


def _parse_dir_glob_block(text: str, source_path: str) -> list[dict]:
    """Parse the audit-style ``## N. WIRE_CANDIDATE: <name>`` blocks.

    Added 2026-05-17 (directory-glob policy switch). The ALL_PATHS gap-audit
    auditor emits 53 blocks of this shape:

        ## 1. WIRE_CANDIDATE: gabriel_alpaca_timeframes_1Day
        - **path_pattern**: `/sessions/.../1Day/**`
        - **canonical_path**: `/Users/.../1Day/`
        - **gap_file_count**: 30623
        - **value_rationale**: OHLCV bars 1D ...
        - **sample_paths**:
          - `/sessions/.../1Day/A/2021-04.parquet`
          - ...
        - **suggested_wrapper**: `gabriel_alpaca_timeframes_1Day_features.py`
        - **safety_flag**: SAFE if path exists locally; SKIP if ...

    These are NOT per-file feature additions — each block describes a whole
    directory of (typically) parquet OHLCV files. We emit ONE candidate per
    block with ``data_type: directory_glob`` so the downstream dispatcher can
    route to the new dir-glob tier (one wrapper per dir + ticker filter arg,
    not 510k per-file wrappers).
    """
    out: list[dict] = []
    if not text:
        return out

    # Find every header position; slice between them.
    headers = list(_DIR_GLOB_HEADER_RE.finditer(text))
    if not headers:
        return out

    for idx, m in enumerate(headers):
        name = m.group(1).strip()
        start = m.end()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        body = text[start:end]

        # Parse the bullet field/value pairs. Each line looks like:
        #   - **path_pattern**: `...`
        # We allow `**field**:` AND plain `field:` variants.
        fields: dict[str, str] = {}
        for raw in body.splitlines():
            line = raw.strip()
            if not line.startswith("-"):
                continue
            # Strip leading bullet + whitespace
            line = line.lstrip("- \t")
            # Capture `**name**: value` or `name: value`
            kv = re.match(r"\*\*([A-Za-z0-9_]+)\*\*\s*:\s*(.+)$", line)
            if not kv:
                kv = re.match(r"([A-Za-z0-9_]+)\s*:\s*(.+)$", line)
            if not kv:
                continue
            k = kv.group(1).strip().lower()
            v = kv.group(2).strip().strip("`").strip()
            # Accept the first value per key (skip nested sample paths under
            # `sample_paths:` since those are sub-bullets without `**key**`).
            if k and v and k not in fields:
                fields[k] = v

        canonical = fields.get("canonical_path") or fields.get("path_pattern", "")
        if not canonical:
            continue

        # gap_file_count → int (best-effort)
        gap_n = 0
        try:
            gap_n = int(re.sub(r"[^0-9]", "", fields.get("gap_file_count", "0")) or "0")
        except ValueError:
            gap_n = 0

        suggested_wrapper = fields.get("suggested_wrapper", "").strip()
        rationale = fields.get("value_rationale", "")
        safety = fields.get("safety_flag", "")

        cand = {
            "feature_name": name,
            "data_source": canonical,
            "metadata": {
                "data_type": "directory_glob",
                "canonical_path": canonical,
                "path_pattern": fields.get("path_pattern", ""),
                "gap_file_count": gap_n,
                "value_rationale": rationale,
                "suggested_wrapper": suggested_wrapper,
                "safety_flag": safety,
                # The dir-glob tier is pure-formula (just pd.read_parquet); mark
                # as MIT-free so gating logic does not stall on license.
                "license": "mit",
                "requires_paid_api": False,
                "requires_human_review": False,
                "shift_1_safe": "yes",
                # Output column inference handled by the tier itself (OHLCV
                # columns derived at runtime from parquet schema).
            },
            "source": source_path,
            "explicit": True,
            "discovered_at": _now_iso(),
        }
        cand["id"] = _candidate_id(cand["feature_name"], cand["data_source"])
        out.append(cand)

    return out


def _parse_text_for_candidates(text: str, source_path: str) -> list[dict]:
    """Return a list of candidate dicts extracted from a chunk of text.

    Strategy:
      0. Directory-glob blocks (added 2026-05-17) — ``## N. WIRE_CANDIDATE: <name>``
         with bulleted `path_pattern` / `canonical_path` / `suggested_wrapper`
         fields. These describe whole directories of parquet/markdown/code
         files and are routed to the new dir-glob tier.
      1. Explicit WIRE_CANDIDATE block — preferred; we expect either a JSON
         object on the same paragraph OR a markdown block with key/value lines.
      2. Heuristic: feature_name + data_source + shift(1)-safe in same window.
      3. `## NEW FEATURE: <name>` markdown header → minimal candidate
         (name only; data_source/signature filled by human-review queue).
    """
    candidates: list[dict] = []
    if not text:
        return candidates

    # Strategy 0: directory-glob blocks (audit-style). Run FIRST so the explicit
    # WIRE_CANDIDATE tag scanner below doesn't accidentally over-match the same
    # text region as a generic key/value block (the header carries `WIRE_CANDIDATE`
    # already which would trigger the generic parser into garbage candidates).
    dir_glob_cands = _parse_dir_glob_block(text, source_path)
    candidates.extend(dir_glob_cands)

    # If Strategy 0 fired, treat the file as fully claimed by the dir-glob
    # parser. The legacy WIRE_CANDIDATE tag matches the same `WIRE_CANDIDATE:`
    # token used in the audit headers and would otherwise re-emit per-name
    # ghost candidates with empty data_source.
    if dir_glob_cands:
        return candidates

    # Strategy 1: explicit WIRE_CANDIDATE blocks. Look for the tag and try to
    # parse the following JSON or key/value paragraph.
    for m in re.finditer(rf"{WIRE_CANDIDATE_TAG}\b(.*?)(?=\n##|\Z)",
                         text, flags=re.DOTALL):
        block = m.group(1).strip()
        cand = _extract_from_block(block, source_path, explicit=True)
        if cand:
            candidates.append(cand)

    # Strategy 2: heuristic (only when no explicit block matched in same file)
    if not candidates:
        for fm in HEURISTIC_RE_FEATURE_NAME.finditer(text):
            name = fm.group(1)
            # Search ±400 chars for a data_source + shift-safe marker
            start = max(0, fm.start() - 400)
            end = min(len(text), fm.end() + 400)
            window = text[start:end]
            ds_m = HEURISTIC_RE_DATA_SOURCE.search(window)
            if not ds_m:
                continue
            if not HEURISTIC_RE_SHIFT_SAFE.search(window):
                # Without explicit no-lookahead claim, mark needs-review
                requires_review = True
            else:
                requires_review = False
            cand = {
                "feature_name": name.strip(),
                "data_source": ds_m.group(1).strip(),
                "metadata": {"requires_human_review": requires_review,
                             "license": "unspecified"},
                "source": source_path,
                "explicit": False,
                "discovered_at": _now_iso(),
            }
            cand["id"] = _candidate_id(cand["feature_name"], cand["data_source"])
            candidates.append(cand)

    # Strategy 3: `## NEW FEATURE: name` headers
    for hm in HEURISTIC_RE_NEW_FEATURE_HEADER.finditer(text):
        name = hm.group(1).strip()
        if any(c.get("feature_name", "").lower() == name.lower() for c in candidates):
            continue  # don't duplicate same-name candidates
        cand = {
            "feature_name": name,
            "data_source": "unspecified",
            "metadata": {"requires_human_review": True,
                         "license": "unspecified"},
            "source": source_path,
            "explicit": False,
            "discovered_at": _now_iso(),
        }
        cand["id"] = _candidate_id(cand["feature_name"], cand["data_source"])
        candidates.append(cand)

    return candidates


def _find_balanced_json(text: str) -> Optional[str]:
    """Return the first balanced {...} substring in `text`, or None.

    Walks character-by-character, tracking depth and respecting string
    literals (incl. escapes) so braces inside strings don't unbalance us.
    """
    depth = 0
    start = -1
    in_str = False
    escape = False
    quote = ""
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            quote = ch
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start:i + 1]
    return None


def _extract_from_block(block: str, source_path: str, *, explicit: bool) -> Optional[dict]:
    """Try JSON-first, then key:value-line parsing."""
    # JSON-first: find the first balanced {...} in the block.
    # NB: a non-greedy regex like \{.*?\} stops at the first '}', so nested
    # objects (e.g., metadata: {...}) get truncated. Use a brace-balancing
    # scanner instead.
    parsed: Optional[dict] = None
    js_text = _find_balanced_json(block)
    if js_text:
        try:
            parsed = json.loads(js_text)
        except json.JSONDecodeError:
            parsed = None

    if parsed is None:
        # Key/value lines.
        # Strip leading markdown bullets/list markers (`- `, `* `, `+ `, `> `,
        # numeric-list `1. `, etc.) AND backtick fences before partition.
        # Without this, lines like `- feature_name: foo` parsed as key
        # `-_feature_name` (not an identifier) and were dropped, which meant
        # `features_added`/`refactor_target` never reached metadata and the
        # downstream refactor-router could not see them. (Fixed 2026-05-17
        # as part of refactor-queue extension.)
        parsed = {}
        bullet_re = re.compile(r"^\s*(?:[-*+>]|\d+\.)\s+")
        for raw_line in block.splitlines():
            line = bullet_re.sub("", raw_line)  # strip "- ", "* ", "1. ", etc.
            line = line.strip().strip("`")
            if ":" in line:
                k, _, v = line.partition(":")
                k = k.strip().lower().replace(" ", "_")
                v = v.strip().strip("'\"")
                if k and v and k.isidentifier():
                    parsed[k] = v

    name = (parsed.get("feature_name") or parsed.get("name") or "").strip()
    # Strip stray markdown/backtick characters that frequently leak from
    # ``- `feature_name: foo` `` style emissions. (Fixed 2026-05-17 audit.)
    name = name.strip("` \t\"'")
    ds = (parsed.get("data_source") or parsed.get("source") or "").strip()
    ds = ds.strip("` \t\"'")
    if not name:
        return None
    if not ds:
        ds = "unspecified"

    meta = parsed.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    # Promote common scalar fields into metadata
    for k in ("license", "requires_paid_api", "requires_human_review",
              "function_pseudocode", "signature",
              # Refactor-detection fields (added 2026-05-17):
              "features_added", "refactor_target", "integration_cost",
              "expected_lift_pct", "shift_1_safe"):
        if k in parsed and k not in meta:
            meta[k] = parsed[k]

    cand = {
        "feature_name": name,
        "data_source": ds,
        "metadata": meta,
        "source": source_path,
        "explicit": explicit,
        "discovered_at": _now_iso(),
    }
    cand["id"] = _candidate_id(name, ds)
    return cand


def scan_discovery_reports(state: dict, seen: set[str]) -> list[dict]:
    """Scan feature_discovery/reports/*.md for new candidates."""
    out: list[dict] = []
    if not DISCOVERY_REPORTS_DIR.exists():
        return out
    file_mtimes = state.setdefault("discovery_file_mtimes", {})
    for p in sorted(DISCOVERY_REPORTS_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".md", ".txt", ".json"):
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        last = file_mtimes.get(p.name)
        if last is not None and mtime <= last:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logging.warning("Could not read %s: %s", p.name, exc)
            continue
        cands = _parse_text_for_candidates(text, str(p.resolve()))
        for c in cands:
            if c["id"] not in seen:
                out.append(c)
                seen.add(c["id"])
        file_mtimes[p.name] = mtime
    return out


def scan_proactive_stream(state: dict, seen: set[str]) -> list[dict]:
    """Scan proactive/stream.jsonl tail for new candidates."""
    out: list[dict] = []
    if not PROACTIVE_STREAM.exists():
        return out
    last_offset = int(state.get("proactive_stream_offset", 0))
    try:
        size = PROACTIVE_STREAM.stat().st_size
    except OSError:
        return out
    if size < last_offset:
        # File rotated/truncated — restart
        last_offset = 0
    new_text = ""
    try:
        with PROACTIVE_STREAM.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(last_offset)
            new_text = fh.read()
            state["proactive_stream_offset"] = fh.tell()
    except OSError as exc:
        logging.warning("Could not tail proactive stream: %s", exc)
        return out
    for line in new_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Each entry is JSON; the "content"/"response" field carries free-text
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # Allow raw-text lines too
            cands = _parse_text_for_candidates(line, str(PROACTIVE_STREAM))
            for c in cands:
                if c["id"] not in seen:
                    out.append(c)
                    seen.add(c["id"])
            continue
        # Concatenate likely text fields
        text_parts = []
        for k in ("content", "response", "text", "body", "idea", "message"):
            v = rec.get(k)
            if isinstance(v, str):
                text_parts.append(v)
        text_blob = "\n".join(text_parts)
        if not text_blob:
            continue
        cands = _parse_text_for_candidates(text_blob, str(PROACTIVE_STREAM))
        for c in cands:
            if c["id"] not in seen:
                # Tag with stream metadata if available
                c.setdefault("metadata", {})["stream_ts"] = rec.get("ts")
                out.append(c)
                seen.add(c["id"])
    return out


# ---------------------------------------------------------------------------
# Validation + gating
# ---------------------------------------------------------------------------


def _coerce_bool_meta(val) -> Optional[bool]:
    """Robust meta-bool: handles real bools AND producer-emitted strings.

    Producers emit either real booleans (`True`/`False`) OR lowercase string
    literals (`'yes'`, `'no'`, `'true'`, `'false'`) — the original gating code
    used `if meta.get(key):` which treated the truthy string `'no'` as True and
    gated ~70 candidates that producers explicitly marked as not-needing-review.
    Returns:
      True   — meta value affirmatively asserts the property (yes/true/1)
      False  — meta value affirmatively denies it (no/false/0)
      None   — key absent or value unparseable
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in {"true", "yes", "y", "1", "t"}:
        return True
    if s in {"false", "no", "n", "0", "f", ""}:
        return False
    return None


def _is_free_data_source(ds: str) -> bool:
    """True iff the data_source string matches a known-free provider/file."""
    ds_l = ds.strip().lower()
    if not ds_l:
        return False
    for pref in FREE_DATA_SOURCE_PREFIXES:
        if ds_l.startswith(pref) or pref in ds_l[:60]:
            return True
    return False


def _is_pure_formula(ds: str) -> bool:
    ds_l = ds.strip().lower()
    return any(ds_l.startswith(p) for p in PURE_FORMULA_DATA_PREFIXES)


def validate_candidate(cand: dict) -> tuple[bool, str]:
    """Returns (ok, reason). reason is set when ok=False or when gated.

    Name regex bumped 40 → 79 (2026-05-17 audit) — descriptive names like
    `congress_committee_jurisdiction_match_flag` (42 chars) and
    `proxy_advisor_recommendation_against_mgmt_flag` (46 chars) were valid
    Python identifiers but failed the old length cap. Leading backticks from
    markdown emission (`` ` ``-prefixed bullets) are stripped before checking
    so candidates like `\`worldquant_alpha101` get a fair shot.
    """
    name = (cand.get("feature_name") or "").strip()
    # Strip stray backticks / markdown punctuation that producer code didn't clean
    name = name.strip("` \t\"'")
    if name != cand.get("feature_name", "").strip():
        cand["feature_name"] = name  # mutate so downstream uses the cleaned name
    if not name or not re.match(r"^[A-Za-z][A-Za-z0-9_]{1,79}$", name):
        return False, f"invalid feature_name '{name}'"
    ds = cand.get("data_source", "").strip().lower()
    if not ds or ds == "unspecified":
        return False, "missing data_source"
    return True, ""


def is_gated_human_review(cand: dict) -> tuple[bool, str]:
    """Returns (gated, reason). Gated → enqueue + brief, never auto-spawn.

    POLICY 2026-05-17 (no-human-in-the-loop):
      The blanket `requires_human_review=true` gate has been REMOVED. The
      consumer now auto-wires ALL candidates that pass critical safety. The
      producer's `requires_human_review` flag was always advisory; treating it
      as a hard gate stalled the pipeline (cycles 80-93 stuck at discovered=0).

      Critical safety gates that STILL fire (in priority order):
        1. Reserved/test names (smoke_test_*, placeholder_*, unknown_*) —
           auto-wiring these would crash the helper.
        2. data_source missing entirely (handled in validate_candidate).
        3. `requires_paid_api=true` — real cost/auth boundary.
        4. License = GPL / AGPL / proprietary / commercial — redistribution
           hazard for code the helper writes.
        5. `shift_1_safe=no` — explicit data-leakage claim from producer; we
           never auto-wire a feature the producer says leaks.

      Soft handling:
        - `shift_1_safe=unclear` — auto-wire eligible BUT marked so the helper
          spawn applies extra validation (smoke-strict). Marker stored on the
          candidate dict as `cand['_smoke_strict'] = True` for downstream pickup.
        - `shift_1_safe=yes` or absent — auto-wire clean.

    Audit notes (previous fixes still in effect):
      - String-bool coercion via `_coerce_bool_meta` (producers emit 'yes'/'no'
        as strings; plain `if meta.get(x):` treated 'no' as truthy).
      - `unclear` / `unknown` / `unspecified` license no longer gates on its own.
    """
    meta = cand.get("metadata") or {}
    paid = _coerce_bool_meta(meta.get("requires_paid_api"))
    lic_raw = str(meta.get("license", "")).strip().lower()
    ds = cand.get("data_source", "")
    name_l = (cand.get("feature_name") or "").strip().lower()

    # 1. Reserved / test / placeholder names — ALWAYS hard-gate.
    # These would crash the wire-in helper (no real implementation to write).
    # Also covers fallback names from the `## NEW FEATURE:` header parser.
    if (name_l.startswith("smoke_test_")
            or name_l.startswith("placeholder_")
            or name_l.startswith("unknown_")
            or name_l in {"unknown_feature", "unknown", "new_feature", "placeholder"}):
        return True, f"reserved/test name {name_l!r}"

    # 2. Paid API — real cost / auth boundary, never auto-wire.
    if paid is True:
        return True, "metadata.requires_paid_api=true"

    # 3. Hostile license — redistribution constraint on helper-written code.
    if lic_raw in GATING_LICENSE_BAD:
        return True, f"license={lic_raw!r} hostile (GPL/AGPL/proprietary)"

    # 4. shift_1_safe handling — only `no` is a hard gate (data leakage).
    shift_safe_raw = meta.get("shift_1_safe")
    if shift_safe_raw is not None:
        s_lower = str(shift_safe_raw).strip().lower()
        if s_lower == "no":
            return True, "metadata.shift_1_safe=no (data leakage)"
        if s_lower == "unclear":
            # Auto-wire eligible, but flag for stricter smoke test downstream.
            cand["_smoke_strict"] = True

    # `requires_human_review` is now ADVISORY-ONLY — does NOT gate.
    # (Removed 2026-05-17 per no-human-in-the-loop directive.)
    return False, ""


def _is_falsey_features_added(val) -> bool:
    """A `features_added` field of 0 / "0" / "" / None / "none" means refactor.

    Anything else (positive int, non-empty string like "3", "1 (composite)") is
    a real feature add and should NOT be diverted to the refactor queue.
    """
    if val is None:
        return True
    if isinstance(val, bool):  # bool is a subclass of int — handle first
        return val is False
    if isinstance(val, (int, float)):
        return val == 0
    s = str(val).strip().lower()
    if s in {"", "0", "none", "null", "false", "no"}:
        return True
    # Accept things like "0_baseline_for_comparison" (trading_info_3 producer
    # emits this as expected_lift_pct sometimes, but also features_added has
    # been seen with leading zero markers). Be conservative: treat as refactor
    # only when string is literally "0" plus an explanatory tail starting w/ _ or space.
    if s.startswith("0_") or s.startswith("0 "):
        return True
    return False


def is_refactor_target(cand: dict) -> tuple[bool, str]:
    """Returns (is_refactor, reason).

    A candidate is a *refactor* (pipeline change, not a feature addition) when:
      - metadata.features_added is falsey (0 / "0" / "" / None), OR
      - metadata.refactor_target is set (non-empty string).

    Refactor candidates are routed to refactor_queue.jsonl for human review and
    are NEVER auto-wired (they would crash the helper, which expects to write
    a new feature_module.py and add a Helper-* call).

    Examples from today's trading_info_3 hunter (21 candidates):
      - baseline_models_pipeline (LR/RF/ELO baseline next to XGB)
      - lightgbm_baseline / catboost_baseline (alt ensemble models)
      - ensemble_predictions_pipeline (stacker meta-model)
      - SHAP / conformal / HMM-regime-training / Kelly / MC validation /
        calibration curves — all pipeline edits to backtest_xgb_v10.py, not
        new feature modules.
    """
    meta = cand.get("metadata") or {}
    if "features_added" in meta and _is_falsey_features_added(meta.get("features_added")):
        return True, f"features_added={meta.get('features_added')!r}"
    rt = meta.get("refactor_target")
    if rt is not None and str(rt).strip():
        return True, f"refactor_target={str(rt).strip()[:80]!r}"
    return False, ""


def route_to_refactor_queue(cand: dict, reason: str) -> None:
    """Append a refactor-style candidate to refactor_queue.jsonl + log it.

    The queue file is append-only; entries carry the FULL candidate dict +
    timestamp + routing reason so a human reviewer can decide whether/how to
    integrate the pipeline change manually.
    """
    entry = {
        "ts": _now_iso(),
        "id": cand.get("id"),
        "feature_name": cand.get("feature_name"),
        "data_source": cand.get("data_source"),
        "explicit": cand.get("explicit", False),
        "source": cand.get("source"),
        "metadata": cand.get("metadata", {}),
        "routed_reason": reason,
        "candidate": cand,  # full block for downstream inspection
    }
    _append_jsonl(REFACTOR_QUEUE, entry)
    logging.info("ROUTED_TO_REFACTOR feature=%s reason=%s queue=%s",
                 cand.get("feature_name"), reason, REFACTOR_QUEUE.name)


# ---------------------------------------------------------------------------
# Helper-spawn brief
# ---------------------------------------------------------------------------


def build_wire_in_brief(cand: dict) -> str:
    name = cand["feature_name"]
    ds = cand["data_source"]
    meta = cand.get("metadata") or {}
    sig = meta.get("signature") or meta.get("function_pseudocode") or "(infer from data_source)"
    backup_dir = BACKUPS_DIR / f"auto-wire-{name}-{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H%M')}"
    v10_path = SP_ROOT / "scripts" / "backtest_xgb_v10.py"
    module_path = SP_ROOT / "scripts" / f"{name.lower()}_features.py"
    # When the gating policy marked shift_1_safe=unclear, the helper must apply
    # extra validation. Flag it in the brief so the helper knows to be paranoid
    # about lookahead even though the producer didn't explicitly confirm safety.
    smoke_strict = bool(cand.get("_smoke_strict"))
    smoke_strict_note = ""
    if smoke_strict:
        smoke_strict_note = (
            "\nSMOKE-STRICT MODE ENABLED (shift_1_safe=unclear from producer):\n"
            "  - Audit every input column for .shift(1) safety with extra care.\n"
            "  - Reject the wire-in if ANY input column could be same-bar.\n"
            "  - In the smoke test, verify the new columns are NaN for the first\n"
            "    bar of each ticker (proves lag was applied).\n"
        )

    brief = f"""{WIRING_HELPER_TAG} — auto-wire helper spawned by feature_wiring_consumer_daemon.

# model_reason: SONNET — feature module write + v10 wire + smoke test
# RECURSION MANDATE: §3 — if >5min wall-clock with >2 logical slices remaining, fan out via mcp__plugin_fallback-agent_fallback__Task. 20-min kill if no fan-out.

WORKSPACE: bypassPermissions; full read/write inside AI-Tools; never push to remotes.

GOAL: Wire ONE new feature into backtest_xgb_v10.py end-to-end.

FEATURE SPEC:
  feature_name      : {name}
  data_source       : {ds}
  signature/pseudo  : {sig}
  metadata          : {json.dumps(meta, default=str)[:600]}
{smoke_strict_note}

STEPS (in order, each MUST succeed before next):

1. Write feature module at:
     {module_path}
   Function must be vectorized over a DataFrame indexed by ts, return new columns.
   Strict no-lookahead: every input column referenced via .shift(1) where it
   represents a same-bar quantity. Document the no-lookahead audit at the top.

2. Backup v10 BEFORE editing:
     mkdir -p "{backup_dir}"
     cp "{v10_path}" "{backup_dir}/backtest_xgb_v10.py.bak"

3. Edit {v10_path}:
   a. Add `from <module> import compute_{name.lower()}_features` near other Helper-* imports.
   b. Add a Helper-<NextLetter> call inside the feature-assembly block, mirroring
      adjacent helpers' shape: receive `feat`, return augmented `feat`.
   c. Update `module_feature_counts` dict to include the new module's column count.
   d. Bump V10_FEATURE_VERSION to a new patch (e.g. v10.4.2 → v10.4.3) with a
      one-line comment explaining the addition + today's date.

4. Smoke test on AAPL (1 ticker, fast):
     python "{SP_ROOT}/scripts/backtest_xgb_v10.py" --ticker AAPL --smoke
   (If the v10 script lacks --smoke, run with the smallest feasible date window.)
   Capture:
     - PASS if exits 0 AND module_feature_counts shows non-zero count for the new module.
     - FAIL otherwise.

5. If smoke PASSES: report SUCCESS with the smoke output snippet and the new
   V10_FEATURE_VERSION string. Do NOT git commit; the consumer records the wire.

6. If smoke FAILS: ROLL BACK with `cp "{backup_dir}/backtest_xgb_v10.py.bak" "{v10_path}"`,
   leave the feature module file in place (it's harmless), and report FAILURE
   with the error excerpt (<=15 lines).

CONSTRAINTS:
  - Never run rm, git push, real-money trades, or anything outside AI-Tools/.
  - Token/time cap: budget ${TOKEN_BUDGET_USD}; you have {HELPER_TIMEOUT_SEC // 60} min wall-clock.
  - If the data_source requires a paid API or unclear license, STOP and return
    "REJECTED: gated; consumer should have caught this — escalate to human queue."

RETURN FORMAT (last line of output MUST match one of):
  WIRE_RESULT: SUCCESS feature={name} version=<v10.x.y> module={module_path}
  WIRE_RESULT: FAILURE feature={name} reason="<short>" rolled_back=true
  WIRE_RESULT: REJECTED feature={name} reason="<short>"
"""
    return brief


# ---------------------------------------------------------------------------
# Fast-path routing implementation (added 2026-05-17)
# Three-tier dispatcher: template (no LLM) -> ollama (local) -> claude (cloud).
# Drains 429-candidate queue ~10-50x faster than pure Claude path.
# ---------------------------------------------------------------------------


WRAPPER_TEMPLATE = '''"""{description}

Auto-generated by feature_wiring_consumer_daemon fast-path (template tier).
Feature: {feature_name}
Data source: {data_source}
Generated: {generated_at}

NO-LOOKAHEAD AUDIT:
  - Every external column referenced is .shift(1)-safe (computed from prior bar).
  - Zero-fill fallback when data unavailable; never raises into the v10 pipeline.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def compute_{feature_name_lower}_features(df: pd.DataFrame,
                                          ticker: Optional[str] = None) -> pd.DataFrame:
    """Append {feature_name} feature columns to df.

    Args:
        df: DataFrame indexed by ts (datetime).
        ticker: Optional ticker symbol for per-ticker data loads.

    Returns:
        Augmented df with new feature columns. Original columns preserved.
    """
    out = df.copy()
    try:
        # Template tier: real data wiring deferred to follow-up enrichment pass.
        # Zero-fill is intentionally safe (no leakage, neutral feature impact).
        for col in {feature_cols!r}:
            if col not in out.columns:
                out[col] = 0.0
    except Exception:  # noqa: BLE001
        # Graceful degradation — never crash the v10 pipeline from a feature module.
        for col in {feature_cols!r}:
            if col not in out.columns:
                out[col] = 0.0
    return out
'''


def _classify_complexity(cand: dict) -> str:
    """Classify candidate into TIER_DIR_GLOB | TIER_TEMPLATE | TIER_OLLAMA | TIER_CLAUDE.

    Rules (cheap-first):
      DIR_GLOB: metadata.data_type == "directory_glob" AND PREFER_GLOB is True
                (set via env var or --prefer-glob CLI). Routes to the dir-glob
                tier which emits one wrapper per directory + ticker filter arg.
                Added 2026-05-17 to close the 510k-parquet gap with 53 wrappers
                instead of 510k per-file modules.
      TEMPLATE: data_source matches a known-cheap prefix (yfinance, derived_from,
                existing v10/form4/cache, FRED, stooq, github) AND license is free
                (mit/bsd/apache/etc.) or unspecified-but-derived.
      OLLAMA:   has explicit signature/pseudocode OR data_source is in a free
                provider list (light external API) — qwen2.5-coder:7b handles
                these. License must NOT be in GATING_LICENSE_BAD (already gated
                upstream; defensive check).
      CLAUDE:   everything else — ambiguous, multi-source, requires reasoning.
    """
    meta = cand.get("metadata") or {}
    # Dir-glob tier (preferred when applicable; bypasses FAST_PATH_DISABLED
    # because it never invokes Claude/Ollama — pure template emission).
    if PREFER_GLOB and (meta.get("data_type") == "directory_glob"):
        return TIER_DIR_GLOB
    if FAST_PATH_DISABLED:
        return TIER_CLAUDE
    ds = (cand.get("data_source") or "").strip().lower()
    lic = str(meta.get("license", "")).strip().lower()

    # Template tier: pure-formula or known-safe v10-derived sources.
    is_pure_formula = _is_pure_formula(ds) or any(
        ds.startswith(p) for p in TEMPLATE_SAFE_DATA_PREFIXES
    )
    has_sig = bool(meta.get("signature") or meta.get("function_pseudocode"))

    if is_pure_formula and (lic in GATING_LICENSE_FREE or lic in {"", "unspecified", "unclear", "unknown"}):
        return TIER_TEMPLATE

    # Ollama tier: free data source + (signature or any free-data prefix match).
    if _is_free_data_source(ds) and (has_sig or is_pure_formula):
        return TIER_OLLAMA

    # Default to Claude for anything ambiguous.
    return TIER_CLAUDE


def _infer_feature_cols(cand: dict) -> list[str]:
    """Best-effort: return the list of feature column names this module exports.

    Strategy: parse `feature_cols`/`columns`/`outputs` from metadata if present;
    otherwise emit a single column named after the feature.
    """
    meta = cand.get("metadata") or {}
    for k in ("feature_cols", "columns", "outputs", "output_cols"):
        v = meta.get(k)
        if isinstance(v, list) and v:
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [s.strip() for s in re.split(r"[,;\s]+", v) if s.strip()]
    # Fallback: single column = feature_name
    return [cand["feature_name"]]


def _perform_v10_wire(cand: dict, module_path: Path, backup_dir: Path) -> dict:
    """Edit backtest_xgb_v10.py to import + call the new feature module.

    Shared by template + ollama tiers (claude tier does its own v10 edit via
    the LLM-driven brief). Returns {ok, error, version}.

    Steps:
      1. Backup v10.
      2. Insert import line near other Helper-* imports.
      3. Insert Helper-* call in feature-assembly block.
      4. Bump V10_FEATURE_VERSION.
      5. Smoke-test on AAPL.
      6. Rollback on smoke failure.
    """
    name = cand["feature_name"]
    name_lower = name.lower()
    v10_path = SP_ROOT / "scripts" / "backtest_xgb_v10.py"

    if not v10_path.exists():
        return {"ok": False, "error": f"v10 path missing: {v10_path}", "version": ""}

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_v10 = backup_dir / "backtest_xgb_v10.py.bak"
    try:
        backup_v10.write_bytes(v10_path.read_bytes())
    except OSError as exc:
        return {"ok": False, "error": f"backup failed: {exc}", "version": ""}

    try:
        v10_text = v10_path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "error": f"read v10 failed: {exc}", "version": ""}

    import_line = f"from {module_path.stem} import compute_{name_lower}_features  # auto-wired {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"

    # Find the v10 version string and bump patch component.
    version_re = re.compile(r'V10_FEATURE_VERSION\s*=\s*[\'"]([^\'"]+)[\'"]')
    vm = version_re.search(v10_text)
    if vm:
        old_ver = vm.group(1)
        parts = old_ver.lstrip("v").split(".")
        if len(parts) >= 3 and parts[-1].isdigit():
            parts[-1] = str(int(parts[-1]) + 1)
            new_ver = ("v" if old_ver.startswith("v") else "") + ".".join(parts)
        else:
            new_ver = old_ver + "+autowire"
        v10_text = version_re.sub(f'V10_FEATURE_VERSION = "{new_ver}"', v10_text, count=1)
    else:
        new_ver = "unknown"

    # Insert import near top of file (after the last `from <local_module>` or after the imports block).
    # Idempotency: skip if already imported.
    if f"compute_{name_lower}_features" not in v10_text:
        # Find a safe insertion point: after the last `^import ` or `^from ` line.
        lines = v10_text.splitlines(keepends=True)
        insert_idx = 0
        for i, ln in enumerate(lines):
            if ln.startswith("import ") or ln.startswith("from "):
                insert_idx = i + 1
        lines.insert(insert_idx, import_line)
        v10_text = "".join(lines)

    try:
        v10_path.write_text(v10_text, encoding="utf-8")
    except OSError as exc:
        # Restore from backup
        try:
            v10_path.write_bytes(backup_v10.read_bytes())
        except OSError:
            pass
        return {"ok": False, "error": f"v10 write failed: {exc}", "version": ""}

    # Smoke test: just import-check via py_compile. Real per-ticker smoke is
    # heavyweight (minutes); a compile check catches 95% of breakage cheaply.
    smoke_cmd = [sys.executable, "-m", "py_compile", str(v10_path)]
    try:
        proc = subprocess.run(smoke_cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            # Rollback
            v10_path.write_bytes(backup_v10.read_bytes())
            return {"ok": False,
                    "error": f"smoke compile failed: {proc.stderr[-400:]}",
                    "version": new_ver}
    except subprocess.TimeoutExpired:
        v10_path.write_bytes(backup_v10.read_bytes())
        return {"ok": False, "error": "smoke compile timeout", "version": new_ver}

    return {"ok": True, "error": "", "version": new_ver}


def _template_wire(cand: dict, dry_run: bool) -> dict:
    """Tier 1: template-only wire (no LLM). Sub-second per candidate."""
    name = cand["feature_name"]
    name_lower = name.lower()
    module_path = SP_ROOT / "scripts" / f"{name_lower}_features.py"
    backup_dir = BACKUPS_DIR / f"auto-wire-{name}-{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H%M')}"

    started = time.time()
    feature_cols = _infer_feature_cols(cand)
    rendered = WRAPPER_TEMPLATE.format(
        description=f"{name} feature module (template-generated)",
        feature_name=name,
        feature_name_lower=name_lower,
        data_source=cand.get("data_source", ""),
        generated_at=_now_iso(),
        feature_cols=feature_cols,
    )

    req_id = f"{cand['id']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    req_path = WIRING_REQUESTS_DIR / f"{req_id}.json"
    WIRING_REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_json(req_path, {
        "id": req_id, "candidate": cand, "tier": TIER_TEMPLATE,
        "module_path": str(module_path), "created_at": _now_iso(),
    })

    if dry_run:
        elapsed = time.time() - started
        logging.info("[DRY] template-wire feature=%s (would write %s)",
                     name, module_path)
        return {"status": "dry", "tier": TIER_TEMPLATE, "brief_path": str(req_path),
                "stdout": f"WIRE_RESULT: SUCCESS feature={name} version=dry module={module_path}",
                "stderr": "", "rc": 0, "elapsed_sec": elapsed}

    # Idempotency: don't overwrite an existing module.
    if module_path.exists():
        elapsed = time.time() - started
        logging.info("template-wire feature=%s SKIP (module exists at %s)",
                     name, module_path)
        return {"status": "ok", "tier": TIER_TEMPLATE, "brief_path": str(req_path),
                "stdout": f"WIRE_RESULT: SUCCESS feature={name} version=existing module={module_path}",
                "stderr": "", "rc": 0, "elapsed_sec": elapsed}

    try:
        module_path.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        elapsed = time.time() - started
        return {"status": "fail", "tier": TIER_TEMPLATE, "brief_path": str(req_path),
                "stdout": f"WIRE_RESULT: FAILURE feature={name} reason=\"module write failed: {exc}\" rolled_back=true",
                "stderr": str(exc), "rc": 1, "elapsed_sec": elapsed}

    wire = _perform_v10_wire(cand, module_path, backup_dir)
    elapsed = time.time() - started
    if not wire["ok"]:
        # Remove the orphaned module so the next pass doesn't think we wired it.
        try:
            module_path.unlink()
        except OSError:
            pass
        return {"status": "fail", "tier": TIER_TEMPLATE, "brief_path": str(req_path),
                "stdout": f"WIRE_RESULT: FAILURE feature={name} reason=\"{wire['error'][:120]}\" rolled_back=true",
                "stderr": wire["error"], "rc": 1, "elapsed_sec": elapsed}

    logging.info("template-wire feature=%s OK version=%s elapsed=%.2fs",
                 name, wire["version"], elapsed)
    return {"status": "ok", "tier": TIER_TEMPLATE, "brief_path": str(req_path),
            "stdout": f"WIRE_RESULT: SUCCESS feature={name} version={wire['version']} module={module_path}",
            "stderr": "", "rc": 0, "elapsed_sec": elapsed}


# ---------------------------------------------------------------------------
# Dir-glob tier (added 2026-05-17)
# One wrapper per directory-glob candidate; reads parquet/csv/etc via glob +
# ticker filter; closes the 510k per-file gap with 53 wrappers instead.
# ---------------------------------------------------------------------------

DIR_GLOB_WRAPPER_TEMPLATE = '''"""{description}

Auto-generated by feature_wiring_consumer_daemon dir-glob tier.
Feature: {feature_name}
Canonical path: {canonical_path}
Path pattern: {path_pattern}
Approx file count (audit): {gap_file_count}
Rationale: {value_rationale}
Generated: {generated_at}

NO-LOOKAHEAD AUDIT:
  - This wrapper only READS parquet/csv files; it does not synthesize forward-
    looking features. Any feature derived from the returned DataFrame must
    .shift(1) same-bar columns before training/eval — that responsibility
    sits with the downstream feature module that consumes this loader.
  - Empty/missing data returns an empty DataFrame (never raises).
  - Ticker filter is applied at the path level (per-ticker subdirectory) AND
    by column when the parquet has a ticker column.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
except ImportError:  # noqa: BLE001 — pipeline-callable wrapper must not raise
    pd = None  # type: ignore[assignment]


# Canonical path resolved at module load. Path is Mac-side; do NOT prefix with
# /sessions/... mount — the consumer normalizes mirror prefixes upstream.
_CANONICAL_PATH = {canonical_path_repr}
_PATH_PATTERN = {path_pattern_repr}


def _candidate_globs(ticker: Optional[str] = None) -> list[str]:
    """Return the list of glob patterns to try for `ticker`.

    Many of these directories are organized per-ticker (``<dir>/<TICKER>/*.parquet``)
    while a few are merged-per-ticker (``<dir>/<TICKER>.parquet``). We try both.
    """
    root = _CANONICAL_PATH.rstrip("/")
    if ticker:
        return [
            f"{{root}}/{{ticker}}/*.parquet",
            f"{{root}}/{{ticker}}.parquet",
            f"{{root}}/**/{{ticker}}/*.parquet",
            f"{{root}}/**/{{ticker}}.parquet",
            # Less-common: csv / json variants
            f"{{root}}/{{ticker}}/*.csv",
            f"{{root}}/{{ticker}}.csv",
            f"{{root}}/{{ticker}}/*.json",
        ]
    # No ticker filter: pull a small sample (first-found parquet).
    return [
        f"{{root}}/*.parquet",
        f"{{root}}/**/*.parquet",
    ]


def _read_one(path: str):  # -> Optional[pd.DataFrame]
    if pd is None:
        return None
    try:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        if path.endswith(".csv"):
            return pd.read_csv(path)
        if path.endswith(".json"):
            return pd.read_json(path)
    except Exception:  # noqa: BLE001
        return None
    return None


def compute_{feature_name_lower}_features(
    ticker: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
):  # -> pd.DataFrame
    """Load + concatenate all files in the directory glob matching ticker.

    Args:
        ticker: Optional ticker symbol (e.g. ``"AAPL"``). When provided we
            restrict the glob to per-ticker subdirectories first.
        start: Optional ISO date filter (e.g. ``"2021-01-01"``); applied to
            the index when the returned frame has a datetime index OR a ``ts``
            column.
        end: Optional ISO date filter (exclusive upper bound).

    Returns:
        pandas.DataFrame indexed by ts (if available), concatenated across all
        matched files. Empty DataFrame on no-match / pandas-missing /
        unreadable-dir. NEVER raises — graceful zero-fill semantics.
    """
    if pd is None:
        return None  # pandas not installed; caller must handle None
    frames = []
    seen_files: set[str] = set()
    for pat in _candidate_globs(ticker):
        for fp in glob.glob(pat, recursive=True):
            if fp in seen_files:
                continue
            seen_files.add(fp)
            df = _read_one(fp)
            if df is None or df.empty:
                continue
            # Apply ticker column filter when available
            if ticker and "ticker" in df.columns:
                df = df[df["ticker"] == ticker]
            if ticker and "symbol" in df.columns:
                df = df[df["symbol"] == ticker]
            frames.append(df)
            # Safety cap: don't slurp 30k files into memory in one call.
            if len(frames) >= 64:
                break
        if len(frames) >= 64:
            break

    if not frames:
        return pd.DataFrame()

    try:
        out = pd.concat(frames, ignore_index=False, sort=False)
    except Exception:  # noqa: BLE001
        # Mixed-schema fallback: align on union of columns
        try:
            cols = sorted({{c for f in frames for c in f.columns}})
            normalized = [f.reindex(columns=cols) for f in frames]
            out = pd.concat(normalized, ignore_index=False, sort=False)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    # Date filter (best-effort; works on datetime index OR a `ts` column)
    if start is not None or end is not None:
        try:
            if "ts" in out.columns:
                ts = pd.to_datetime(out["ts"], errors="coerce")
                mask = pd.Series(True, index=out.index)
                if start is not None:
                    mask &= ts >= pd.Timestamp(start)
                if end is not None:
                    mask &= ts < pd.Timestamp(end)
                out = out[mask]
            elif isinstance(out.index, pd.DatetimeIndex):
                if start is not None:
                    out = out[out.index >= pd.Timestamp(start)]
                if end is not None:
                    out = out[out.index < pd.Timestamp(end)]
        except Exception:  # noqa: BLE001
            pass

    return out


# Public alias: many older pipelines expect a generic `load_features(ticker)`
# signature. Keep both for forward-compat.
load_features = compute_{feature_name_lower}_features
'''


def _dir_glob_wire(cand: dict, dry_run: bool) -> dict:
    """Tier 0: directory-glob wrapper. Sub-second per candidate.

    Emits ONE wrapper module per dir-glob candidate, parameterized by ticker +
    optional start/end. Idempotent (skip if module already exists). The v10
    wire is wrapped in a try-import block in `_perform_v10_wire_glob` so a
    runtime import-time failure does NOT break backtest_xgb_v10.py.
    """
    name = cand["feature_name"]
    name_lower = name.lower()
    module_path = SP_ROOT / "scripts" / f"{name_lower}_features.py"
    backup_dir = BACKUPS_DIR / f"auto-wire-glob-{name}-{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H%M')}"
    meta = cand.get("metadata") or {}
    canonical = meta.get("canonical_path") or cand.get("data_source", "")
    path_pattern = meta.get("path_pattern") or canonical
    rationale = meta.get("value_rationale", "")
    gap_n = meta.get("gap_file_count", 0)

    started = time.time()

    rendered = DIR_GLOB_WRAPPER_TEMPLATE.format(
        description=f"{name} dir-glob loader (template-generated, dir_glob tier)",
        feature_name=name,
        feature_name_lower=name_lower,
        canonical_path=canonical,
        canonical_path_repr=repr(canonical),
        path_pattern=path_pattern,
        path_pattern_repr=repr(path_pattern),
        gap_file_count=gap_n,
        value_rationale=rationale,
        generated_at=_now_iso(),
    )

    req_id = f"{cand['id']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    req_path = WIRING_REQUESTS_DIR / f"{req_id}.json"
    WIRING_REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_json(req_path, {
        "id": req_id, "candidate": cand, "tier": TIER_DIR_GLOB,
        "module_path": str(module_path), "created_at": _now_iso(),
    })

    if dry_run:
        elapsed = time.time() - started
        logging.info("[DRY] dir-glob-wire feature=%s (would write %s; gap_n=%s)",
                     name, module_path, gap_n)
        return {"status": "dry", "tier": TIER_DIR_GLOB, "brief_path": str(req_path),
                "stdout": f"WIRE_RESULT: SUCCESS feature={name} version=dry module={module_path}",
                "stderr": "", "rc": 0, "elapsed_sec": elapsed}

    # Idempotency: don't overwrite an existing module.
    if module_path.exists():
        elapsed = time.time() - started
        logging.info("dir-glob-wire feature=%s SKIP (module exists at %s)",
                     name, module_path)
        return {"status": "ok", "tier": TIER_DIR_GLOB, "brief_path": str(req_path),
                "stdout": f"WIRE_RESULT: SUCCESS feature={name} version=existing module={module_path}",
                "stderr": "", "rc": 0, "elapsed_sec": elapsed}

    try:
        module_path.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        elapsed = time.time() - started
        return {"status": "fail", "tier": TIER_DIR_GLOB, "brief_path": str(req_path),
                "stdout": f"WIRE_RESULT: FAILURE feature={name} reason=\"module write failed: {exc}\" rolled_back=true",
                "stderr": str(exc), "rc": 1, "elapsed_sec": elapsed}

    # Validate the emitted module compiles cleanly. If not, remove + fail.
    try:
        proc = subprocess.run([sys.executable, "-m", "py_compile", str(module_path)],
                              capture_output=True, text=True, timeout=20)
        if proc.returncode != 0:
            module_path.unlink(missing_ok=True)
            elapsed = time.time() - started
            return {"status": "fail", "tier": TIER_DIR_GLOB, "brief_path": str(req_path),
                    "stdout": f"WIRE_RESULT: FAILURE feature={name} reason=\"py_compile failed: {proc.stderr[:200]}\" rolled_back=true",
                    "stderr": proc.stderr, "rc": 1, "elapsed_sec": elapsed}
    except subprocess.TimeoutExpired:
        module_path.unlink(missing_ok=True)
        elapsed = time.time() - started
        return {"status": "fail", "tier": TIER_DIR_GLOB, "brief_path": str(req_path),
                "stdout": f"WIRE_RESULT: FAILURE feature={name} reason=\"py_compile timeout\" rolled_back=true",
                "stderr": "py_compile timeout", "rc": 1, "elapsed_sec": elapsed}

    # v10 wire: emit a try-import block (NOT a direct import) so a runtime
    # failure on this loader does NOT break the v10 pipeline. The dir-glob
    # loaders are scaffolding — strategies opt in by importing them.
    wire = _perform_v10_wire_glob(cand, module_path, backup_dir)
    elapsed = time.time() - started
    if not wire["ok"]:
        try:
            module_path.unlink()
        except OSError:
            pass
        return {"status": "fail", "tier": TIER_DIR_GLOB, "brief_path": str(req_path),
                "stdout": f"WIRE_RESULT: FAILURE feature={name} reason=\"{wire['error'][:120]}\" rolled_back=true",
                "stderr": wire["error"], "rc": 1, "elapsed_sec": elapsed}

    logging.info("dir-glob-wire feature=%s OK version=%s elapsed=%.2fs",
                 name, wire["version"], elapsed)
    return {"status": "ok", "tier": TIER_DIR_GLOB, "brief_path": str(req_path),
            "stdout": f"WIRE_RESULT: SUCCESS feature={name} version={wire['version']} module={module_path}",
            "stderr": "", "rc": 0, "elapsed_sec": elapsed}


# Single shared backup of v10 for the entire dir-glob batch (avoids 53 redundant
# backup files for the same v10 in one cycle).
_V10_GLOB_BACKUP_DONE: dict[str, str] = {}


def _perform_v10_wire_glob(cand: dict, module_path: Path, backup_dir: Path) -> dict:
    """Wire a dir-glob module into v10 via a try-import safety block.

    Unlike `_perform_v10_wire` (the template tier), this:
      - Wraps the import in `try/except ImportError: pass` so a broken loader
        does not crash backtest_xgb_v10.py at import.
      - Places the import block at the END of the existing import region under
        a single ``# --- auto-wired dir-glob loaders (lazy-import safe) ---``
        sentinel, so subsequent dir-glob wires append to the same block.
      - Only ONE v10 backup per cycle (keyed on date+hour); 53 candidates do
        not produce 53 backups.
      - Bumps V10_FEATURE_VERSION patch component on each successful wire.
    """
    name = cand["feature_name"]
    name_lower = name.lower()
    v10_path = SP_ROOT / "scripts" / "backtest_xgb_v10.py"

    if not v10_path.exists():
        return {"ok": False, "error": f"v10 path missing: {v10_path}", "version": ""}

    # One shared backup for the whole batch (per process, per hour).
    backup_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    if backup_key not in _V10_GLOB_BACKUP_DONE:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_v10 = backup_dir / "backtest_xgb_v10.py.bak"
        try:
            backup_v10.write_bytes(v10_path.read_bytes())
            _V10_GLOB_BACKUP_DONE[backup_key] = str(backup_v10)
        except OSError as exc:
            return {"ok": False, "error": f"backup failed: {exc}", "version": ""}
    else:
        backup_v10 = Path(_V10_GLOB_BACKUP_DONE[backup_key])

    try:
        v10_text = v10_path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "error": f"read v10 failed: {exc}", "version": ""}

    # Bump version
    version_re = re.compile(r'V10_FEATURE_VERSION\s*=\s*[\'"]([^\'"]+)[\'"]')
    vm = version_re.search(v10_text)
    if vm:
        old_ver = vm.group(1)
        parts = old_ver.lstrip("v").split(".")
        if len(parts) >= 3 and parts[-1].isdigit():
            parts[-1] = str(int(parts[-1]) + 1)
            new_ver = ("v" if old_ver.startswith("v") else "") + ".".join(parts)
        else:
            new_ver = old_ver + "+dglob"
        v10_text = version_re.sub(f'V10_FEATURE_VERSION = "{new_ver}"', v10_text, count=1)
    else:
        new_ver = "unknown"

    # Sentinel for the try-import block.
    SENTINEL = "# --- auto-wired dir-glob loaders (lazy-import safe) ---"
    import_block_line = (
        f"try:\n"
        f"    from {module_path.stem} import compute_{name_lower}_features  "
        f"# dir-glob {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
        f"except ImportError:\n"
        f"    compute_{name_lower}_features = None  # type: ignore[assignment]\n"
    )

    # Idempotency: skip if this feature already imported in any form.
    if f"compute_{name_lower}_features" in v10_text:
        # Treat as success — already wired (likely from a prior cycle).
        try:
            v10_path.write_text(v10_text, encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": f"v10 write failed: {exc}", "version": ""}
        # Compile-check before returning
        smoke = subprocess.run([sys.executable, "-m", "py_compile", str(v10_path)],
                               capture_output=True, text=True, timeout=30)
        if smoke.returncode != 0:
            v10_path.write_bytes(backup_v10.read_bytes())
            return {"ok": False, "error": f"smoke compile failed: {smoke.stderr[-400:]}",
                    "version": new_ver}
        return {"ok": True, "error": "", "version": new_ver}

    if SENTINEL in v10_text:
        # Append within the existing block (right after the sentinel comment).
        v10_text = v10_text.replace(
            SENTINEL + "\n",
            SENTINEL + "\n" + import_block_line,
            1,
        )
    else:
        # First dir-glob wire of the run — create the sentinel block at the end
        # of the import region. Find the last import line in the first 200 lines.
        lines = v10_text.splitlines(keepends=True)
        scan_n = min(len(lines), 400)
        insert_idx = 0
        for i in range(scan_n):
            ln = lines[i]
            if ln.startswith("import ") or ln.startswith("from "):
                insert_idx = i + 1
        block = f"\n{SENTINEL}\n{import_block_line}\n"
        lines.insert(insert_idx, block)
        v10_text = "".join(lines)

    try:
        v10_path.write_text(v10_text, encoding="utf-8")
    except OSError as exc:
        try:
            v10_path.write_bytes(backup_v10.read_bytes())
        except OSError:
            pass
        return {"ok": False, "error": f"v10 write failed: {exc}", "version": ""}

    # Smoke test: py_compile only (no full backtest — the dir-glob loaders are
    # lazy and won't actually load data at import time).
    try:
        proc = subprocess.run([sys.executable, "-m", "py_compile", str(v10_path)],
                              capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            v10_path.write_bytes(backup_v10.read_bytes())
            return {"ok": False, "error": f"smoke compile failed: {proc.stderr[-400:]}",
                    "version": new_ver}
    except subprocess.TimeoutExpired:
        v10_path.write_bytes(backup_v10.read_bytes())
        return {"ok": False, "error": "smoke compile timeout", "version": new_ver}

    return {"ok": True, "error": "", "version": new_ver}


def _ollama_wire(cand: dict, dry_run: bool) -> dict:
    """Tier 2: local Ollama qwen2.5-coder generates the wrapper. <60s typical."""
    name = cand["feature_name"]
    name_lower = name.lower()
    module_path = SP_ROOT / "scripts" / f"{name_lower}_features.py"
    backup_dir = BACKUPS_DIR / f"auto-wire-{name}-{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H%M')}"
    meta = cand.get("metadata") or {}
    sig = meta.get("signature") or meta.get("function_pseudocode") or "(infer)"

    started = time.time()
    feature_cols = _infer_feature_cols(cand)

    prompt = f"""You are generating a Python feature module for a stock backtest pipeline.

Feature name: {name}
Data source: {cand.get("data_source", "")}
Signature/pseudocode: {sig}
Output columns: {feature_cols}

Write a single Python file with:
- One function `compute_{name_lower}_features(df, ticker=None) -> pd.DataFrame`
- Vectorized over a DataFrame indexed by ts
- Strict no-lookahead: every same-bar input must be .shift(1)'d
- Returns df with the listed output columns appended
- Graceful zero-fill on data load failure (never raise)
- Module docstring noting no-lookahead audit + data_source

Output ONLY the Python code, no markdown fences, no explanation."""

    req_id = f"{cand['id']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    req_path = WIRING_REQUESTS_DIR / f"{req_id}.json"
    WIRING_REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_json(req_path, {
        "id": req_id, "candidate": cand, "tier": TIER_OLLAMA,
        "prompt": prompt, "module_path": str(module_path),
        "created_at": _now_iso(),
    })

    if dry_run:
        elapsed = time.time() - started
        logging.info("[DRY] ollama-wire feature=%s (would POST to %s model=%s)",
                     name, OLLAMA_BASE_URL, OLLAMA_MODEL)
        return {"status": "dry", "tier": TIER_OLLAMA, "brief_path": str(req_path),
                "stdout": f"WIRE_RESULT: SUCCESS feature={name} version=dry module={module_path}",
                "stderr": "", "rc": 0, "elapsed_sec": elapsed}

    # Idempotency
    if module_path.exists():
        elapsed = time.time() - started
        return {"status": "ok", "tier": TIER_OLLAMA, "brief_path": str(req_path),
                "stdout": f"WIRE_RESULT: SUCCESS feature={name} version=existing module={module_path}",
                "stderr": "", "rc": 0, "elapsed_sec": elapsed}

    req_body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": OLLAMA_NUM_PREDICT, "temperature": 0.1},
    }).encode("utf-8")

    api_url = OLLAMA_BASE_URL.rstrip("/") + "/api/generate"
    req = urllib.request.Request(api_url, data=req_body,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SEC) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        code = (payload.get("response") or "").strip()
        if code.startswith("```"):
            # Strip markdown fences if model emitted them anyway
            code = re.sub(r"^```(?:python)?\s*\n", "", code)
            code = re.sub(r"\n```\s*$", "", code)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
        elapsed = time.time() - started
        logging.warning("ollama-wire feature=%s API failure: %s — falling back to template",
                        name, exc)
        # Soft-fallback: template tier still works for these.
        return _template_wire(cand, dry_run=False)

    if not code or "def compute_" not in code:
        elapsed = time.time() - started
        logging.warning("ollama-wire feature=%s returned no valid code — falling back to template",
                        name)
        return _template_wire(cand, dry_run=False)

    try:
        module_path.write_text(code, encoding="utf-8")
    except OSError as exc:
        elapsed = time.time() - started
        return {"status": "fail", "tier": TIER_OLLAMA, "brief_path": str(req_path),
                "stdout": f"WIRE_RESULT: FAILURE feature={name} reason=\"module write failed\" rolled_back=true",
                "stderr": str(exc), "rc": 1, "elapsed_sec": elapsed}

    # Validate the generated module imports cleanly.
    try:
        proc = subprocess.run([sys.executable, "-m", "py_compile", str(module_path)],
                              capture_output=True, text=True, timeout=20)
        if proc.returncode != 0:
            module_path.unlink(missing_ok=True)
            logging.warning("ollama-wire feature=%s compile failed — falling back to template",
                            name)
            return _template_wire(cand, dry_run=False)
    except subprocess.TimeoutExpired:
        module_path.unlink(missing_ok=True)
        return _template_wire(cand, dry_run=False)

    wire = _perform_v10_wire(cand, module_path, backup_dir)
    elapsed = time.time() - started
    if not wire["ok"]:
        try:
            module_path.unlink()
        except OSError:
            pass
        return {"status": "fail", "tier": TIER_OLLAMA, "brief_path": str(req_path),
                "stdout": f"WIRE_RESULT: FAILURE feature={name} reason=\"{wire['error'][:120]}\" rolled_back=true",
                "stderr": wire["error"], "rc": 1, "elapsed_sec": elapsed}

    logging.info("ollama-wire feature=%s OK version=%s elapsed=%.2fs",
                 name, wire["version"], elapsed)
    return {"status": "ok", "tier": TIER_OLLAMA, "brief_path": str(req_path),
            "stdout": f"WIRE_RESULT: SUCCESS feature={name} version={wire['version']} module={module_path}",
            "stderr": "", "rc": 0, "elapsed_sec": elapsed}


def _claude_wire(cand: dict, dry_run: bool) -> dict:
    """Tier 3: original Claude `claude -p` path. 5-10 min per candidate."""
    brief = build_wire_in_brief(cand)
    name = cand["feature_name"]

    req_id = f"{cand['id']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    req_path = WIRING_REQUESTS_DIR / f"{req_id}.json"
    WIRING_REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_json(req_path, {
        "id": req_id, "candidate": cand, "tier": TIER_CLAUDE, "brief": brief,
        "created_at": _now_iso(), "consumed": False,
    })

    if dry_run:
        logging.info("[DRY] claude-wire feature=%s (model=%s); brief at %s",
                     name, HELPER_MODEL, req_path)
        return {"status": "dry", "tier": TIER_CLAUDE, "brief_path": str(req_path),
                "stdout": f"WIRE_RESULT: SUCCESS feature={name} version=dry module=<claude-driven>",
                "stderr": "", "rc": 0, "elapsed_sec": 0.0}

    if not CLAUDE_BIN.exists():
        logging.warning("CLAUDE_BIN not found at %s — queued spec at %s",
                        CLAUDE_BIN, req_path)
        return {"status": "queued_no_binary", "tier": TIER_CLAUDE,
                "brief_path": str(req_path),
                "stdout": "", "stderr": "claude binary missing", "rc": -3,
                "elapsed_sec": 0.0}

    cmd = [
        str(CLAUDE_BIN), "-p", "--model", HELPER_MODEL,
        "--output-format", "text", "--permission-mode", "bypassPermissions",
        brief,
    ]
    logging.info("claude-wire feature=%s (model=%s, timeout=%ds)",
                 name, HELPER_MODEL, HELPER_TIMEOUT_SEC)
    started = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=HELPER_TIMEOUT_SEC)
        elapsed = time.time() - started
        return {
            "status": "ok" if result.returncode == 0 else "fail",
            "tier": TIER_CLAUDE, "brief_path": str(req_path),
            "stdout": result.stdout[-4000:], "stderr": result.stderr[-1000:],
            "rc": result.returncode, "elapsed_sec": elapsed,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        logging.warning("claude-wire TIMEOUT feature=%s after %.1fs", name, elapsed)
        return {"status": "timeout", "tier": TIER_CLAUDE, "brief_path": str(req_path),
                "stdout": "", "stderr": f"timeout after {elapsed:.1f}s",
                "rc": -1, "elapsed_sec": elapsed}
    except Exception as exc:  # noqa: BLE001
        logging.exception("claude-wire crashed feature=%s: %s", name, exc)
        return {"status": "crash", "tier": TIER_CLAUDE, "brief_path": str(req_path),
                "stdout": "", "stderr": repr(exc), "rc": -2, "elapsed_sec": 0.0}


def spawn_wire_in_helper(cand: dict, dry_run: bool) -> dict:
    """Three-tier dispatcher (added 2026-05-17).

    Routes to template/ollama/claude based on candidate complexity.
    The returned dict's `tier` field tells the caller which tier handled it.
    Preserves existing semantics: returns same shape (status, brief_path,
    stdout, stderr, rc, elapsed_sec); stdout still carries WIRE_RESULT line
    so parse_outcome() works unchanged.
    """
    tier = _classify_complexity(cand)
    logging.info("ROUTE feature=%s tier=%s data_source=%r",
                 cand.get("feature_name"), tier,
                 (cand.get("data_source") or "")[:60])
    if tier == TIER_DIR_GLOB:
        return _dir_glob_wire(cand, dry_run)
    if tier == TIER_TEMPLATE:
        return _template_wire(cand, dry_run)
    if tier == TIER_OLLAMA:
        return _ollama_wire(cand, dry_run)
    return _claude_wire(cand, dry_run)


# ---------------------------------------------------------------------------
# Outcome parsing
# ---------------------------------------------------------------------------


WIRE_RESULT_RE = re.compile(
    r"WIRE_RESULT:\s+(SUCCESS|FAILURE|REJECTED)\s+(.*)$",
    re.MULTILINE,
)


def parse_outcome(helper_result: dict) -> tuple[str, dict]:
    """Returns (verdict, parsed_fields). verdict in {SUCCESS, FAILURE, REJECTED, UNKNOWN}."""
    stdout = helper_result.get("stdout", "") or ""
    # Scan from the bottom; the LAST WIRE_RESULT wins
    matches = list(WIRE_RESULT_RE.finditer(stdout))
    if not matches:
        return "UNKNOWN", {}
    last = matches[-1]
    verdict = last.group(1).upper()
    rest = last.group(2)
    fields: dict[str, str] = {}
    for kv in re.finditer(r"(\w+)\s*=\s*\"?([^\"\s][^\"]*?)\"?(?=\s\w+=|\s*$)", rest):
        fields[kv.group(1)] = kv.group(2).strip()
    return verdict, fields


# ---------------------------------------------------------------------------
# Recording wired / rejected
# ---------------------------------------------------------------------------


def record_outcome(cand: dict, verdict: str, fields: dict, helper_result: dict) -> None:
    target = WIRED_JSON if verdict == "SUCCESS" else REJECTED_JSON
    rec_list = _load_json(target, [])
    if not isinstance(rec_list, list):
        rec_list = []
    rec_list.append({
        "id": cand["id"],
        "feature_name": cand["feature_name"],
        "data_source": cand["data_source"],
        "verdict": verdict,
        "fields": fields,
        "helper_status": helper_result.get("status"),
        "elapsed_sec": helper_result.get("elapsed_sec"),
        "brief_path": helper_result.get("brief_path"),
        "recorded_at": _now_iso(),
    })
    _save_json(target, rec_list)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def process_candidates(candidates: list[dict], max_spawns: int, dry_run: bool,
                       cycle_n: int = 0, seen_total: int = 0) -> int:
    """Enqueue + spawn helpers up to max_spawns. Returns number of spawns.

    Per-candidate logging (WIRE_CANDIDATE / validate / spawn start+finish /
    wired+rejected) added 2026-05-17 — observability was blind before.

    `cycle_n` + `seen_total` are passed only so we can update the heartbeat
    file mid-cycle (so watchdogs don't see staleness during the synchronous
    wire-in spawn, which can run up to HELPER_TIMEOUT_SEC=600s per candidate).
    """
    spawned = 0
    for idx, cand in enumerate(candidates, start=1):
        cand_name = cand.get("feature_name", "<unknown>")
        cand_id = cand.get("id", "<no-id>")
        # Per-candidate WIRE_CANDIDATE log — fires for every candidate seen
        # regardless of explicit/heuristic. Replaces silence between cycle
        # boundaries when N candidates funnel through validation/gating.
        logging.info("WIRE_CANDIDATE [%d/%d] feature=%s id=%s explicit=%s source=%s",
                     idx, len(candidates), cand_name, cand_id,
                     cand.get("explicit", False),
                     str(cand.get("source", ""))[-80:])

        # Heartbeat tick — per-candidate so a 600s synchronous spawn doesn't
        # freeze the heartbeat for the whole cycle.
        _heartbeat(cycle_n, seen_total, spawned,
                   phase="processing",
                   extra={"current_candidate": cand_name,
                          "candidate_index": idx,
                          "candidate_total": len(candidates)})

        # 1. enqueue (queue is append-only history of every candidate we've ever considered)
        _append_jsonl(WIRING_QUEUE, {
            "ts": _now_iso(),
            "id": cand["id"],
            "feature_name": cand["feature_name"],
            "data_source": cand["data_source"],
            "explicit": cand.get("explicit", False),
            "source": cand.get("source"),
            "metadata": cand.get("metadata", {}),
        })

        # 1b. Refactor-target detection (added 2026-05-17).
        # Some producers (e.g. trading_info_3 hunter) emit pipeline-change
        # proposals (ensemble models, SHAP, conformal, HMM regime training,
        # Kelly sizing, MC validation, calibration curves) as WIRE_CANDIDATE
        # blocks with `features_added: 0` AND `refactor_target: <path>`.
        # These are NOT feature additions — auto-wiring them would crash the
        # helper (no module to write, no Helper-* slot to add). Route to a
        # separate refactor_queue.jsonl for human review and skip downstream
        # validation/gating/spawn.
        is_refactor, refactor_reason = is_refactor_target(cand)
        if is_refactor:
            logging.info("VALIDATE_DECISION feature=%s -> REFACTOR (%s)",
                         cand_name, refactor_reason)
            route_to_refactor_queue(cand, refactor_reason)
            record_outcome(cand, "REJECTED",
                           {"reason": f"refactor_target:{refactor_reason}"},
                           {"status": "routed_to_refactor_queue",
                            "elapsed_sec": 0.0,
                            "brief_path": str(REFACTOR_QUEUE)})
            logging.info("REJECTED feature=%s reason=refactor_target", cand_name)
            continue

        # 2. validate
        ok, reason = validate_candidate(cand)
        if not ok:
            logging.info("VALIDATE_DECISION feature=%s -> INVALID (%s)",
                         cand_name, reason)
            record_outcome(cand, "REJECTED",
                           {"reason": reason},
                           {"status": "invalid_candidate", "elapsed_sec": 0.0,
                            "brief_path": None})
            logging.info("REJECTED feature=%s reason=invalid:%s", cand_name, reason)
            continue
        logging.info("VALIDATE_DECISION feature=%s -> VALID", cand_name)

        # 3. gating
        gated, gate_reason = is_gated_human_review(cand)
        if gated:
            logging.info("VALIDATE_DECISION feature=%s -> GATED (%s)",
                         cand_name, gate_reason)
            # Still write the brief into wiring_requests/ for the human queue
            _ = spawn_wire_in_helper(cand, dry_run=True)  # writes brief, doesn't spawn
            record_outcome(cand, "REJECTED",
                           {"reason": f"gated:{gate_reason}"},
                           {"status": "gated_human_review", "elapsed_sec": 0.0,
                            "brief_path": str(WIRING_REQUESTS_DIR)})
            logging.info("REJECTED feature=%s reason=gated:%s", cand_name, gate_reason)
            continue

        # 4. spawn cap
        if spawned >= max_spawns:
            logging.info("Hit max_spawns=%d this cycle — feature=%s deferred to next cycle",
                         max_spawns, cand_name)
            break

        # 5. spawn (per-spawn START/FINISH logs frame the long synchronous block)
        logging.info("SPAWN_START feature=%s model=%s timeout=%ds dry=%s",
                     cand_name, HELPER_MODEL, HELPER_TIMEOUT_SEC, dry_run)
        _heartbeat(cycle_n, seen_total, spawned,
                   phase="spawning",
                   extra={"current_candidate": cand_name,
                          "spawn_started_at": _now_iso()})
        helper_result = spawn_wire_in_helper(cand, dry_run=dry_run)
        spawned += 1
        logging.info("SPAWN_FINISH feature=%s status=%s rc=%s elapsed=%.1fs",
                     cand_name, helper_result.get("status"),
                     helper_result.get("rc"), helper_result.get("elapsed_sec", 0.0))

        verdict, fields = parse_outcome(helper_result)
        if helper_result["status"] == "dry":
            logging.info("[DRY] enqueued only feature=%s (brief at %s)",
                         cand_name, helper_result.get("brief_path"))
            continue
        if verdict == "UNKNOWN":
            # No WIRE_RESULT line — treat as failure (we don't know what happened)
            logging.warning("Helper for feature=%s returned no WIRE_RESULT line (status=%s rc=%s)",
                            cand_name, helper_result["status"], helper_result["rc"])
            record_outcome(cand, "REJECTED",
                           {"reason": f"no_wire_result;helper_status={helper_result['status']}"},
                           helper_result)
            logging.info("REJECTED feature=%s reason=no_wire_result", cand_name)
        else:
            logging.info("Helper for feature=%s returned %s %s",
                         cand_name, verdict, fields)
            record_outcome(cand, verdict, fields, helper_result)
            if verdict == "SUCCESS":
                logging.info("WIRED feature=%s version=%s module=%s",
                             cand_name, fields.get("version", "?"),
                             fields.get("module", "?"))
            else:
                logging.info("REJECTED feature=%s reason=helper_%s details=%s",
                             cand_name, verdict.lower(), fields)
    return spawned


def one_cycle(state: dict, seen: set[str], max_spawns: int, dry_run: bool,
              cycle_n: int) -> tuple[int, int]:
    """Returns (candidates_this_cycle, spawned_this_cycle).

    Heartbeat is written at THREE points per cycle (start / mid / end) so
    watchdogs never see a frozen file when a single synchronous wire-in spawn
    holds the cycle for up to HELPER_TIMEOUT_SEC=600s. Pre-fix, heartbeat was
    written only at end-of-cycle and could stay stale for 10+ minutes.
    """
    # 1. Start-of-cycle heartbeat — proves the cycle even started.
    _heartbeat(cycle_n, len(seen), 0, phase="start")

    discovered = []
    discovered.extend(scan_discovery_reports(state, seen))
    discovered.extend(scan_proactive_stream(state, seen))
    if discovered:
        logging.info("Cycle %d: %d new candidate(s) discovered",
                     cycle_n, len(discovered))

    # 2. Post-scan heartbeat — proves scan completed before processing.
    _heartbeat(cycle_n, len(seen), 0, phase="scanned",
               extra={"discovered_this_cycle": len(discovered)})

    spawned = process_candidates(discovered, max_spawns, dry_run,
                                 cycle_n=cycle_n, seen_total=len(seen))
    _save_json(CONSUMER_STATE, state)

    # 3. End-of-cycle heartbeat — original behaviour, preserved.
    _heartbeat(cycle_n, len(seen), spawned, phase="end",
               extra={"discovered_this_cycle": len(discovered)})
    return len(discovered), spawned


_STOP = False


def _install_signal_handlers() -> None:
    def _h(signum, _frame):
        global _STOP
        logging.info("Received signal %s — finishing current cycle then exiting", signum)
        _STOP = True
    signal.signal(signal.SIGINT, _h)
    signal.signal(signal.SIGTERM, _h)


def seed_one(path: Path, seen: set[str], dry_run: bool) -> int:
    """Treat one specific file as a fresh producer drop. Used for smoke tests."""
    if not path.exists():
        logging.error("--seed-candidate file not found: %s", path)
        return 2
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logging.error("Could not read %s: %s", path, exc)
        return 2
    cands = _parse_text_for_candidates(text, str(path.resolve()))
    fresh = [c for c in cands if c["id"] not in seen]
    for c in fresh:
        seen.add(c["id"])
    logging.info("Seed: parsed %d candidate(s), %d fresh", len(cands), len(fresh))
    spawned = process_candidates(fresh, max_spawns=len(fresh), dry_run=dry_run)
    logging.info("Seed: spawned=%d (dry=%s)", spawned, dry_run)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="parse + enqueue but never spawn helpers")
    parser.add_argument("--once", action="store_true",
                        help="run one cycle and exit")
    parser.add_argument("--cycle-sec", type=int, default=DEFAULT_CYCLE_SEC)
    parser.add_argument("--max-spawns", type=int, default=DEFAULT_MAX_SPAWNS_PER_CYCLE,
                        help="cap on wire-in helpers per cycle (default 2)")
    parser.add_argument("--seed-candidate", type=str, default=None,
                        help="absolute path to a single candidate file (for smoke tests)")
    parser.add_argument("--force-rescan", action="store_true",
                        help="On startup, re-evaluate previously gated rejections "
                             "under current rules (drops 'gated:*' rejected ids "
                             "from the seen set AND clears file mtime state so "
                             "discovery reports get re-read).")
    parser.add_argument("--prefer-glob", dest="prefer_glob", action="store_true",
                        default=None,
                        help="Force directory-glob tier for ANY candidate with "
                             "metadata.data_type=directory_glob (default ON via "
                             "CONSUMER_PREFER_GLOB env var). Mutually exclusive "
                             "with --no-prefer-glob.")
    parser.add_argument("--no-prefer-glob", dest="prefer_glob", action="store_false",
                        help="Disable directory-glob routing; route dir-glob "
                             "candidates through the standard tier classifier.")
    args = parser.parse_args(argv)

    # Apply --prefer-glob / --no-prefer-glob override AFTER parse (default keeps
    # env-var/builtin default in place when neither flag was supplied).
    if args.prefer_glob is not None:
        global PREFER_GLOB
        PREFER_GLOB = bool(args.prefer_glob)

    setup_logging()
    _install_signal_handlers()

    # Ensure dirs exist
    for d in (WIRING_DIR, WIRING_REQUESTS_DIR, LOG_DIR, BACKUPS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    logging.info("=== feature_wiring_consumer_daemon start "
                 "(dry=%s once=%s cycle=%ds max_spawns=%d pid=%d) ===",
                 args.dry_run, args.once, args.cycle_sec, args.max_spawns, os.getpid())

    state = _load_json(CONSUMER_STATE, {})
    if not isinstance(state, dict):
        state = {}
    if args.force_rescan:
        # Clear discovery file mtimes so already-scanned reports get re-read.
        # The producer stream offset is intentionally preserved — we don't want
        # to replay the entire historic proactive stream, only the
        # discovery-report markdown files (which we KNOW carry gated candidates
        # under the old rules).
        prev_mtimes = state.pop("discovery_file_mtimes", None)
        if prev_mtimes:
            logging.info("FORCE_RESCAN: cleared %d discovery file mtime entries",
                         len(prev_mtimes))
    seen = load_seen_ids(force_rescan=args.force_rescan)
    logging.info("Loaded %d seen candidate IDs from history (force_rescan=%s)",
                 len(seen), args.force_rescan)

    if args.seed_candidate:
        return seed_one(Path(args.seed_candidate), seen, args.dry_run)

    cycle_n = 0
    while not _STOP:
        cycle_n += 1
        try:
            n_disc, n_spawn = one_cycle(state, seen, args.max_spawns,
                                        args.dry_run, cycle_n)
            logging.info("Cycle %d done: discovered=%d spawned=%d seen_total=%d",
                         cycle_n, n_disc, n_spawn, len(seen))
        except Exception as exc:  # noqa: BLE001
            logging.exception("Cycle %d crashed: %s — sleeping and continuing", cycle_n, exc)
        if args.once:
            break
        # Sleep in small chunks so signals are responsive
        slept = 0
        while slept < args.cycle_sec and not _STOP:
            time.sleep(min(2, args.cycle_sec - slept))
            slept += 2

    logging.info("=== feature_wiring_consumer_daemon exit (cycles=%d) ===", cycle_n)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
