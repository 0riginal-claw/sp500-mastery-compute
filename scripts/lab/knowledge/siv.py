"""lab.knowledge.siv — Statistical Inference Validation gate loader.

Reads the pre-flight gate file at:
  AI-Tools/state/lab_siv/SIV_<DATE>.signed.json

Used by `championship_search` (and other promotion paths) to refuse promoting a
SAP to paper trading unless the full SIV checklist is PRESENT.

Public API:
    siv_status() -> dict
        Return the current SIV record (loads from Drive, falls back to local
        ZG-2TB tier). Raises FileNotFoundError if neither exists.

    is_promotion_allowed() -> tuple[bool, str]
        Convenience: returns (allowed, reason). `allowed=False` if overall
        status is not 'PRESENT' or file is missing.

    enforce_promotion_gate(soft: bool = True) -> None
        Soft (default): print a stderr warning if gate is closed.
        Hard: raise PermissionError if gate is closed.

Design notes:
  - The SIV file is locked by the user (not auto-rebuilt). A re-lock requires
    running scripts/siv_verifier.py again.
  - The loader caches the parsed record at module import + 60s after.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

DRIVE_BASE = Path(
    os.environ.get(
        "DRIVE_BASE",
        "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive",
    )
)

# Default SIV file (date-stamped). Look at DRIVE first, ZG-2TB fallback second.
_DEFAULT_SIV_NAME = "SIV_2026-05-29.signed.json"
_SIV_DRIVE = DRIVE_BASE / "AI-Tools" / "state" / "lab_siv" / _DEFAULT_SIV_NAME
_SIV_FALLBACK = Path("/Volumes/ZG-2TB/zg/state/lab_siv") / _DEFAULT_SIV_NAME

_CACHE: dict[str, Any] = {"record": None, "loaded_at": 0.0, "source": None}
_TTL_SEC = 60.0


def _siv_paths() -> list[Path]:
    """Resolution order: env override -> Drive -> ZG-2TB."""
    override = os.environ.get("SIV_PATH")
    paths: list[Path] = []
    if override:
        paths.append(Path(override))
    paths.extend([_SIV_DRIVE, _SIV_FALLBACK])
    return paths


def siv_status(force_reload: bool = False) -> dict:
    """Load + return the SIV record. Raises FileNotFoundError if missing."""
    now = time.time()
    if (not force_reload
            and _CACHE["record"] is not None
            and (now - _CACHE["loaded_at"]) < _TTL_SEC):
        return _CACHE["record"]

    last_err: Exception | None = None
    for p in _siv_paths():
        try:
            if p.exists():
                rec = json.loads(p.read_text())
                _CACHE["record"] = rec
                _CACHE["loaded_at"] = now
                _CACHE["source"] = str(p)
                return rec
        except (OSError, json.JSONDecodeError) as e:
            last_err = e
            continue

    raise FileNotFoundError(
        f"SIV file not found in any tier. Searched: "
        f"{[str(p) for p in _siv_paths()]}. Last error: {last_err}"
    )


def is_promotion_allowed() -> tuple[bool, str]:
    """Return (allowed, reason). Allowed iff overall_status == 'PRESENT'."""
    try:
        rec = siv_status()
    except FileNotFoundError as e:
        return False, f"SIV file missing: {e}"
    overall = rec.get("overall_status")
    if overall == "PRESENT":
        return True, "SIV checklist PRESENT — promotion gate OPEN."
    next_actions = rec.get("next_action_to_complete", [])
    reason = (
        f"SIV overall_status={overall!r}; cannot promote. "
        f"Next actions: {next_actions}"
    )
    return False, reason


def enforce_promotion_gate(soft: bool = True) -> None:
    """Check the gate. Soft = stderr warning. Hard = raise PermissionError."""
    allowed, reason = is_promotion_allowed()
    if allowed:
        return
    msg = f"[SIV-GATE] {reason}"
    if soft:
        print(msg, file=sys.stderr)
        return
    raise PermissionError(msg)


def source() -> str | None:
    """Return the path the cached record was loaded from (or None)."""
    return _CACHE["source"]


__all__ = [
    "siv_status",
    "is_promotion_allowed",
    "enforce_promotion_gate",
    "source",
]
