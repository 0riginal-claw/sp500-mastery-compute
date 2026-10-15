#!/usr/bin/env python3
"""
imessage_otp_listener.py
========================

Read-only iMessage / SMS OTP listener for macOS.

Polls ~/Library/Messages/chat.db (SQLite) every N seconds, filters incoming
SMS messages by sender and OTP-shaped body, and writes captured OTPs to
~/.config/auto_signup/otp_queue/<provider>_<UTC>.json (chmod 600).

Requirements
------------
- macOS with Messages.app
- iPhone -> Settings -> Messages -> Text Message Forwarding -> enable this Mac
  (so SMS appears in chat.db, not just iMessage)
- Full Disk Access granted to the python interpreter that runs this script
  (System Settings -> Privacy & Security -> Full Disk Access)
- stdlib only (sqlite3, json, re, argparse, time, datetime, pathlib, os, sys)
- Optional: PyObjC for attributedBody decoding (graceful fallback if missing)

Safety contract
---------------
- READ-ONLY access to chat.db; no INSERT/UPDATE/DELETE
- chat.db opened with mode=ro URI; immutable=1 to avoid lock contention
- Never writes inside ~/Library/Messages/
- Never logs full SMS body; only sender + UTC timestamp + extracted OTP
- License: stdlib + optional PyObjC (Apache 2.0). No third-party network calls.

CLI
---
  python imessage_otp_listener.py --help
  python imessage_otp_listener.py --dry-run --provider kaggle
  python imessage_otp_listener.py --since "5 min ago" --sender kaggle --timeout 300
  python imessage_otp_listener.py --sender "+1*KAGGLE*" --json --timeout 600

Exit codes
----------
  0  - OTP captured (or dry-run completed)
  1  - Full Disk Access denied (prints setup instructions)
  2  - Timeout reached without OTP
  3  - Invalid arguments / other error
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# Apple Core Data epoch: 2001-01-01 UTC. Mac stores nanoseconds since this.
MAC_EPOCH_UTC = datetime(2001, 1, 1, tzinfo=timezone.utc)
MAC_EPOCH_OFFSET_S = 978307200  # seconds between 1970-01-01 and 2001-01-01

# Real home (NOT the launcher-remapped $HOME inside AI-Tools).
# We resolve via getpwuid to dodge any $HOME monkey-patching.
import pwd  # noqa: E402

REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir)
CHAT_DB = REAL_HOME / "Library" / "Messages" / "chat.db"

# OTP queue location. We DON'T put this inside Messages/.
OTP_QUEUE_DIR = REAL_HOME / ".config" / "auto_signup" / "otp_queue"

# OTP regex: 4-8 digit code, often with separator. We pull standalone digit groups.
OTP_REGEX = re.compile(r"(?<!\d)(\d{4,8})(?!\d)")

# Keywords that hint a body is actually an OTP/verification message.
OTP_KEYWORDS = re.compile(
    r"\b(code|verify|verification|otp|confirm|passcode|pin|security|one[- ]?time|2fa|auth)\b",
    re.IGNORECASE,
)

POLL_INTERVAL_DEFAULT_S = 1.5


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------


@dataclass
class CapturedOTP:
    timestamp_utc: str
    sender: str
    otp: str
    provider: str
    rowid: int
    service: str  # "SMS" or "iMessage"
    # NOTE: raw_body intentionally omitted from the on-disk queue payload by default.
    # Set --include-body to include it (sensitive).
    raw_body: Optional[str] = None


# ----------------------------------------------------------------------
# Time helpers
# ----------------------------------------------------------------------


def mac_ns_to_utc(mac_ns: int) -> datetime:
    """
    Convert message.date (nanoseconds since 2001-01-01 UTC) to a UTC datetime.

    Defensive: some old rows store seconds, not nanoseconds. We detect by
    magnitude: a sane ns value is > 10**16 for 2010+; a seconds value is < 10**11.
    """
    if mac_ns == 0:
        return MAC_EPOCH_UTC
    if mac_ns > 10**14:
        seconds = mac_ns / 1_000_000_000
    else:
        seconds = float(mac_ns)
    return MAC_EPOCH_UTC + timedelta(seconds=seconds)


def utc_to_mac_ns(when: datetime) -> int:
    """Inverse of mac_ns_to_utc, returning nanoseconds since 2001-01-01."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    delta = when - MAC_EPOCH_UTC
    return int(delta.total_seconds() * 1_000_000_000)


def parse_since(spec: str) -> datetime:
    """
    Parse a --since spec into a UTC datetime.

    Accepts:
      - 'now'              -> current UTC time
      - 'N min ago'        -> N minutes ago
      - 'N sec ago'        -> N seconds ago
      - 'N hour ago'       -> N hours ago
      - ISO-8601 string    -> parsed as UTC (Z) or with offset
    """
    spec = spec.strip().lower()
    now = datetime.now(timezone.utc)
    if spec in ("now", "0", ""):
        return now
    m = re.match(r"^(\d+)\s*(sec|min|hour|hr|h|m|s)(?:s)?\s*ago$", spec)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit in ("sec", "s"):
            return now - timedelta(seconds=n)
        if unit in ("min", "m"):
            return now - timedelta(minutes=n)
        if unit in ("hour", "hr", "h"):
            return now - timedelta(hours=n)
    # Try ISO-8601.
    try:
        # Allow trailing 'Z'.
        iso = spec.replace("z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        raise ValueError(f"could not parse --since: {spec!r}")


# ----------------------------------------------------------------------
# attributedBody decoder (PyObjC preferred, manual fallback)
# ----------------------------------------------------------------------


def _decode_attributed_body_pyobjc(blob: bytes) -> Optional[str]:
    """Decode attributedBody via NSKeyedUnarchiver. Returns None on failure."""
    try:
        from Foundation import NSData, NSKeyedUnarchiver  # type: ignore
    except Exception:
        return None
    try:
        ns = NSData.dataWithBytes_length_(blob, len(blob))
        unarchiver = NSKeyedUnarchiver.alloc().initForReadingWithData_(ns)
        unarchiver.setRequiresSecureCoding_(False)
        obj = unarchiver.decodeObjectForKey_("root")
        if obj is not None and hasattr(obj, "string"):
            return str(obj.string())
    except Exception:
        return None
    return None


def _decode_attributed_body_manual(blob: bytes) -> Optional[str]:
    """
    Manual byte-pattern fallback (LangChain-style).

    Looks for the b'NSString' marker and reads the following length-prefixed
    UTF-8 body. Returns None if the marker isn't present.
    """
    try:
        parts = blob.split(b"NSString")
        if len(parts) < 2:
            return None
        content = parts[1][5:]
        if not content:
            return None
        first = content[0]
        if first == 0x81:  # 129 -> 2-byte little-endian length
            length = int.from_bytes(content[1:3], "little")
            start = 3
        else:
            length = first
            start = 1
        return content[start : start + length].decode("utf-8", errors="ignore")
    except Exception:
        return None


def decode_attributed_body(blob: Optional[bytes]) -> Optional[str]:
    if not blob:
        return None
    decoded = _decode_attributed_body_pyobjc(blob)
    if decoded:
        return decoded
    return _decode_attributed_body_manual(blob)


# ----------------------------------------------------------------------
# Sender matching
# ----------------------------------------------------------------------


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """
    Convert a simple glob (* and ?) to a case-insensitive regex.

    Plain substrings (no glob chars) are treated as case-insensitive substring
    matches. This makes --sender 'kaggle' match handle.id 'KAGGLE' or '5-KAGG'.
    """
    has_glob = "*" in pattern or "?" in pattern
    if not has_glob:
        return re.compile(re.escape(pattern), re.IGNORECASE)
    parts: list[str] = []
    for ch in pattern:
        if ch == "*":
            parts.append(".*")
        elif ch == "?":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    return re.compile("^" + "".join(parts) + "$", re.IGNORECASE)


def normalize_phone(handle: str) -> str:
    """Strip whitespace, return canonical form (preserves +country if present)."""
    return re.sub(r"\s+", "", handle)


def sender_matches(handle_id: str, patterns: list[re.Pattern[str]]) -> bool:
    if not patterns:
        return True
    h = normalize_phone(handle_id)
    return any(p.search(h) for p in patterns)


# ----------------------------------------------------------------------
# OTP extraction
# ----------------------------------------------------------------------


def extract_otp(body: str) -> Optional[str]:
    """Return the most likely OTP from a body, or None if no OTP-shaped digits."""
    if not body:
        return None
    has_keyword = bool(OTP_KEYWORDS.search(body))
    candidates = OTP_REGEX.findall(body)
    if not candidates:
        return None
    if not has_keyword:
        # Be permissive: a body that's just '123456' is still a valid OTP signal.
        if len(body.strip()) <= 12 and len(candidates) == 1 and 4 <= len(candidates[0]) <= 8:
            return candidates[0]
        return None
    # Prefer the longest digit run within OTP length bounds, ties -> earliest.
    candidates.sort(key=lambda c: (-len(c), body.index(c)))
    for c in candidates:
        if 4 <= len(c) <= 8:
            return c
    return None


def guess_provider(sender: str, body: str, override: Optional[str]) -> str:
    if override:
        return override.lower()
    # Pull a likely provider keyword from sender or body.
    src = f"{sender} {body}".lower()
    for word in (
        "kaggle",
        "google",
        "github",
        "gitlab",
        "huggingface",
        "vercel",
        "lightning",
        "openai",
        "anthropic",
        "aws",
        "azure",
        "modal",
        "stripe",
        "paypal",
        "apple",
    ):
        if word in src:
            return word
    return "unknown"


# ----------------------------------------------------------------------
# DB access (read-only)
# ----------------------------------------------------------------------


class FullDiskAccessDenied(RuntimeError):
    pass


def open_chat_db(path: Path = CHAT_DB) -> sqlite3.Connection:
    """
    Open chat.db read-only. Raises FullDiskAccessDenied if macOS blocks us.

    Using immutable=1 lets us read even while Messages.app holds a lock.
    """
    if not path.exists():
        raise FullDiskAccessDenied(
            f"chat.db not found at {path}. Either Messages.app has never run "
            f"or Full Disk Access is denied (the file appears absent from this "
            f"process's view)."
        )
    uri = f"file:{path}?mode=ro&immutable=1"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        # Quick canary: a SELECT that will hit FDA if anything will.
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        return conn
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "authoriz" in msg or "permission" in msg or "unable to open" in msg:
            raise FullDiskAccessDenied(str(e)) from e
        raise


QUERY_NEW_MESSAGES = """
SELECT
    m.ROWID         AS rowid,
    m.date          AS mac_ns,
    m.text          AS text,
    m.attributedBody AS attributed_body,
    m.is_from_me    AS is_from_me,
    m.service       AS service,
    h.id            AS handle_id
FROM message m
LEFT JOIN handle h ON m.handle_id = h.ROWID
WHERE m.is_from_me = 0
  AND m.date > ?
  AND m.ROWID > ?
ORDER BY m.ROWID ASC
LIMIT 200
"""


def fetch_new_messages(
    conn: sqlite3.Connection, since_mac_ns: int, last_rowid: int
) -> list[dict]:
    rows = conn.execute(QUERY_NEW_MESSAGES, (since_mac_ns, last_rowid)).fetchall()
    out = []
    for r in rows:
        rowid, mac_ns, text, attr_body, is_from_me, service, handle_id = r
        body = text or decode_attributed_body(attr_body)
        if not body:
            continue
        out.append(
            {
                "rowid": rowid,
                "mac_ns": mac_ns,
                "body": body,
                "is_from_me": is_from_me,
                "service": service or "Unknown",
                "handle_id": handle_id or "",
            }
        )
    return out


# ----------------------------------------------------------------------
# OTP queue writer
# ----------------------------------------------------------------------


def write_otp(captured: CapturedOTP, queue_dir: Path = OTP_QUEUE_DIR) -> Path:
    queue_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(queue_dir, 0o700)
    except OSError:
        pass
    stamp = captured.timestamp_utc.replace(":", "").replace("-", "").replace(".", "")
    fname = f"{captured.provider}_{stamp}_{captured.rowid}.json"
    path = queue_dir / fname
    payload = asdict(captured)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)
    return path


# ----------------------------------------------------------------------
# Core loop
# ----------------------------------------------------------------------


def listen(
    *,
    since: datetime,
    sender_patterns: list[re.Pattern[str]],
    provider_override: Optional[str],
    timeout_s: float,
    poll_interval_s: float,
    include_body: bool,
    emit_json: bool,
    stop_after_first: bool = True,
) -> int:
    """
    Returns:
       0 on capture, 2 on timeout, 1 on FDA denial.
    """
    try:
        conn = open_chat_db()
    except FullDiskAccessDenied as e:
        print_fda_instructions(str(e))
        return 1

    since_mac_ns = utc_to_mac_ns(since)
    last_rowid = 0
    deadline = time.monotonic() + timeout_s

    if not emit_json:
        print(
            f"[listener] watching chat.db since {since.isoformat()} "
            f"(timeout={timeout_s:.0f}s, poll={poll_interval_s:.1f}s, "
            f"senders={len(sender_patterns) or 'any'})",
            file=sys.stderr,
        )

    while True:
        rows = fetch_new_messages(conn, since_mac_ns, last_rowid)
        for row in rows:
            last_rowid = max(last_rowid, row["rowid"])
            handle = row["handle_id"]
            body = row["body"]
            if not sender_matches(handle, sender_patterns):
                continue
            otp = extract_otp(body)
            if not otp:
                continue
            cap = CapturedOTP(
                timestamp_utc=mac_ns_to_utc(row["mac_ns"]).isoformat().replace("+00:00", "Z"),
                sender=handle,
                otp=otp,
                provider=guess_provider(handle, body, provider_override),
                rowid=row["rowid"],
                service=row["service"],
                raw_body=body if include_body else None,
            )
            path = write_otp(cap)
            payload = {
                "status": "captured",
                "otp": cap.otp,
                "provider": cap.provider,
                "sender": cap.sender,
                "timestamp_utc": cap.timestamp_utc,
                "rowid": cap.rowid,
                "service": cap.service,
                "queue_path": str(path),
            }
            if emit_json:
                print(json.dumps(payload))
            else:
                print(
                    f"[listener] captured OTP={cap.otp} provider={cap.provider} "
                    f"sender={cap.sender} service={cap.service} -> {path}",
                    file=sys.stderr,
                )
            if stop_after_first:
                return 0
        if time.monotonic() > deadline:
            if emit_json:
                print(json.dumps({"status": "timeout"}))
            else:
                print("[listener] timeout reached without OTP", file=sys.stderr)
            return 2
        time.sleep(poll_interval_s)


# ----------------------------------------------------------------------
# Dry-run synthesis
# ----------------------------------------------------------------------


def dry_run(provider: str, emit_json: bool, include_body: bool) -> int:
    """Synthesize a fake captured OTP and write to the queue. No DB read."""
    fake_body = f"Your {provider.capitalize()} verification code is 482917. Do not share."
    otp = extract_otp(fake_body) or "482917"
    cap = CapturedOTP(
        timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        sender=f"+1*{provider.upper()}*",
        otp=otp,
        provider=provider.lower(),
        rowid=-1,
        service="SMS",
        raw_body=fake_body if include_body else None,
    )
    path = write_otp(cap)
    payload = {
        "status": "dry_run_captured",
        "otp": cap.otp,
        "provider": cap.provider,
        "sender": cap.sender,
        "timestamp_utc": cap.timestamp_utc,
        "queue_path": str(path),
    }
    if emit_json:
        print(json.dumps(payload))
    else:
        print(
            f"[listener][dry-run] synthesized OTP={cap.otp} provider={cap.provider} "
            f"-> {path}",
            file=sys.stderr,
        )
    return 0


# ----------------------------------------------------------------------
# FDA setup instructions
# ----------------------------------------------------------------------


def print_fda_instructions(detail: str) -> None:
    python_path = Path(sys.executable).resolve()
    msg = f"""
============================================================
 FULL DISK ACCESS REQUIRED
============================================================
The current Python interpreter cannot read ~/Library/Messages/chat.db.

Detail: {detail}

To fix on macOS:

1. Open: System Settings -> Privacy & Security -> Full Disk Access
2. Click the (+) button to add a binary.
3. Press Cmd+Shift+G and paste this exact path:

       {python_path}

   Select the binary and click Open.
4. Toggle the new entry to ON.
5. Restart Terminal (or whatever process runs this script) and try again.

Notes:
- Granting FDA to a venv python (e.g. .venvs/sp500-mastery/bin/python) ONLY
  grants access for THAT interpreter. If you swap interpreters, re-add.
- Granting FDA to the symlink is fine; macOS resolves the real binary.
- Also make sure on iPhone: Settings -> Messages -> Text Message Forwarding
  -> enable this Mac, so SMS appears in chat.db (not just iMessage).

After granting, verify with:
   sqlite3 ~/Library/Messages/chat.db "SELECT count(*) FROM message;"
============================================================
""".strip()
    print(msg, file=sys.stderr)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="imessage_otp_listener.py",
        description="Read-only macOS iMessage/SMS OTP listener.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--since",
        default="now",
        help="Start watching from this point in time. Examples: 'now', "
        "'5 min ago', '30 sec ago', '2026-05-17T18:00:00Z'. Default: now.",
    )
    p.add_argument(
        "--sender",
        action="append",
        default=[],
        help="Sender match pattern (case-insensitive substring or glob with "
        "* and ?). Repeatable. Examples: 'kaggle', '+1*KAGGLE*', "
        "'+14155551234'. If omitted, matches any sender.",
    )
    p.add_argument(
        "--provider",
        default=None,
        help="Override provider name for the queue filename (kaggle, github, "
        "openai, etc.). If omitted, inferred from sender/body keywords.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Max seconds to wait for an OTP. Default: 300.",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL_DEFAULT_S,
        help=f"Seconds between DB polls. Default: {POLL_INTERVAL_DEFAULT_S}.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Don't stop after the first OTP; keep listening until timeout.",
    )
    p.add_argument(
        "--include-body",
        action="store_true",
        help="Include full SMS body in the JSON output / queue payload. "
        "DEFAULT: omitted (only sender + UTC + OTP digits are persisted). "
        "Bodies are sensitive (may contain links/personal info).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="emit_json",
        help="Emit JSON lines on stdout instead of human-readable stderr.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Synthesize a fake OTP message and write to the queue. No DB read.",
    )
    p.add_argument(
        "--queue-dir",
        default=str(OTP_QUEUE_DIR),
        help=f"OTP queue directory. Default: {OTP_QUEUE_DIR}",
    )
    p.add_argument(
        "--version",
        action="version",
        version="imessage_otp_listener.py 1.0.0",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Override queue dir globally (simple module-level shim).
    global OTP_QUEUE_DIR
    OTP_QUEUE_DIR = Path(args.queue_dir).expanduser()

    if args.dry_run:
        return dry_run(
            provider=args.provider or "kaggle",
            emit_json=args.emit_json,
            include_body=args.include_body,
        )

    try:
        since = parse_since(args.since)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    sender_patterns = [_glob_to_regex(s) for s in args.sender]

    return listen(
        since=since,
        sender_patterns=sender_patterns,
        provider_override=args.provider,
        timeout_s=args.timeout,
        poll_interval_s=max(0.25, args.poll_interval),
        include_body=args.include_body,
        emit_json=args.emit_json,
        stop_after_first=not args.all,
    )


if __name__ == "__main__":
    raise SystemExit(main())
