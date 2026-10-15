#!/usr/bin/env python3
"""yahoo_mail_reader.py — Read Yahoo Mail via IMAP and extract confirmation
links / OTP codes from incoming signup-confirmation emails.

Designed to support the cloud-provider auto-signup orchestrator (companion
to ``scripts/auto_signup_*.sh``): each signup triggers a confirmation email
to ``orginal_clawdbot@yahoo.com``; this reader polls the inbox, locates the
matching message (filtered by sender / subject / since-time), and returns
parsed URLs + numeric codes so the orchestrator can complete sign-up
without human intervention.

Authentication
--------------
Yahoo deprecated basic-auth IMAP in May 2024. Two paths exist in 2026:

1. **App password** (this script's default) — generated at
   https://login.yahoo.com/account/security under
   "Generate and manage app passwords". Requires 2-step verification.
   Set the resulting 16-character password in the env var
   ``YAHOO_APP_PASSWORD``. Never hard-code.
2. **OAuth 2.0 (OAUTHBEARER / XOAUTH2)** — Yahoo developer review required
   for the ``mail`` scope. Out of scope for this prototype.

IMAP server: ``imap.mail.yahoo.com:993`` over implicit TLS (verified
2026-05-17 against help.yahoo.com SLN4075).

Safety
------
* Password read from env var only — never CLI arg, never persisted.
* Messages are NOT marked ``\\Seen`` unless ``--mark-read`` is passed.
* Messages are NEVER deleted, moved, or replied-to.
* ``--dry-run`` exits cleanly without any network I/O.
"""
from __future__ import annotations

import argparse
import email
import email.header
import email.utils
import imaplib
import logging
import os
import re
import socket
import ssl
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.message import Message
from typing import Iterable, Sequence

try:
    from bs4 import BeautifulSoup  # type: ignore
    _HAVE_BS4 = True
except ImportError:  # pragma: no cover - documented optional dep
    _HAVE_BS4 = False

try:
    import dateparser  # type: ignore
    _HAVE_DATEPARSER = True
except ImportError:
    _HAVE_DATEPARSER = False

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

YAHOO_IMAP_HOST = "imap.mail.yahoo.com"
YAHOO_IMAP_PORT = 993
DEFAULT_FOLDER = "INBOX"
DEFAULT_POLL_TIMEOUT_S = 300
POLL_INITIAL_S = 1.0
POLL_MAX_S = 30.0
POLL_BACKOFF = 1.6

URL_RE = re.compile(r"https?://[^\s<>\"')]{20,500}", re.IGNORECASE)
# Numeric OTPs near keywords (within ~60 chars window, either side).
_OTP_KEYWORD = r"(?:code|verify|verification|confirm|confirmation|otp|pin|token)"
OTP_NEAR_RE = re.compile(
    rf"(?:{_OTP_KEYWORD}[^\d]{{0,60}}?(\d{{4,8}}))|(?:(\d{{4,8}})[^\d]{{0,60}}?{_OTP_KEYWORD})",
    re.IGNORECASE,
)

LOG = logging.getLogger("yahoo_mail_reader")


# --------------------------------------------------------------------------- #
# Data classes                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class ParsedMessage:
    uid: str
    from_addr: str
    subject: str
    date: str  # ISO-8601 UTC
    body_text: str
    urls: list[str] = field(default_factory=list)
    otp_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Time parsing                                                                #
# --------------------------------------------------------------------------- #


_RELATIVE_RE = re.compile(
    r"^\s*(\d+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|"
    r"h|hr|hrs|hour|hours|d|day|days)\s+ago\s*$",
    re.IGNORECASE,
)


def parse_since(value: str) -> datetime:
    """Parse a user-supplied since-time. Returns a tz-aware UTC datetime.

    Accepts:
      * "5 min ago", "2 hours ago", "1 day ago"
      * ISO-8601 ("2026-05-17T10:00:00Z", "2026-05-17 10:00")
      * Falls back to ``dateparser`` if installed.
    """
    value = value.strip()
    now = datetime.now(timezone.utc)

    m = _RELATIVE_RE.match(value)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("s"):
            delta = timedelta(seconds=n)
        elif unit.startswith("m"):
            # 'm' is ambiguous with month; treat plain 'm' as minute (most common).
            delta = timedelta(minutes=n)
        elif unit.startswith("h"):
            delta = timedelta(hours=n)
        elif unit.startswith("d"):
            delta = timedelta(days=n)
        else:
            raise ValueError(f"unsupported relative unit: {unit!r}")
        return now - delta

    # Try ISO-8601 a few obvious ways.
    iso_candidates = [value]
    if value.endswith("Z"):
        iso_candidates.append(value[:-1] + "+00:00")
    for cand in iso_candidates:
        try:
            dt = datetime.fromisoformat(cand)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass

    if _HAVE_DATEPARSER:
        dt = dateparser.parse(value, settings={"TIMEZONE": "UTC", "RETURN_AS_TIMEZONE_AWARE": True})  # type: ignore[union-attr]
        if dt is not None:
            return dt.astimezone(timezone.utc)

    raise ValueError(
        f"could not parse --since {value!r}; try '5 min ago' or '2026-05-17T10:00:00Z'"
    )


def _imap_date(dt: datetime) -> str:
    """Format a datetime as IMAP SINCE date (DD-Mon-YYYY)."""
    return dt.strftime("%d-%b-%Y")


# --------------------------------------------------------------------------- #
# Message parsing                                                             #
# --------------------------------------------------------------------------- #


def _decode_header(raw: str | None) -> str:
    if not raw:
        return ""
    parts = email.header.decode_header(raw)
    out: list[str] = []
    for chunk, charset in parts:
        if isinstance(chunk, bytes):
            try:
                out.append(chunk.decode(charset or "utf-8", errors="replace"))
            except (LookupError, TypeError):
                out.append(chunk.decode("utf-8", errors="replace"))
        else:
            out.append(chunk)
    return "".join(out).strip()


def _extract_body(msg: Message) -> str:
    """Prefer text/plain; fall back to stripped text/html via BS4 if available."""
    text_parts: list[str] = []
    html_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = (part.get("Content-Disposition") or "").lower()
            if "attachment" in disp:
                continue
            try:
                payload = part.get_payload(decode=True)
            except (TypeError, AttributeError):
                continue
            if payload is None:
                continue
            charset = part.get_content_charset() or "utf-8"
            try:
                decoded = payload.decode(charset, errors="replace")
            except LookupError:
                decoded = payload.decode("utf-8", errors="replace")
            if ctype == "text/plain":
                text_parts.append(decoded)
            elif ctype == "text/html":
                html_parts.append(decoded)
    else:
        payload = msg.get_payload(decode=True)
        if payload is not None:
            charset = msg.get_content_charset() or "utf-8"
            try:
                decoded = payload.decode(charset, errors="replace")
            except LookupError:
                decoded = payload.decode("utf-8", errors="replace")
            if msg.get_content_type() == "text/html":
                html_parts.append(decoded)
            else:
                text_parts.append(decoded)

    if text_parts:
        return "\n\n".join(text_parts).strip()
    if html_parts:
        html = "\n\n".join(html_parts)
        if _HAVE_BS4:
            soup = BeautifulSoup(html, "html.parser")
            # Preserve href targets as " (URL)" so URL_RE still finds them.
            for a in soup.find_all("a", href=True):
                a.append(f" ({a['href']})")
            return soup.get_text(separator="\n").strip()
        # Crude fallback: strip tags.
        return re.sub(r"<[^>]+>", " ", html).strip()
    return ""


def _extract_urls(body: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in URL_RE.finditer(body):
        url = m.group(0).rstrip(".,);]>'\"")
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _extract_otps(body: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in OTP_NEAR_RE.finditer(body):
        code = m.group(1) or m.group(2)
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def parse_message(uid: str, raw_bytes: bytes) -> ParsedMessage:
    # Defensive: a single bad message must NOT break the whole fetch batch
    # (cascade-unwind path observed in koyeb wave2c crash 2026-05-18 — an
    # AttributeError here propagated through magic_link_signup.py's only
    # `except RuntimeError` clause). Return an empty ParsedMessage stub
    # on any unexpected exception; the caller will simply find no urls/otps
    # on it and move on.
    try:
        msg = email.message_from_bytes(raw_bytes)
        from_addr = _decode_header(msg.get("From"))
        subject = _decode_header(msg.get("Subject"))
        date_hdr = msg.get("Date")
        if date_hdr:
            try:
                date_dt = email.utils.parsedate_to_datetime(date_hdr)
                if date_dt.tzinfo is None:
                    date_dt = date_dt.replace(tzinfo=timezone.utc)
                date_iso = date_dt.astimezone(timezone.utc).isoformat()
            except (TypeError, ValueError):
                date_iso = date_hdr
        else:
            date_iso = ""
        body = _extract_body(msg)
        return ParsedMessage(
            uid=uid,
            from_addr=from_addr,
            subject=subject,
            date=date_iso,
            body_text=body,
            urls=_extract_urls(body),
            otp_codes=_extract_otps(body),
        )
    except (AttributeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        LOG.warning("parse_message failed for uid=%s (%s); returning empty stub", uid, exc)
        return ParsedMessage(
            uid=uid,
            from_addr="",
            subject="",
            date="",
            body_text="",
        )


# --------------------------------------------------------------------------- #
# IMAP fetch                                                                  #
# --------------------------------------------------------------------------- #


def _build_search(since: datetime, sender: str | None, subject: str | None) -> list[str]:
    """Build IMAP SEARCH criteria. SINCE is day-granular per RFC 3501."""
    crit: list[str] = ["SINCE", _imap_date(since - timedelta(days=1))]
    if sender:
        crit += ["FROM", sender]
    if subject:
        crit += ["SUBJECT", subject]
    return crit


def fetch_messages(
    user: str,
    app_password: str,
    since: datetime,
    sender_filter: str | None = None,
    subject_filter: str | None = None,
    folder: str = DEFAULT_FOLDER,
    mark_read: bool = False,
    limit: int | None = None,
    host: str = YAHOO_IMAP_HOST,
    port: int = YAHOO_IMAP_PORT,
) -> list[ParsedMessage]:
    """Connect once and return matching messages newer than ``since``."""
    ctx = ssl.create_default_context()
    LOG.debug("connecting to %s:%d as %s", host, port, user)
    with imaplib.IMAP4_SSL(host, port, ssl_context=ctx) as imap:
        try:
            imap.login(user, app_password)
        except imaplib.IMAP4.error as exc:
            raise RuntimeError(
                f"IMAP login failed for {user!r}: {exc}. "
                "Likely causes: wrong app password, 2FA not enabled, or IMAP "
                "access disabled. See report yahoo_imap_setup_2026-05-17.md."
            ) from exc

        status, _ = imap.select(folder, readonly=not mark_read)
        if status != "OK":
            raise RuntimeError(f"could not select folder {folder!r}: {status}")

        crit = _build_search(since, sender_filter, subject_filter)
        LOG.debug("IMAP SEARCH %s", crit)
        status, data = imap.search(None, *crit)
        if status != "OK":
            raise RuntimeError(f"IMAP SEARCH failed: {status}")

        uids = data[0].split()
        if limit:
            uids = uids[-limit:]
        LOG.debug("matched %d uids", len(uids))

        results: list[ParsedMessage] = []
        for uid in uids:
            fetch_cmd = "(BODY.PEEK[])" if not mark_read else "(RFC822)"
            status, parts = imap.fetch(uid, fetch_cmd)
            if status != "OK" or not parts:
                LOG.warning("fetch failed for uid %s", uid)
                continue
            raw_bytes = None
            for part in parts:
                if isinstance(part, tuple) and len(part) >= 2:
                    raw_bytes = part[1]
                    break
            if raw_bytes is None:
                continue
            parsed = parse_message(uid.decode("ascii", errors="replace"), raw_bytes)
            # Drop messages strictly older than ``since`` (SEARCH is day-level).
            try:
                date_dt = datetime.fromisoformat(parsed.date)
                if date_dt.astimezone(timezone.utc) < since:
                    continue
            except (ValueError, TypeError):
                pass
            results.append(parsed)

        imap.close()
        imap.logout()
    return results


# --------------------------------------------------------------------------- #
# Polling                                                                     #
# --------------------------------------------------------------------------- #


def poll_until_match(
    user: str,
    app_password: str,
    since: datetime,
    sender_filter: str | None = None,
    subject_filter: str | None = None,
    folder: str = DEFAULT_FOLDER,
    mark_read: bool = False,
    timeout_s: int = DEFAULT_POLL_TIMEOUT_S,
    require_url: bool = True,
    require_otp: bool = False,
    host: str = YAHOO_IMAP_HOST,
    port: int = YAHOO_IMAP_PORT,
) -> list[ParsedMessage]:
    """Poll the inbox until at least one matching message is found or timeout."""
    deadline = time.monotonic() + timeout_s
    delay = POLL_INITIAL_S
    attempt = 0
    while True:
        attempt += 1
        LOG.debug("poll attempt %d (timeout in %.0fs)", attempt, deadline - time.monotonic())
        try:
            msgs = fetch_messages(
                user=user,
                app_password=app_password,
                since=since,
                sender_filter=sender_filter,
                subject_filter=subject_filter,
                folder=folder,
                mark_read=mark_read,
                host=host,
                port=port,
            )
        except (socket.error, ssl.SSLError, imaplib.IMAP4.error) as exc:
            LOG.warning("transient IMAP error on attempt %d: %s", attempt, exc)
            msgs = []

        if msgs:
            matched = [
                m
                for m in msgs
                if (not require_url or m.urls) and (not require_otp or m.otp_codes)
            ]
            if matched:
                return matched

        if time.monotonic() >= deadline:
            return []
        time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
        delay = min(delay * POLL_BACKOFF, POLL_MAX_S)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


SETUP_INSTRUCTIONS = """
YAHOO_APP_PASSWORD env var not set.

Setup steps:
  1. Open https://login.yahoo.com/account/security
  2. Ensure 2-step verification is ON (required for app passwords).
  3. Click 'Generate and manage app passwords' (or 'Generate app password').
  4. Enter an app name (e.g. 'clawdbot-orchestrator') and click Generate.
  5. Copy the 16-character password Yahoo displays. You cannot retrieve it
     later — only delete + regenerate.
  6. Export it for this shell:
        export YAHOO_APP_PASSWORD='xxxxxxxxxxxxxxxx'
     Or add to a NOT-COMMITTED file (e.g. ~/.config/yahoo_clawdbot.env).
  7. Re-run this command.

Server config used by this script (verified 2026-05-17):
  IMAP host  : imap.mail.yahoo.com
  IMAP port  : 993 (implicit TLS)
""".rstrip()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yahoo_mail_reader.py",
        description="Read Yahoo Mail via IMAP; extract confirmation URLs / OTPs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python yahoo_mail_reader.py --user orginal_clawdbot@yahoo.com \\\n"
            "      --since '5 min ago' --sender kaggle.com\n"
            "  python yahoo_mail_reader.py --user $YAHOO_USER --since '5 min ago' --dry-run\n"
            "\n"
            "Env vars:\n"
            "  YAHOO_APP_PASSWORD  16-char app password (required unless --dry-run)\n"
        ),
    )
    p.add_argument("--user", required=True, help="Yahoo address (e.g. user@yahoo.com)")
    p.add_argument("--since", default="10 min ago", help="Cutoff (e.g. '5 min ago' or ISO-8601)")
    p.add_argument("--sender", help="Filter From: contains this substring (IMAP FROM)")
    p.add_argument("--subject", help="Filter Subject: contains this substring (IMAP SUBJECT)")
    p.add_argument("--folder", default=DEFAULT_FOLDER, help="Mailbox folder (default INBOX)")
    p.add_argument("--mark-read", action="store_true",
                   help="Mark fetched messages as \\Seen (default: leave unread)")
    p.add_argument("--poll", action="store_true",
                   help="Poll with backoff until match or --timeout")
    p.add_argument("--timeout", type=int, default=DEFAULT_POLL_TIMEOUT_S,
                   help=f"Polling timeout seconds (default {DEFAULT_POLL_TIMEOUT_S})")
    p.add_argument("--require-url", action="store_true", default=True,
                   help="Only return messages containing >=1 URL (default true)")
    p.add_argument("--no-require-url", dest="require_url", action="store_false",
                   help="Disable the URL requirement")
    p.add_argument("--require-otp", action="store_true",
                   help="Only return messages containing >=1 numeric OTP")
    p.add_argument("--limit", type=int, help="Cap number of messages returned")
    p.add_argument("--json", action="store_true", help="Emit JSON array on stdout")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse args + exit; no network I/O, no password needed")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="Increase log verbosity (-v, -vv)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(level=max(level, logging.DEBUG), format="%(levelname)s %(message)s")

    try:
        since = parse_since(args.since)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        plan = {
            "mode": "dry-run",
            "host": YAHOO_IMAP_HOST,
            "port": YAHOO_IMAP_PORT,
            "user": args.user,
            "since_utc": since.isoformat(),
            "sender_filter": args.sender,
            "subject_filter": args.subject,
            "folder": args.folder,
            "mark_read": args.mark_read,
            "poll": args.poll,
            "timeout_s": args.timeout,
            "require_url": args.require_url,
            "require_otp": args.require_otp,
            "have_bs4": _HAVE_BS4,
            "have_dateparser": _HAVE_DATEPARSER,
        }
        if args.json:
            import json
            print(json.dumps(plan, indent=2))
        else:
            for k, v in plan.items():
                print(f"{k:16s} {v}")
        return 0

    app_password = os.environ.get("YAHOO_APP_PASSWORD")
    if not app_password:
        print(SETUP_INSTRUCTIONS, file=sys.stderr)
        return 1

    try:
        if args.poll:
            msgs = poll_until_match(
                user=args.user,
                app_password=app_password,
                since=since,
                sender_filter=args.sender,
                subject_filter=args.subject,
                folder=args.folder,
                mark_read=args.mark_read,
                timeout_s=args.timeout,
                require_url=args.require_url,
                require_otp=args.require_otp,
            )
        else:
            msgs = fetch_messages(
                user=args.user,
                app_password=app_password,
                since=since,
                sender_filter=args.sender,
                subject_filter=args.subject,
                folder=args.folder,
                mark_read=args.mark_read,
                limit=args.limit,
            )
            if args.require_url:
                msgs = [m for m in msgs if m.urls]
            if args.require_otp:
                msgs = [m for m in msgs if m.otp_codes]
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    if args.json:
        import json
        print(json.dumps([m.to_dict() for m in msgs], indent=2, default=str))
    else:
        if not msgs:
            print("(no matching messages)")
            return 4
        for m in msgs:
            print(f"--- uid={m.uid} ---")
            print(f"From    : {m.from_addr}")
            print(f"Subject : {m.subject}")
            print(f"Date    : {m.date}")
            if m.urls:
                print("URLs    :")
                for u in m.urls:
                    print(f"  - {u}")
            if m.otp_codes:
                print(f"OTPs    : {', '.join(m.otp_codes)}")
            print()
    return 0 if msgs else 4


if __name__ == "__main__":
    sys.exit(main())
