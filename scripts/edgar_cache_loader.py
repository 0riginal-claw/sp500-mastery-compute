# autosolve_skip: data loader infra (read-only SQLite + parsed.json scanner)
"""
edgar_cache_loader.py — EDGAR filings cache loader.

Two sources, in priority order:
  1. SQLite mirror at `claudes test/data/edgar/data/edgar.db`
     (57,066 filings, 500 tickers, 2020-2026, schema: accession_number/ticker/
      cik/form/filed_at/period_of_report/primary_url/storage_path/sha256)
  2. On-disk parsed.json under `Ph0tis/Edgar/data/filings/<year>/<TICKER>/<FORM>/
     <YYYY-MM-DD>_<accession>/parsed.json`

The SQLite is the metadata index (used by edgar_extras_features.py for
form/date filtering). When callers need the parsed body (sections, items,
period_of_report) we look up the matching parsed.json in the Drive tree.

Public API:
  EdgarCache.get_filings(ticker, form_type=None, start=None, end=None)
      -> List[Dict] with keys: accession_number, ticker, form, filed_at,
         period_of_report, primary_url, parsed (optional dict from parsed.json)
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

LOG = logging.getLogger(__name__)

EDGAR_DB_DRIVE = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/claudes test/data/edgar/data/edgar.db"
)
EDGAR_FILINGS_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/Ph0tis/Edgar/data/filings"
)
EDGAR_DB_LOCAL = "/tmp/edgar_cache_loader.db"


class EdgarCache:
    """Read-only cache loader. Pull SQLite once per process, then query in-mem."""

    _local_db_ready: bool = False

    @classmethod
    def _ensure_local_db(cls) -> bool:
        if cls._local_db_ready and os.path.exists(EDGAR_DB_LOCAL):
            return True
        if not os.path.exists(EDGAR_DB_DRIVE):
            LOG.warning("[edgar_cache] source DB missing: %s", EDGAR_DB_DRIVE)
            return False
        # Local copy via SQLite online backup (WAL-safe)
        try:
            src = sqlite3.connect(
                f"file:{EDGAR_DB_DRIVE}?mode=ro", uri=True, timeout=30.0
            )
            dst = sqlite3.connect(EDGAR_DB_LOCAL)
            with dst:
                src.backup(dst)
            src.close()
            dst.close()
            cls._local_db_ready = True
            return True
        except Exception as e:
            LOG.warning("[edgar_cache] backup failed (%s); using Drive read-only", e)
            cls._local_db_ready = False
            return os.path.exists(EDGAR_DB_DRIVE)

    @classmethod
    def _connect(cls):
        if cls._ensure_local_db() and os.path.exists(EDGAR_DB_LOCAL):
            return sqlite3.connect(EDGAR_DB_LOCAL, timeout=10.0)
        return sqlite3.connect(
            f"file:{EDGAR_DB_DRIVE}?mode=ro", uri=True, timeout=10.0
        )

    @classmethod
    def _drive_parsed_path(cls, ticker: str, form: str, filed_at: str,
                           accession: str) -> Optional[Path]:
        """Resolve the parsed.json file on Drive for a given filing.

        Layout: <root>/<year>/<TICKER>/<FORM>/<filed_at>_<accession>/parsed.json
        """
        try:
            year = filed_at[:4]
            folder = f"{filed_at}_{accession}"
            p = EDGAR_FILINGS_ROOT / year / ticker.upper() / form / folder / "parsed.json"
            return p if p.exists() else None
        except Exception:
            return None

    @classmethod
    def get_filings(
        cls,
        ticker: str,
        form_type: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        include_parsed: bool = False,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Return filings rows for ticker filtered by form/date range.

        Args:
          ticker:      symbol (case-insensitive on lookup).
          form_type:   "10-K", "10-Q", "8-K", "DEF 14A", "S-1", "8-K/A", or None.
                       If None, returns all forms.
          start, end:  ISO date strings, inclusive. Compared against filed_at.
          include_parsed: when True, also reads parsed.json from Drive for each row.
          limit:       cap on results.

        Returns list of dicts; empty list on miss or DB unavailable.
        """
        if not ticker:
            return []
        sql = (
            "SELECT accession_number, ticker, cik, form, filed_at, "
            "period_of_report, primary_url, storage_path "
            "FROM filings WHERE ticker = ?"
        )
        params: List = [ticker.upper()]
        if form_type:
            sql += " AND form = ?"
            params.append(form_type)
        if start:
            sql += " AND filed_at >= ?"
            params.append(start)
        if end:
            sql += " AND filed_at <= ?"
            params.append(end)
        sql += " ORDER BY filed_at"
        if limit:
            sql += f" LIMIT {int(limit)}"

        try:
            with cls._connect() as con:
                cur = con.execute(sql, params)
                cols = [c[0] for c in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception as e:
            LOG.warning("[edgar_cache] query failed %s/%s: %s", ticker, form_type, e)
            return []

        if include_parsed:
            for r in rows:
                pj = cls._drive_parsed_path(
                    r["ticker"], r["form"], r["filed_at"], r["accession_number"]
                )
                if pj is not None:
                    try:
                        with open(pj, "r") as fh:
                            r["parsed"] = json.load(fh)
                    except Exception:
                        r["parsed"] = None
                else:
                    r["parsed"] = None
        return rows

    @classmethod
    def has_form(cls, ticker: str, form_type: str, start: str, end: str) -> bool:
        """Cheap existence check."""
        return len(cls.get_filings(ticker, form_type, start, end, limit=1)) > 0


# Smoke
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    rows = EdgarCache.get_filings(tk, "10-K", "2024-01-01", "2024-12-31")
    print(f"{tk} 10-K 2024: {len(rows)} filings")
    for r in rows[:3]:
        print(f"  {r['filed_at']} {r['form']} {r['accession_number']}")
    rows8k = EdgarCache.get_filings(tk, "8-K", "2024-01-01", "2024-12-31",
                                     include_parsed=True, limit=2)
    print(f"{tk} 8-K 2024 (w/ parsed): {len(rows8k)} filings")
    for r in rows8k:
        has_parsed = r.get("parsed") is not None
        print(f"  {r['filed_at']} parsed={has_parsed}")
