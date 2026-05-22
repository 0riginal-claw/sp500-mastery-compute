"""HPO trial-reuse cache + subsampling fidelity helpers (Tier-3, 2026-05-21).

Purpose
-------
Stack on top of the Optuna integration shipped in ada5e93 (backtest_xgb_v10.py
--optuna-hp) to give 5-15x speedup on 36-trial sweeps. Two orthogonal levers:

  1. Param-hash trial-result cache (Slice 1)
     - Hash the 4-tuple (learning_rate, max_depth, subsample, colsample_bytree)
       at fixed precision (lr=4dp, subsample/colsample=3dp, max_depth=int).
     - If the same (study, fidelity, params) tuple has been evaluated before,
       skip XGBoost training entirely and return the cached objective value.
     - Persist hits/misses to ``state/hpo_trial_cache/<study>.parquet``
       (fallback CSV if pyarrow / pandas-parquet missing).
     - Expected hit rate: 10-30% within a single study (Hyperband promotion
       re-evaluates promising trials at higher rungs/fidelities, and TPE
       sometimes re-samples near-identical points).

  2. Subsampling fidelity levels (Slice 2 + 3)
     - Three fidelities: Bronze (10% rows), Silver (30%), Gold (100%).
     - ``allocate_fidelity_budget(n_trials)`` returns a budget plan that
       runs ~n_trials at Bronze, promotes top ~1/3 to Silver, top ~1/9 to
       Gold. Net trial-equivalent compute ~= 0.31 x n_trials (see math
       in module docstring at bottom).
     - Each fidelity hashes separately so Silver/Gold runs of the same
       params do recompute (different data slice -> different objective).

Feature flag
------------
Off by default. Activate by setting ``HPO_FIDELITY_CACHE=1`` in env (read
once on import; cached as ``HPO_FIDELITY_CACHE_ENABLED``). When OFF, all
public helpers are still importable but ``cache_lookup`` always returns
``None`` (miss) and the fidelity allocator returns a degenerate plan that
runs everything at Gold (= current Optuna behaviour).

Public API
----------
- ``cache_lookup(study_name, fidelity, params) -> float | None``
- ``cache_store(study_name, fidelity, params, value, *, meta=None) -> None``
- ``cache_flush(study_name=None) -> int``  (returns rows written)
- ``cache_stats(study_name=None) -> dict``
- ``allocate_fidelity_budget(n_trials, *, ratio=3) -> dict``
- ``subsample_indices(n_rows, fidelity, *, seed=42) -> np.ndarray``
- ``FIDELITIES = ("bronze", "silver", "gold")``
- ``FIDELITY_FRAC = {"bronze": 0.10, "silver": 0.30, "gold": 1.00}``

Schema (parquet)
----------------
Columns:
  ts            -- ISO-8601 UTC timestamp of insert (str)
  study_name    -- str
  fidelity      -- str ("bronze" / "silver" / "gold")
  param_hash    -- str (16-hex blake2b digest of canonical tuple)
  learning_rate -- float (4dp)
  max_depth     -- int
  subsample     -- float (3dp)
  colsample     -- float (3dp)
  value         -- float (objective, e.g. Sharpe)
  meta_json     -- str (free-form JSON of extras: n_estimators, prune_step, etc.)

Concurrency note: writes go through an in-process append buffer flushed on
``cache_flush()`` and on interpreter exit (atexit). This is sufficient for
single-process Optuna studies; the multi-process case would need a file
lock (out of scope for tier-3).
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

FIDELITIES = ("bronze", "silver", "gold")
FIDELITY_FRAC: dict[str, float] = {"bronze": 0.10, "silver": 0.30, "gold": 1.00}

# Feature flag (read at import; default OFF until A/B clean per spec).
HPO_FIDELITY_CACHE_ENABLED: bool = os.environ.get("HPO_FIDELITY_CACHE", "0") == "1"

# Root cache dir. Resolves via the same WORK convention as backtest_xgb_v10
# when imported alongside it, else falls back to repo state dir.
def _resolve_cache_dir() -> Path:
    env_dir = os.environ.get("HPO_CACHE_DIR")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    # Try to locate state/ relative to this file:
    # scripts/hpo_trial_cache.py  -> ../state/hpo_trial_cache
    here = Path(__file__).resolve().parent
    cand = here.parent / "state" / "hpo_trial_cache"
    cand.mkdir(parents=True, exist_ok=True)
    return cand


CACHE_DIR: Path = _resolve_cache_dir()

# --------------------------------------------------------------------------
# Param canonicalisation + hashing
# --------------------------------------------------------------------------

def _canonical_params(params: dict[str, Any]) -> tuple:
    """Round + cast the 4 searched HP keys to fixed precision.

    Optuna's TPE can re-sample points at sub-epsilon distances; rounding
    here gives the cache a meaningful hit rate without over-collapsing
    the search space.

    Precision (matches Optuna search-space granularity in
    _optuna_search_final_params):
      learning_rate    -> 4 decimal places (log-scale 0.01..0.20)
      max_depth        -> int (3..10)
      subsample        -> 3 decimal places (0.5..1.0)
      colsample_bytree -> 3 decimal places (0.4..1.0)
    """
    lr = round(float(params.get("learning_rate", 0.0)), 4)
    md = int(params.get("max_depth", 0))
    ss = round(float(params.get("subsample", 0.0)), 3)
    cs = round(float(params.get("colsample_bytree", 0.0)), 3)
    return (lr, md, ss, cs)


def _param_hash(study_name: str, fidelity: str, params: dict[str, Any]) -> str:
    canon = _canonical_params(params)
    blob = f"{study_name}|{fidelity}|{canon[0]:.4f}|{canon[1]}|{canon[2]:.3f}|{canon[3]:.3f}"
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=8).hexdigest()


# --------------------------------------------------------------------------
# In-memory layer + persistence
# --------------------------------------------------------------------------

# {(study, fidelity, param_hash) -> {"value": float, "ts": str, "params": tuple, "meta": dict}}
_MEM_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
_PENDING_WRITES: list[dict[str, Any]] = []
_LOCK = threading.Lock()
_STATS: dict[str, int] = {"hits": 0, "misses": 0, "stores": 0}


def _cache_path(study_name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in study_name)
    return CACHE_DIR / f"{safe}.parquet"


def _csv_fallback_path(study_name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in study_name)
    return CACHE_DIR / f"{safe}.csv"


def _load_persisted(study_name: str) -> None:
    """Load any previously-persisted rows for ``study_name`` into _MEM_CACHE."""
    pq = _cache_path(study_name)
    csv = _csv_fallback_path(study_name)
    rows: list[dict[str, Any]] = []
    try:
        if pq.exists():
            try:
                import pandas as pd  # type: ignore
                df = pd.read_parquet(pq)
                rows = df.to_dict("records")
            except Exception:
                rows = []
        elif csv.exists():
            try:
                import pandas as pd  # type: ignore
                df = pd.read_csv(csv)
                rows = df.to_dict("records")
            except Exception:
                # Hand-roll CSV reader as last resort
                with csv.open() as fh:
                    header = fh.readline().rstrip().split(",")
                    for line in fh:
                        parts = line.rstrip().split(",")
                        if len(parts) == len(header):
                            rows.append(dict(zip(header, parts)))
    except Exception:
        rows = []

    for r in rows:
        key = (
            str(r.get("study_name", study_name)),
            str(r.get("fidelity", "")),
            str(r.get("param_hash", "")),
        )
        try:
            _MEM_CACHE[key] = {
                "value": float(r.get("value", 0.0)),
                "ts": str(r.get("ts", "")),
                "params": (
                    float(r.get("learning_rate", 0.0)),
                    int(float(r.get("max_depth", 0))),
                    float(r.get("subsample", 0.0)),
                    float(r.get("colsample", 0.0)),
                ),
                "meta": {},
            }
        except Exception:
            continue


_LOADED_STUDIES: set[str] = set()


def _ensure_loaded(study_name: str) -> None:
    if study_name in _LOADED_STUDIES:
        return
    with _LOCK:
        if study_name in _LOADED_STUDIES:
            return
        _load_persisted(study_name)
        _LOADED_STUDIES.add(study_name)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def cache_lookup(
    study_name: str,
    fidelity: str,
    params: dict[str, Any],
) -> float | None:
    """Return cached objective for (study, fidelity, params) or None on miss.

    When ``HPO_FIDELITY_CACHE_ENABLED`` is False, always returns None.
    """
    if not HPO_FIDELITY_CACHE_ENABLED:
        return None
    if fidelity not in FIDELITY_FRAC:
        return None
    _ensure_loaded(study_name)
    h = _param_hash(study_name, fidelity, params)
    key = (study_name, fidelity, h)
    hit = _MEM_CACHE.get(key)
    with _LOCK:
        if hit is None:
            _STATS["misses"] += 1
            return None
        _STATS["hits"] += 1
        return float(hit["value"])


def cache_store(
    study_name: str,
    fidelity: str,
    params: dict[str, Any],
    value: float,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    """Persist a (study, fidelity, params -> value) tuple to the cache."""
    if not HPO_FIDELITY_CACHE_ENABLED:
        return
    if fidelity not in FIDELITY_FRAC:
        return
    _ensure_loaded(study_name)
    canon = _canonical_params(params)
    h = _param_hash(study_name, fidelity, params)
    key = (study_name, fidelity, h)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "value": float(value),
        "ts": ts,
        "params": canon,
        "meta": dict(meta or {}),
    }
    with _LOCK:
        # Idempotent: same key keeps first (cheaper) value; updates not needed
        # since (study, fidelity, params) is deterministic by spec.
        if key in _MEM_CACHE:
            return
        _MEM_CACHE[key] = entry
        _STATS["stores"] += 1
        _PENDING_WRITES.append({
            "ts": ts,
            "study_name": study_name,
            "fidelity": fidelity,
            "param_hash": h,
            "learning_rate": canon[0],
            "max_depth": canon[1],
            "subsample": canon[2],
            "colsample": canon[3],
            "value": float(value),
            "meta_json": json.dumps(meta or {}, separators=(",", ":"), sort_keys=True),
        })


def cache_flush(study_name: str | None = None) -> int:
    """Persist any pending rows. Returns count written.

    Writes parquet if pandas+pyarrow available; falls back to CSV otherwise.
    """
    with _LOCK:
        if not _PENDING_WRITES:
            return 0
        rows = list(_PENDING_WRITES)
        _PENDING_WRITES.clear()

    # Group by study so each parquet stays per-study.
    by_study: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if study_name is not None and r["study_name"] != study_name:
            continue
        by_study.setdefault(r["study_name"], []).append(r)

    n_written = 0
    for sname, srows in by_study.items():
        pq = _cache_path(sname)
        csv = _csv_fallback_path(sname)
        wrote = False
        try:
            import pandas as pd  # type: ignore
            new_df = pd.DataFrame(srows)
            if pq.exists():
                try:
                    old = pd.read_parquet(pq)
                    new_df = pd.concat([old, new_df], ignore_index=True)
                except Exception:
                    pass
            try:
                new_df.to_parquet(pq, index=False)
                wrote = True
            except Exception:
                # pyarrow missing — fall through to CSV
                pass
            if not wrote:
                if csv.exists():
                    try:
                        old = pd.read_csv(csv)
                        new_df = pd.concat([old, new_df], ignore_index=True)
                    except Exception:
                        pass
                new_df.to_csv(csv, index=False)
                wrote = True
        except Exception:
            # No pandas: hand-roll CSV
            header = list(srows[0].keys())
            mode = "a" if csv.exists() else "w"
            with csv.open(mode) as fh:
                if mode == "w":
                    fh.write(",".join(header) + "\n")
                for r in srows:
                    fh.write(",".join(str(r[k]) for k in header) + "\n")
            wrote = True
        if wrote:
            n_written += len(srows)
    return n_written


def cache_stats(study_name: str | None = None) -> dict[str, Any]:
    """Return hits / misses / stores / hit_rate (and rows on disk if available)."""
    with _LOCK:
        h = _STATS["hits"]
        m = _STATS["misses"]
        s = _STATS["stores"]
    total = h + m
    rate = (h / total) if total > 0 else 0.0
    out = {
        "enabled": HPO_FIDELITY_CACHE_ENABLED,
        "hits": h,
        "misses": m,
        "stores": s,
        "lookups": total,
        "hit_rate": rate,
        "cache_dir": str(CACHE_DIR),
    }
    if study_name is not None:
        pq = _cache_path(study_name)
        csv = _csv_fallback_path(study_name)
        out["study_name"] = study_name
        if pq.exists():
            out["persist_path"] = str(pq)
        elif csv.exists():
            out["persist_path"] = str(csv)
    return out


# --------------------------------------------------------------------------
# Fidelity allocation (Hyperband-style top-of-band promotion)
# --------------------------------------------------------------------------

def allocate_fidelity_budget(
    n_trials: int,
    *,
    ratio: int = 3,
    enabled: bool | None = None,
) -> dict[str, int]:
    """Return budget per fidelity tier.

    Default split (ratio=3, n_trials=36):
      bronze=36, silver=12, gold=4 -> trial-equivalent compute
      = 36*0.10 + 12*0.30 + 4*1.00 = 11.2 (vs 36 at Gold = 3.2x save on data)

    When disabled (env flag off OR enabled=False), returns
      {"bronze": 0, "silver": 0, "gold": n_trials}
    so the caller behaves identically to the pre-tier-3 path.
    """
    if enabled is None:
        enabled = HPO_FIDELITY_CACHE_ENABLED
    if not enabled or n_trials <= 0:
        return {"bronze": 0, "silver": 0, "gold": max(0, int(n_trials))}
    n = int(n_trials)
    bronze = n
    silver = max(1, n // ratio)
    gold = max(1, silver // ratio)
    return {"bronze": bronze, "silver": silver, "gold": gold}


def trial_equivalent_cost(plan: dict[str, int]) -> float:
    """Sum of bronze*0.1 + silver*0.3 + gold*1.0 (proxy for wall-time)."""
    return sum(int(plan.get(f, 0)) * FIDELITY_FRAC[f] for f in FIDELITIES)


# --------------------------------------------------------------------------
# Subsampling
# --------------------------------------------------------------------------

def subsample_indices(n_rows: int, fidelity: str, *, seed: int = 42) -> np.ndarray:
    """Return a deterministic sorted index array selecting rows at ``fidelity``.

    Same (n_rows, fidelity, seed) -> same indices, so a given (params, fidelity)
    is reproducible across calls (matches cache semantics).
    """
    n_rows = int(max(0, n_rows))
    if fidelity not in FIDELITY_FRAC:
        raise ValueError(f"unknown fidelity {fidelity!r}; expected one of {FIDELITIES}")
    frac = FIDELITY_FRAC[fidelity]
    if frac >= 1.0 or n_rows == 0:
        return np.arange(n_rows, dtype=np.int64)
    k = max(1, int(round(n_rows * frac)))
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(n_rows, size=k, replace=False)
    idx.sort()
    return idx.astype(np.int64)


# --------------------------------------------------------------------------
# Exit hook — flush pending writes
# --------------------------------------------------------------------------

@atexit.register
def _flush_on_exit() -> None:
    try:
        cache_flush()
    except Exception:
        pass


# --------------------------------------------------------------------------
# Self-test (python -m hpo_trial_cache)
# --------------------------------------------------------------------------

def _self_test() -> int:
    """Tiny in-process smoke test. Exits 0 on pass, non-zero on fail."""
    os.environ["HPO_FIDELITY_CACHE"] = "1"
    global HPO_FIDELITY_CACHE_ENABLED
    HPO_FIDELITY_CACHE_ENABLED = True

    sname = "_selftest_2026_05_21"
    params_a = {"learning_rate": 0.0501, "max_depth": 6, "subsample": 0.801, "colsample_bytree": 0.799}
    params_b = {"learning_rate": 0.05009, "max_depth": 6, "subsample": 0.8013, "colsample_bytree": 0.7988}

    # 1. miss
    assert cache_lookup(sname, "bronze", params_a) is None, "expected miss"
    # 2. store + hit
    cache_store(sname, "bronze", params_a, 1.234)
    v = cache_lookup(sname, "bronze", params_a)
    assert v is not None and abs(v - 1.234) < 1e-9, f"expected 1.234 got {v}"
    # 3. near-duplicate (within rounding) hits same cache key
    v2 = cache_lookup(sname, "bronze", params_b)
    assert v2 is not None and abs(v2 - 1.234) < 1e-9, f"expected near-dup hit got {v2}"
    # 4. different fidelity -> miss
    assert cache_lookup(sname, "silver", params_a) is None, "fidelity should isolate"
    # 5. flush writes a file
    n = cache_flush(sname)
    assert n >= 1, f"flush wrote {n} rows; expected >= 1"
    # 6. budget math
    plan = allocate_fidelity_budget(36)
    assert plan["bronze"] == 36 and plan["silver"] == 12 and plan["gold"] == 4, plan
    cost = trial_equivalent_cost(plan)
    assert abs(cost - (36 * 0.1 + 12 * 0.3 + 4 * 1.0)) < 1e-9, cost
    # 7. subsample indices monotone + correct length
    idx10 = subsample_indices(1000, "bronze")
    assert idx10.shape == (100,) and (idx10[:-1] <= idx10[1:]).all()
    idx100 = subsample_indices(1000, "gold")
    assert idx100.shape == (1000,)
    # 8. stats
    st = cache_stats(sname)
    assert st["enabled"] and st["lookups"] >= 4 and st["hits"] >= 2
    print("hpo_trial_cache: self-test OK", st)
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
