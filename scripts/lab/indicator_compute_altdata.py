"""indicator_compute_altdata.py — Alt-data NUMERIC indicators (PATH B, 2026-05-29).

These are numeric per-bar series (NOT boolean event tokens — those live in the role
parser elsewhere). Each function takes:

    ticker:          str — the equity ticker the indicator is computed for
    bar_timestamps:  pd.DatetimeIndex — the per-bar timestamps to anchor on

and returns:

    pd.Series — same length as `bar_timestamps`, NaN where no data is available.

Sources used (causal — no lookahead):
    lab.knowledge.edgar.get_filings(ticker, form=...) — SEC filings (Form 4, 8-K, etc.)
        timestamp col: 'filed_at' (when SEC accepted; the public-as-of timestamp)
    lab.knowledge.govtrades.get_congress_trades(ticker) — congress disclosures
        timestamp col: 'disclosure_date'  (the moment the public learns)
    lab.knowledge.govtrades.get_offexchange(ticker) — Dark Pool Index (FINRA SHO)
        timestamp col: 'as_of_date'
    lab.knowledge.govtrades.get_lobbying(ticker) — lobbying $
        timestamp col: 'period_end' or 'report_date'
    lab.knowledge.govtrades.get_contracts(ticker) — gov contract awards
        timestamp col: 'awarded_at' or 'date'
    lab.knowledge.news.get_news(ticker, ...) — news velocity
        timestamp col: 'published_utc'

NO-LOOKAHEAD discipline
-----------------------
For each bar_timestamp `t`, we filter source rows to those with their timestamp
<= t — never the raw transaction/event date, but the public-as-of-disclosure date.
That's the moment the information is actually tradable.

Axes
----
Most are 'volume_conviction' (confirmation flavor) or 'structure_geometry' (event
anchor flavor). The new 'alt_data' axis is reserved if a 7th axis is later added.

Functions
---------
- form4_insider_cluster_score   — Form 4 buy cluster intensity
- congress_lead_lag             — days since latest congress disclosure
- news_velocity_zscore          — 7d news count Z vs 90d baseline
- dark_pool_divergence_z        — offexchange short-vol-ratio z (5d)
- lobbying_intensity            — 90d lobby $ vs trailing-year avg
- gov_contract_inflow           — 90d contract $ awarded
- 8k_pulse / eight_k_pulse      — count of 8-Ks in last 5d
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-variant fetch cache (task #76, 2026-05-29).
#
# Identical pattern + rationale to _ALTDATA_FETCH_CACHE in hypothesis_runner.py:
# `_AltDataNumericResolver` is recreated per championship variant. Each variant
# uses one or more numeric altdata tokens (form4_insider_cluster_score,
# congress_lead_lag, news_velocity_zscore, etc.), and each call below would
# otherwise re-fetch via lab.knowledge.{edgar, govtrades, news} → sqlite hit
# every time. Caching the DataFrames keyed by (ticker, source_kind, [args])
# eliminates the per-variant amortization cost so 23 variants run for the same
# fetch cost as 1.
#
# Source-kind keys (paired with the _fetch_* function below):
#   "filings:<form>"  → _fetch_filings(ticker, form=<form>)
#   "filings:ALL"     → _fetch_filings(ticker, form=None)
#   "congress"        → _fetch_congress(ticker)
#   "offex"           → _fetch_offex(ticker)
#   "lobbying"        → _fetch_lobbying(ticker)
#   "contracts"       → _fetch_contracts(ticker)
#   "news:<start>:<end>" → _fetch_news(ticker, start, end) — keyed by date range
#
# DataFrames returned by `_altdata_cache_get` MUST NOT be mutated by callers;
# they're shared references across variants. All downstream consumers in this
# module read-only (slice / filter / sort).
# ---------------------------------------------------------------------------
_ALTDATA_FETCH_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}


def _altdata_cache_get(ticker: str, source_kind: str) -> Optional[pd.DataFrame]:
    return _ALTDATA_FETCH_CACHE.get((ticker.upper(), source_kind))


def _altdata_cache_set(ticker: str, source_kind: str, df: pd.DataFrame) -> None:
    _ALTDATA_FETCH_CACHE[(ticker.upper(), source_kind)] = df


def _altdata_cache_clear(ticker: Optional[str] = None) -> int:
    if ticker is None:
        n = len(_ALTDATA_FETCH_CACHE)
        _ALTDATA_FETCH_CACHE.clear()
        return n
    tk = ticker.upper()
    keys = [k for k in _ALTDATA_FETCH_CACHE if k[0] == tk]
    for k in keys:
        del _ALTDATA_FETCH_CACHE[k]
    return len(keys)


# ---------------------------------------------------------------------------
# Helpers — fetch + normalize alt-data once per (ticker, source) per process call.
# ---------------------------------------------------------------------------


def _to_dataframe(rows: Any) -> pd.DataFrame:
    """Coerce loader return (list[dict] OR DataFrame OR None) into a DataFrame."""
    if rows is None:
        return pd.DataFrame()
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    if isinstance(rows, list):
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    raise TypeError(f"Cannot coerce {type(rows).__name__} to DataFrame for alt-data")


def _parse_dt(s: pd.Series) -> pd.Series:
    """Parse a column to tz-naive datetime64[ns]. Drops timezone info for consistent comparison."""
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)


def _ensure_naive_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Ensure bar_timestamps is tz-naive DatetimeIndex for safe comparison."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def _empty_series(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(np.nan, index=_ensure_naive_index(idx), dtype="float64")


def _fetch_filings(ticker: str, form: Optional[str] = None) -> pd.DataFrame:
    cache_key = f"filings:{form}" if form else "filings:ALL"
    cached = _altdata_cache_get(ticker, cache_key)
    if cached is not None:
        return cached
    rows = None
    try:
        from lab.knowledge import edgar
        try:
            rows = edgar.get_filings(ticker, form=form) if form else edgar.get_filings(ticker)
        except TypeError:
            # Underlying EdgarCache.get_filings doesn't accept form kwarg; pull all then filter.
            try:
                import edgar_cache_loader as _ecl
                cache = _ecl.EdgarCache()
                rows = cache.get_filings(ticker)
            except Exception as e:  # noqa: BLE001
                logger.debug("edgar direct-cache fetch failed for %s: %s", ticker, e)
                rows = None
    except Exception as e:  # noqa: BLE001
        logger.debug("edgar fetch failed for %s/%s: %s", ticker, form, e)
    df = _to_dataframe(rows)
    if df.empty:
        _altdata_cache_set(ticker, cache_key, df)
        return df
    if "filed_at" in df.columns:
        df["filed_at"] = _parse_dt(df["filed_at"])
    if form and "form" in df.columns:
        # Form 4 stored variously as "4", "Form 4"; 8-K as "8-K"
        form_norm = form.replace("Form ", "").strip()
        df = df[df["form"].astype(str).str.replace("Form ", "").str.strip() == form_norm]
    _altdata_cache_set(ticker, cache_key, df)
    return df


def _fetch_congress(ticker: str) -> pd.DataFrame:
    cached = _altdata_cache_get(ticker, "congress")
    if cached is not None:
        return cached
    try:
        from lab.knowledge import govtrades
        rows = govtrades.get_congress_trades(ticker)
    except Exception as e:  # noqa: BLE001
        logger.debug("congress fetch failed for %s: %s", ticker, e)
        empty = pd.DataFrame()
        _altdata_cache_set(ticker, "congress", empty)
        return empty
    df = _to_dataframe(rows)
    if df.empty:
        _altdata_cache_set(ticker, "congress", df)
        return df
    for col in ("disclosure_date", "disclosed_at", "report_date"):
        if col in df.columns:
            df["disclosure_date"] = _parse_dt(df[col])
            break
    _altdata_cache_set(ticker, "congress", df)
    return df


def _fetch_offex(ticker: str) -> pd.DataFrame:
    cached = _altdata_cache_get(ticker, "offex")
    if cached is not None:
        return cached
    try:
        from lab.knowledge import govtrades
        rows = govtrades.get_offexchange(ticker)
    except Exception as e:  # noqa: BLE001
        logger.debug("offex fetch failed for %s: %s", ticker, e)
        empty = pd.DataFrame()
        _altdata_cache_set(ticker, "offex", empty)
        return empty
    df = _to_dataframe(rows)
    if df.empty:
        _altdata_cache_set(ticker, "offex", df)
        return df
    for col in ("as_of_date", "date", "period_end"):
        if col in df.columns:
            df["as_of_date"] = _parse_dt(df[col])
            break
    _altdata_cache_set(ticker, "offex", df)
    return df


def _fetch_lobbying(ticker: str) -> pd.DataFrame:
    cached = _altdata_cache_get(ticker, "lobbying")
    if cached is not None:
        return cached
    try:
        from lab.knowledge import govtrades
        rows = govtrades.get_lobbying(ticker)
    except Exception as e:  # noqa: BLE001
        logger.debug("lobby fetch failed for %s: %s", ticker, e)
        empty = pd.DataFrame()
        _altdata_cache_set(ticker, "lobbying", empty)
        return empty
    df = _to_dataframe(rows)
    if df.empty:
        _altdata_cache_set(ticker, "lobbying", df)
        return df
    for col in ("period_end", "report_date", "filed_at", "date"):
        if col in df.columns:
            df["period_end"] = _parse_dt(df[col])
            break
    _altdata_cache_set(ticker, "lobbying", df)
    return df


def _fetch_contracts(ticker: str) -> pd.DataFrame:
    cached = _altdata_cache_get(ticker, "contracts")
    if cached is not None:
        return cached
    try:
        from lab.knowledge import govtrades
        rows = govtrades.get_contracts(ticker)
    except Exception as e:  # noqa: BLE001
        logger.debug("contracts fetch failed for %s: %s", ticker, e)
        empty = pd.DataFrame()
        _altdata_cache_set(ticker, "contracts", empty)
        return empty
    df = _to_dataframe(rows)
    if df.empty:
        _altdata_cache_set(ticker, "contracts", df)
        return df
    for col in ("awarded_at", "award_date", "date", "period_end"):
        if col in df.columns:
            df["awarded_at"] = _parse_dt(df[col])
            break
    _altdata_cache_set(ticker, "contracts", df)
    return df


def _fetch_news(ticker: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    # Key by date range so reuse only happens when the requested window is identical.
    # Most callers (news_velocity_zscore + 8k_pulse) use the full bar timeframe so the
    # ranges agree across variants for a given ticker/timeframe.
    cache_key = f"news:{start_dt.date()}:{end_dt.date()}"
    cached = _altdata_cache_get(ticker, cache_key)
    if cached is not None:
        return cached
    try:
        from lab.knowledge import news
        rows = news.get_news(ticker, start=str(start_dt.date()), end=str(end_dt.date()))
    except Exception as e:  # noqa: BLE001
        logger.debug("news fetch failed for %s: %s", ticker, e)
        empty = pd.DataFrame()
        _altdata_cache_set(ticker, cache_key, empty)
        return empty
    df = _to_dataframe(rows)
    if df.empty:
        _altdata_cache_set(ticker, cache_key, df)
        return df
    if "published_utc" in df.columns:
        df["published_utc"] = _parse_dt(df["published_utc"])
    _altdata_cache_set(ticker, cache_key, df)
    return df


# ---------------------------------------------------------------------------
# 1. Form 4 insider cluster score
# ---------------------------------------------------------------------------


def _form4_insider_cluster_score_loop_v1(ticker: str, bar_timestamps: pd.DatetimeIndex,
                                          lookback_d: int = 5) -> pd.Series:
    """Original O(N*M) reference implementation — kept for equality regression tests."""
    idx = _ensure_naive_index(bar_timestamps)
    df = _fetch_filings(ticker, form="Form 4")
    if df.empty or "filed_at" not in df.columns:
        return _empty_series(idx)

    value_col = None
    for c in ("value_usd", "transaction_value", "amount_usd", "amount"):
        if c in df.columns:
            value_col = c
            break
    insider_col = None
    for c in ("reporting_owner", "insider_name", "reporter_name", "filer"):
        if c in df.columns:
            insider_col = c
            break
    if "transaction_code" in df.columns:
        df = df[df["transaction_code"].astype(str).str.upper().isin(("P", "B"))]
    elif "side" in df.columns:
        df = df[df["side"].astype(str).str.lower().str.startswith("buy")]

    df = df.dropna(subset=["filed_at"]).sort_values("filed_at")
    out = pd.Series(0.0, index=idx, dtype="float64")
    if df.empty:
        return out

    window = pd.Timedelta(days=lookback_d)
    for t in idx:
        sub = df[(df["filed_at"] <= t) & (df["filed_at"] > (t - window))]
        if sub.empty:
            continue
        n_insider = (sub[insider_col].nunique() if insider_col else len(sub))
        if value_col and value_col in sub.columns:
            total_val = pd.to_numeric(sub[value_col], errors="coerce").fillna(0.0).sum()
        else:
            total_val = float(len(sub))
        out.loc[t] = float(n_insider) * float(np.log1p(max(total_val, 0.0)))
    return out


def form4_insider_cluster_score(ticker: str, bar_timestamps: pd.DatetimeIndex,
                                 lookback_d: int = 5) -> pd.Series:
    """For each bar_timestamp t: cluster_score = n_distinct_insiders * log(1 + total_value)
    from Form 4 BUY rows accepted in (t - lookback_d, t]. Causal via `filed_at <= t`.

    Returns same-length Series, 0 where no Form 4 activity, NaN if source missing entirely.
    axis=volume_conviction (insider conviction confirmation).

    Vectorized (2026-05-29, task #78): replaces O(N*M) per-bar loop with O((N+M) log M)
    searchsorted + per-interval distinct count. The distinct-insider count cannot be
    pure cumsum (set cardinality is not additive), but each per-bar set computation is
    bounded to the events inside the lookback window — typically <50 rows even for
    very active tickers — so the inner loop is small and cache-friendly.
    """
    idx = _ensure_naive_index(bar_timestamps)
    df = _fetch_filings(ticker, form="Form 4")
    if df.empty or "filed_at" not in df.columns:
        return _empty_series(idx)

    value_col = None
    for c in ("value_usd", "transaction_value", "amount_usd", "amount"):
        if c in df.columns:
            value_col = c
            break
    insider_col = None
    for c in ("reporting_owner", "insider_name", "reporter_name", "filer"):
        if c in df.columns:
            insider_col = c
            break
    if "transaction_code" in df.columns:
        df = df[df["transaction_code"].astype(str).str.upper().isin(("P", "B"))]
    elif "side" in df.columns:
        df = df[df["side"].astype(str).str.lower().str.startswith("buy")]

    df = df.dropna(subset=["filed_at"]).sort_values("filed_at")
    out = pd.Series(0.0, index=idx, dtype="float64")
    if df.empty:
        return out

    # Sorted event timestamps + paired arrays
    event_times = df["filed_at"].values.astype("datetime64[ns]")
    if value_col and value_col in df.columns:
        event_values = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    else:
        event_values = None  # signal: use count instead
    if insider_col:
        # Map insider strings to int codes for fast unique count via np.unique
        codes, _ = pd.factorize(df[insider_col].astype(str), sort=False)
        insider_codes = codes.astype(np.int64)
    else:
        insider_codes = None

    bar_arr = idx.values.astype("datetime64[ns]")
    window_ns = np.timedelta64(int(lookback_d), "D")
    # upper: events with filed_at <= t   (side='right')
    upper = np.searchsorted(event_times, bar_arr, side="right")
    # lower: events with filed_at > t - window  (side='right' on (t-window))
    lower = np.searchsorted(event_times, bar_arr - window_ns, side="right")

    # Cumsum trick for SUM aggregation (or count fallback)
    if event_values is not None:
        cum_val = np.concatenate(([0.0], np.cumsum(event_values)))
        sum_window = cum_val[upper] - cum_val[lower]
    else:
        sum_window = (upper - lower).astype("float64")  # count fallback

    # Per-bar distinct-insider count: small inner loop bounded by lookback window
    if insider_codes is not None:
        n_insider = np.zeros(len(bar_arr), dtype="float64")
        # Skip bars with zero window-events
        nonempty = np.flatnonzero(upper > lower)
        for i in nonempty:
            lo, hi = lower[i], upper[i]
            # Distinct count on a small slice — uses optimized C-level np.unique
            n_insider[i] = np.unique(insider_codes[lo:hi]).size
    else:
        # No insider column → use raw event count as proxy
        n_insider = (upper - lower).astype("float64")

    # Clamp sum_window to >=0 then apply log1p
    sum_window = np.maximum(sum_window, 0.0)
    out_arr = n_insider * np.log1p(sum_window)
    # Where no events at all, keep 0.0 (matches loop version's `continue` semantics)
    out = pd.Series(out_arr, index=idx, dtype="float64")
    return out


# ---------------------------------------------------------------------------
# 2. Congress lead-lag (days since most recent disclosure)
# ---------------------------------------------------------------------------


def _congress_lead_lag_loop_v1(ticker: str, bar_timestamps: pd.DatetimeIndex,
                                lookback_d: int = 30) -> pd.Series:
    """Original O(N*M) reference — kept for equality tests."""
    idx = _ensure_naive_index(bar_timestamps)
    df = _fetch_congress(ticker)
    if df.empty or "disclosure_date" not in df.columns:
        return _empty_series(idx)
    df = df.dropna(subset=["disclosure_date"]).sort_values("disclosure_date")
    if df.empty:
        return _empty_series(idx)
    out = pd.Series(np.nan, index=idx, dtype="float64")
    window = pd.Timedelta(days=lookback_d)
    for t in idx:
        sub = df[(df["disclosure_date"] <= t) & (df["disclosure_date"] > (t - window))]
        if not sub.empty:
            most_recent = sub["disclosure_date"].max()
            out.loc[t] = float((t - most_recent).days)
    return out


def congress_lead_lag(ticker: str, bar_timestamps: pd.DatetimeIndex,
                       lookback_d: int = 30) -> pd.Series:
    """Days since the latest congress trade disclosure for `ticker`, capped at `lookback_d`.
    NaN if no disclosure within lookback. Causal via `disclosure_date <= t`.
    axis=volume_conviction (informed-flow confirmation).

    Vectorized (2026-05-29, task #78): "days since latest event <= t (within window)"
    reduces to a single searchsorted — pick the event at index upper-1 if upper > lower.
    Events are pre-sorted so events[upper-1] is the most recent <= t.
    """
    idx = _ensure_naive_index(bar_timestamps)
    df = _fetch_congress(ticker)
    if df.empty or "disclosure_date" not in df.columns:
        return _empty_series(idx)
    df = df.dropna(subset=["disclosure_date"]).sort_values("disclosure_date")
    if df.empty:
        return _empty_series(idx)

    event_times = df["disclosure_date"].values.astype("datetime64[ns]")
    bar_arr = idx.values.astype("datetime64[ns]")
    window_ns = np.timedelta64(int(lookback_d), "D")
    upper = np.searchsorted(event_times, bar_arr, side="right")
    lower = np.searchsorted(event_times, bar_arr - window_ns, side="right")

    out_arr = np.full(len(bar_arr), np.nan, dtype="float64")
    mask = upper > lower  # at least one event in window
    if mask.any():
        most_recent_idx = (upper[mask] - 1).astype(np.int64)
        most_recent = event_times[most_recent_idx]
        # (t - most_recent).days — matches pandas Timedelta.days (integer floor of total days)
        delta = bar_arr[mask] - most_recent
        # Convert numpy timedelta64[ns] → integer days (floor toward -inf to match pandas)
        days = (delta / np.timedelta64(1, "D")).astype(np.float64)
        # Pandas Timedelta.days uses floor div for negative, but here delta>=0 so floor == int
        out_arr[mask] = np.floor(days)
    return pd.Series(out_arr, index=idx, dtype="float64")


# ---------------------------------------------------------------------------
# 3. News velocity Z-score
# ---------------------------------------------------------------------------


def _news_velocity_zscore_loop_v1(ticker: str, bar_timestamps: pd.DatetimeIndex,
                                   lookback_d: int = 7, baseline_d: int = 90) -> pd.Series:
    """Original O(N*M) reference — kept for equality tests."""
    idx = _ensure_naive_index(bar_timestamps)
    if len(idx) == 0:
        return _empty_series(idx)
    start = idx.min() - pd.Timedelta(days=baseline_d + lookback_d + 1)
    end = idx.max()
    df = _fetch_news(ticker, start, end)
    if df.empty or "published_utc" not in df.columns:
        return _empty_series(idx)
    df = df.dropna(subset=["published_utc"]).sort_values("published_utc")
    if df.empty:
        return _empty_series(idx)
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    daily = df.assign(d=df["published_utc"].dt.normalize()).groupby("d").size().reindex(days).fillna(0)
    rolling_recent = daily.rolling(lookback_d, min_periods=1).sum()
    rolling_mean = rolling_recent.rolling(baseline_d, min_periods=lookback_d).mean()
    rolling_std = rolling_recent.rolling(baseline_d, min_periods=lookback_d).std(ddof=0)
    out = pd.Series(np.nan, index=idx, dtype="float64")
    daily_index = rolling_recent.index
    for t in idx:
        d = pd.Timestamp(t).normalize()
        if d < daily_index[0] or d > daily_index[-1]:
            continue
        recent = rolling_recent.loc[d]
        mean = rolling_mean.loc[d]
        std = rolling_std.loc[d]
        if pd.notna(std) and std > 0:
            out.loc[t] = float((recent - mean) / std)
        elif pd.notna(mean):
            out.loc[t] = 0.0
    return out


def news_velocity_zscore(ticker: str, bar_timestamps: pd.DatetimeIndex,
                          lookback_d: int = 7, baseline_d: int = 90) -> pd.Series:
    """Rolling Z-score of `lookback_d`-day news count vs `baseline_d`-day trailing baseline.

    For each bar_timestamp t:
      n_recent = count(news.published_utc in (t - lookback_d, t])
      baseline = mean of `lookback_d`-counts over the past `baseline_d` days
      std = std of those counts
      z = (n_recent - baseline) / std

    Causal via `published_utc <= t`. axis=volume_conviction (catalyst flow).

    Vectorized (2026-05-29, task #78): daily rolling stats were already vectorized,
    but the per-bar lookup loop was O(M log K). Replace it with a single .reindex()
    that broadcasts daily values to bar_timestamps in one shot.
    """
    idx = _ensure_naive_index(bar_timestamps)
    if len(idx) == 0:
        return _empty_series(idx)
    start = idx.min() - pd.Timedelta(days=baseline_d + lookback_d + 1)
    end = idx.max()
    df = _fetch_news(ticker, start, end)
    if df.empty or "published_utc" not in df.columns:
        return _empty_series(idx)
    df = df.dropna(subset=["published_utc"]).sort_values("published_utc")
    if df.empty:
        return _empty_series(idx)
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    daily = df.assign(d=df["published_utc"].dt.normalize()).groupby("d").size().reindex(days).fillna(0)
    rolling_recent = daily.rolling(lookback_d, min_periods=1).sum()
    rolling_mean = rolling_recent.rolling(baseline_d, min_periods=lookback_d).mean()
    rolling_std = rolling_recent.rolling(baseline_d, min_periods=lookback_d).std(ddof=0)

    # Vectorized lookup: normalize each bar timestamp to its day then reindex against the daily Series
    bar_days = idx.normalize()
    recent_v = rolling_recent.reindex(bar_days).to_numpy()
    mean_v = rolling_mean.reindex(bar_days).to_numpy()
    std_v = rolling_std.reindex(bar_days).to_numpy()

    out_arr = np.full(len(idx), np.nan, dtype="float64")
    has_data = ~np.isnan(recent_v)  # bar day within daily range
    std_pos = np.isfinite(std_v) & (std_v > 0)
    mean_ok = np.isfinite(mean_v)
    # Where std > 0: z = (recent - mean) / std
    z_mask = has_data & std_pos
    out_arr[z_mask] = (recent_v[z_mask] - mean_v[z_mask]) / std_v[z_mask]
    # Where std is 0/nan but mean is finite: 0.0 (constant baseline)
    zero_mask = has_data & ~std_pos & mean_ok
    out_arr[zero_mask] = 0.0
    return pd.Series(out_arr, index=idx, dtype="float64")


# ---------------------------------------------------------------------------
# 4. Dark pool divergence Z-score
# ---------------------------------------------------------------------------


def _dark_pool_divergence_z_loop_v1(ticker: str, bar_timestamps: pd.DatetimeIndex,
                                     lookback_d: int = 5, baseline_d: int = 30) -> pd.Series:
    """Original O(N*M) reference — kept for equality tests."""
    idx = _ensure_naive_index(bar_timestamps)
    df = _fetch_offex(ticker)
    if df.empty or "as_of_date" not in df.columns:
        return _empty_series(idx)
    df = df.dropna(subset=["as_of_date"]).sort_values("as_of_date").set_index("as_of_date")
    metric: Optional[pd.Series] = None
    for c in ("short_volume_ratio", "dpi", "dark_pool_index"):
        if c in df.columns:
            metric = pd.to_numeric(df[c], errors="coerce")
            break
    if metric is None and "short_volume" in df.columns and "total_volume" in df.columns:
        sv = pd.to_numeric(df["short_volume"], errors="coerce")
        tv = pd.to_numeric(df["total_volume"], errors="coerce")
        metric = sv / tv.replace(0.0, np.nan)
    if metric is None:
        return _empty_series(idx)
    metric = metric.sort_index()
    rolling = metric.rolling(lookback_d, min_periods=1).mean()
    base_mean = metric.rolling(baseline_d, min_periods=lookback_d).mean()
    base_std = metric.rolling(baseline_d, min_periods=lookback_d).std(ddof=0)
    out = pd.Series(np.nan, index=idx, dtype="float64")
    for t in idx:
        sub = rolling.loc[:t]
        if sub.empty:
            continue
        recent = sub.iloc[-1]
        if pd.isna(recent):
            continue
        sub_b_mean = base_mean.loc[:t]
        sub_b_std = base_std.loc[:t]
        if sub_b_mean.empty or sub_b_std.empty:
            continue
        b_mean = sub_b_mean.iloc[-1]
        b_std = sub_b_std.iloc[-1]
        if pd.notna(b_std) and b_std > 0:
            out.loc[t] = float((recent - b_mean) / b_std)
    return out


def dark_pool_divergence_z(ticker: str, bar_timestamps: pd.DatetimeIndex,
                            lookback_d: int = 5, baseline_d: int = 30) -> pd.Series:
    """Z-score of offexchange short-volume ratio over `lookback_d` vs `baseline_d` baseline.

    Uses 'short_volume_ratio' or 'dpi' column if available; falls back to a derived ratio
    if raw volume + short_volume present.

    Causal via `as_of_date <= t`. axis=volume_conviction (institutional positioning).

    Vectorized (2026-05-29, task #78): "most recent rolling stat <= t" reduces to
    searchsorted then index — replace per-bar loop with a single vectorized lookup.
    """
    idx = _ensure_naive_index(bar_timestamps)
    df = _fetch_offex(ticker)
    if df.empty or "as_of_date" not in df.columns:
        return _empty_series(idx)
    df = df.dropna(subset=["as_of_date"]).sort_values("as_of_date").set_index("as_of_date")
    metric: Optional[pd.Series] = None
    for c in ("short_volume_ratio", "dpi", "dark_pool_index"):
        if c in df.columns:
            metric = pd.to_numeric(df[c], errors="coerce")
            break
    if metric is None and "short_volume" in df.columns and "total_volume" in df.columns:
        sv = pd.to_numeric(df["short_volume"], errors="coerce")
        tv = pd.to_numeric(df["total_volume"], errors="coerce")
        metric = sv / tv.replace(0.0, np.nan)
    if metric is None:
        return _empty_series(idx)

    metric = metric.sort_index()
    rolling = metric.rolling(lookback_d, min_periods=1).mean()
    base_mean = metric.rolling(baseline_d, min_periods=lookback_d).mean()
    base_std = metric.rolling(baseline_d, min_periods=lookback_d).std(ddof=0)

    # Vectorized "as-of" lookup: for each bar t, find index of most recent metric row <= t
    metric_times = rolling.index.values.astype("datetime64[ns]")
    bar_arr = idx.values.astype("datetime64[ns]")
    pos = np.searchsorted(metric_times, bar_arr, side="right") - 1
    valid = pos >= 0

    recent_arr = np.full(len(bar_arr), np.nan, dtype="float64")
    bmean_arr = np.full(len(bar_arr), np.nan, dtype="float64")
    bstd_arr = np.full(len(bar_arr), np.nan, dtype="float64")
    if valid.any():
        idx_v = pos[valid].astype(np.int64)
        recent_arr[valid] = rolling.values[idx_v]
        bmean_arr[valid] = base_mean.values[idx_v]
        bstd_arr[valid] = base_std.values[idx_v]

    out_arr = np.full(len(bar_arr), np.nan, dtype="float64")
    ok = np.isfinite(recent_arr) & np.isfinite(bmean_arr) & np.isfinite(bstd_arr) & (bstd_arr > 0)
    out_arr[ok] = (recent_arr[ok] - bmean_arr[ok]) / bstd_arr[ok]
    return pd.Series(out_arr, index=idx, dtype="float64")


# ---------------------------------------------------------------------------
# 5. Lobbying intensity
# ---------------------------------------------------------------------------


def _lobbying_intensity_loop_v1(ticker: str, bar_timestamps: pd.DatetimeIndex,
                                 lookback_d: int = 90, baseline_d: int = 365) -> pd.Series:
    """Original O(N*M) reference — kept for equality tests."""
    idx = _ensure_naive_index(bar_timestamps)
    df = _fetch_lobbying(ticker)
    if df.empty or "period_end" not in df.columns:
        return _empty_series(idx)
    amt_col = None
    for c in ("amount_usd", "amount", "spend", "total_spending"):
        if c in df.columns:
            amt_col = c
            break
    if amt_col is None:
        return _empty_series(idx)
    df = df.dropna(subset=["period_end"]).copy()
    df["amount"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)
    df = df.sort_values("period_end")
    out = pd.Series(np.nan, index=idx, dtype="float64")
    w = pd.Timedelta(days=lookback_d)
    b = pd.Timedelta(days=baseline_d)
    for t in idx:
        recent = df[(df["period_end"] <= t) & (df["period_end"] > (t - w))]["amount"].sum()
        base = df[(df["period_end"] <= t) & (df["period_end"] > (t - b))]["amount"].sum()
        base_per_day = base / max(baseline_d, 1)
        if base_per_day > 0:
            out.loc[t] = float(recent / lookback_d) / float(base_per_day)
    return out


def lobbying_intensity(ticker: str, bar_timestamps: pd.DatetimeIndex,
                        lookback_d: int = 90, baseline_d: int = 365) -> pd.Series:
    """90d lobbying $ normalized by trailing-year ($/day) average. Causal via `period_end <= t`.
    axis=volume_conviction (policy-driver context).

    Vectorized (2026-05-29, task #78): both window-sums computed via cumsum +
    searchsorted (lookback and baseline). The two-interval window-sum trick is the
    canonical vectorization for rolling-window event aggregations.
    """
    idx = _ensure_naive_index(bar_timestamps)
    df = _fetch_lobbying(ticker)
    if df.empty or "period_end" not in df.columns:
        return _empty_series(idx)
    amt_col = None
    for c in ("amount_usd", "amount", "spend", "total_spending"):
        if c in df.columns:
            amt_col = c
            break
    if amt_col is None:
        return _empty_series(idx)
    df = df.dropna(subset=["period_end"]).copy()
    df["amount"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)
    df = df.sort_values("period_end")

    event_times = df["period_end"].values.astype("datetime64[ns]")
    event_amounts = df["amount"].to_numpy(dtype="float64")
    cum_amt = np.concatenate(([0.0], np.cumsum(event_amounts)))

    bar_arr = idx.values.astype("datetime64[ns]")
    w_ns = np.timedelta64(int(lookback_d), "D")
    b_ns = np.timedelta64(int(baseline_d), "D")
    upper = np.searchsorted(event_times, bar_arr, side="right")
    lower_w = np.searchsorted(event_times, bar_arr - w_ns, side="right")
    lower_b = np.searchsorted(event_times, bar_arr - b_ns, side="right")

    recent = cum_amt[upper] - cum_amt[lower_w]
    base = cum_amt[upper] - cum_amt[lower_b]
    base_per_day = base / max(baseline_d, 1)

    out_arr = np.full(len(bar_arr), np.nan, dtype="float64")
    ok = base_per_day > 0
    out_arr[ok] = (recent[ok] / lookback_d) / base_per_day[ok]
    return pd.Series(out_arr, index=idx, dtype="float64")


# ---------------------------------------------------------------------------
# 6. Gov contract inflow
# ---------------------------------------------------------------------------


def _gov_contract_inflow_loop_v1(ticker: str, bar_timestamps: pd.DatetimeIndex,
                                  lookback_d: int = 90) -> pd.Series:
    """Original O(N*M) reference — kept for equality tests."""
    idx = _ensure_naive_index(bar_timestamps)
    df = _fetch_contracts(ticker)
    if df.empty or "awarded_at" not in df.columns:
        return _empty_series(idx)
    amt_col = None
    for c in ("amount_usd", "obligated_amount", "amount", "award_amount", "value"):
        if c in df.columns:
            amt_col = c
            break
    df = df.dropna(subset=["awarded_at"]).copy()
    if amt_col:
        df["amount"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)
    else:
        df["amount"] = 1.0
    df = df.sort_values("awarded_at")
    out = pd.Series(np.nan, index=idx, dtype="float64")
    w = pd.Timedelta(days=lookback_d)
    for t in idx:
        sub = df[(df["awarded_at"] <= t) & (df["awarded_at"] > (t - w))]
        out.loc[t] = float(sub["amount"].sum())
    return out


def gov_contract_inflow(ticker: str, bar_timestamps: pd.DatetimeIndex,
                         lookback_d: int = 90) -> pd.Series:
    """Total gov contract $ awarded in last `lookback_d` days. Causal via `awarded_at <= t`.
    axis=volume_conviction (revenue-driver inflow).

    Vectorized (2026-05-29, task #78): cumsum + searchsorted dual-interval rolling sum.
    """
    idx = _ensure_naive_index(bar_timestamps)
    df = _fetch_contracts(ticker)
    if df.empty or "awarded_at" not in df.columns:
        return _empty_series(idx)
    amt_col = None
    for c in ("amount_usd", "obligated_amount", "amount", "award_amount", "value"):
        if c in df.columns:
            amt_col = c
            break
    df = df.dropna(subset=["awarded_at"]).copy()
    if amt_col:
        df["amount"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)
    else:
        df["amount"] = 1.0
    df = df.sort_values("awarded_at")

    event_times = df["awarded_at"].values.astype("datetime64[ns]")
    event_amounts = df["amount"].to_numpy(dtype="float64")
    cum_amt = np.concatenate(([0.0], np.cumsum(event_amounts)))

    bar_arr = idx.values.astype("datetime64[ns]")
    w_ns = np.timedelta64(int(lookback_d), "D")
    upper = np.searchsorted(event_times, bar_arr, side="right")
    lower = np.searchsorted(event_times, bar_arr - w_ns, side="right")
    sum_window = cum_amt[upper] - cum_amt[lower]
    return pd.Series(sum_window, index=idx, dtype="float64")


# ---------------------------------------------------------------------------
# 7. 8-K pulse
# ---------------------------------------------------------------------------


def _eight_k_pulse_loop_v1(ticker: str, bar_timestamps: pd.DatetimeIndex,
                            lookback_d: int = 5) -> pd.Series:
    """Original O(N*M) reference — kept for equality tests."""
    idx = _ensure_naive_index(bar_timestamps)
    df = _fetch_filings(ticker, form="8-K")
    if df.empty or "filed_at" not in df.columns:
        return _empty_series(idx).fillna(0.0)
    df = df.dropna(subset=["filed_at"]).sort_values("filed_at")
    out = pd.Series(0.0, index=idx, dtype="float64")
    w = pd.Timedelta(days=lookback_d)
    for t in idx:
        sub = df[(df["filed_at"] <= t) & (df["filed_at"] > (t - w))]
        out.loc[t] = float(len(sub))
    return out


def eight_k_pulse(ticker: str, bar_timestamps: pd.DatetimeIndex,
                   lookback_d: int = 5) -> pd.Series:
    """Count of 8-K filings in last `lookback_d` days (material-event catalyst pulse).
    Causal via `filed_at <= t`. axis=structure_geometry (event anchor).

    Vectorized (2026-05-29, task #78): pure count via searchsorted diff (cumsum unneeded).
    """
    idx = _ensure_naive_index(bar_timestamps)
    df = _fetch_filings(ticker, form="8-K")
    if df.empty or "filed_at" not in df.columns:
        return _empty_series(idx).fillna(0.0)
    df = df.dropna(subset=["filed_at"]).sort_values("filed_at")
    if df.empty:
        return pd.Series(0.0, index=idx, dtype="float64")

    event_times = df["filed_at"].values.astype("datetime64[ns]")
    bar_arr = idx.values.astype("datetime64[ns]")
    w_ns = np.timedelta64(int(lookback_d), "D")
    upper = np.searchsorted(event_times, bar_arr, side="right")
    lower = np.searchsorted(event_times, bar_arr - w_ns, side="right")
    counts = (upper - lower).astype("float64")
    return pd.Series(counts, index=idx, dtype="float64")


# alias matching task spec naming
form_8k_pulse = eight_k_pulse


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


ALTDATA_AXIS: dict[str, str] = {
    "form4_insider_cluster_score": "volume_conviction",
    "congress_lead_lag": "volume_conviction",
    "news_velocity_zscore": "volume_conviction",
    "dark_pool_divergence_z": "volume_conviction",
    "lobbying_intensity": "volume_conviction",
    "gov_contract_inflow": "volume_conviction",
    "eight_k_pulse": "structure_geometry",
    "form_8k_pulse": "structure_geometry",
}


ALTDATA_REGISTRY: dict[str, dict] = {
    "form4_insider_cluster_score": {"fn": form4_insider_cluster_score, "source": "edgar.Form 4"},
    "congress_lead_lag": {"fn": congress_lead_lag, "source": "govtrades.congress_trades"},
    "news_velocity_zscore": {"fn": news_velocity_zscore, "source": "news.articles"},
    "dark_pool_divergence_z": {"fn": dark_pool_divergence_z, "source": "govtrades.offexchange"},
    "lobbying_intensity": {"fn": lobbying_intensity, "source": "govtrades.lobbying"},
    "gov_contract_inflow": {"fn": gov_contract_inflow, "source": "govtrades.gov_contracts"},
    "eight_k_pulse": {"fn": eight_k_pulse, "source": "edgar.8-K"},
}


def altdata_axis_for(name: str) -> str:
    return ALTDATA_AXIS.get(name, "unknown")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_bar_timestamps(n: int = 90, end: str = "2026-05-27") -> pd.DatetimeIndex:
    return pd.date_range(end=end, periods=n, freq="B")


if __name__ == "__main__":
    ticker = "AAPL"
    bts = _smoke_bar_timestamps(90)
    print(f"# altdata smoke — ticker={ticker}, bars={len(bts)} ({bts[0].date()} → {bts[-1].date()})")
    print(f"# total altdata indicators: {len(ALTDATA_REGISTRY)}")
    print()
    print(f"{'STATUS':<8}{'AXIS':<22}{'NAME':<32}{'NOTES'}")

    ok = fail = empty = 0
    for name, cfg in ALTDATA_REGISTRY.items():
        try:
            ser = cfg["fn"](ticker, bts)
            ax = ALTDATA_AXIS.get(name, "?")
            arr = ser.to_numpy() if isinstance(ser, pd.Series) else np.asarray(ser)
            n_finite = int(np.sum(np.isfinite(arr)))
            n_nonzero = int(np.sum(np.isfinite(arr) & (arr != 0)))
            if len(arr) != len(bts):
                print(f"{'FAIL':<8}{ax:<22}{name:<32}shape {len(arr)} != {len(bts)}")
                fail += 1
                continue
            if n_finite == 0:
                # All-NaN means source missing — degrade but not a code failure
                print(f"{'EMPTY':<8}{ax:<22}{name:<32}all-NaN (source: {cfg['source']} unavailable)")
                empty += 1
                continue
            finite = arr[np.isfinite(arr)]
            rng = f"[{finite.min():.3g}, {finite.max():.3g}]"
            print(f"{'OK':<8}{ax:<22}{name:<32}finite={n_finite}/{len(bts)} non0={n_nonzero} {rng}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"{'FAIL':<8}{ALTDATA_AXIS.get(name,'?'):<22}{name:<32}EXC {type(e).__name__}: {e}")
            fail += 1
    print()
    print(f"# altdata smoke: {ok} ok / {empty} empty(no-data) / {fail} fail / {len(ALTDATA_REGISTRY)} total")
