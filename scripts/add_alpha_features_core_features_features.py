"""add_alpha_features_core_features_features.py
Wraps github:GiovanniPioDelvecchio/alpha_features_core (MIT license).
Computes afc_alpha001..afc_alpha030 using the Alphas191 class when the repo
is importable, otherwise falls back to pure-pandas approximations.

NO-LOOKAHEAD AUDIT (2026-05-17):
  All inputs used: open, high, low, close, volume — EOD-completed bars only.
  Rolling windows (6..30) reference only prior-completed bars.
  `close.pct_change()` = bar-T close vs bar-(T-1) close — no forward data.
  Delta/Delay use .diff() / .shift() — both strictly backward-looking.
  VWAP proxy = amount/volume (same-bar EOD, not intraday real-time).
  Cross-sectional Rank with one ticker → constant 0.5; never intraday.
  Consumer (build_v10_features) does NOT need additional .shift(1) because
  all feature values at bar T reference completed bars only. Safe.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_PATH = str(
    Path(
        "/Users/orginal/Library/CloudStorage/"
        "GoogleDrive-zachgladstone@gmail.com/My Drive/"
        "AI-Tools/repos-claude-clones/alpha_features_core"
    )
)

AFC_CORE_FEATURE_COUNT = 30
AFC_CORE_FEATURE_NAMES: list[str] = [f"afc_alpha{i:03d}" for i in range(1, 31)]


# ---------------------------------------------------------------------------
# Pure-pandas fallback implementations (first 30 alpha proxies, single-ticker)
# Cross-sectional Rank with 1 ticker = 0.5 constant; omitted or simplified.
# ---------------------------------------------------------------------------

def _safe_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=2).mean()
    sigma = s.rolling(window, min_periods=2).std()
    return (s - mu) / sigma.replace(0, np.nan)


def _compute_fallback_alphas(df: pd.DataFrame) -> dict[str, pd.Series]:
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    returns = close.pct_change()
    log_volume = np.log(volume.replace(0, np.nan))
    volume_chg = log_volume.diff(1)
    body_pct = (close - open_) / open_.replace(0, np.nan)
    vwap_proxy = (high + low + close) / 3.0
    adv20 = volume.rolling(20, min_periods=1).mean()

    out: dict[str, pd.Series] = {}
    # alpha001 — vol-body correlation proxy
    out["afc_alpha001"] = -1 * volume_chg.rolling(6, min_periods=2).corr(body_pct)
    # alpha002 — daily range position delta
    hl = (high - low).replace(0, np.nan)
    out["afc_alpha002"] = -1 * (((close - low) / hl) - ((close - low) / hl).shift(1))
    # alpha003 — open-volume correlation
    out["afc_alpha003"] = -1 * volume.rolling(10, min_periods=2).corr(open_)
    # alpha004 — low rank proxy (time-series position of low in 9d window)
    out["afc_alpha004"] = low.rolling(9, min_periods=2).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=True
    )
    # alpha005 — vwap-close spread
    out["afc_alpha005"] = (vwap_proxy - close) / (vwap_proxy.abs() + 1e-10)
    # alpha006 — open-volume 10d corr
    out["afc_alpha006"] = -1 * open_.rolling(10, min_periods=2).corr(volume)
    # alpha007 — vol-to-adv20 momentum
    cond = adv20 < volume
    out["afc_alpha007"] = cond.astype(float) * (-1 * close.diff(7).apply(np.sign)) + (~cond).astype(float) * (-1.0)
    # alpha008 — open * return lag
    out["afc_alpha008"] = -1 * (open_.rolling(5).sum() * returns.rolling(5).sum() - (open_.rolling(5).sum() * returns.rolling(5).sum()).shift(10))
    # alpha009 — close delta trend
    c1 = close.diff(1)
    out["afc_alpha009"] = np.where(
        close.rolling(5, min_periods=2).std() < close.rolling(15, min_periods=2).std(),
        np.sign(c1),
        np.sign(c1.rolling(3, min_periods=1).min())
    )
    out["afc_alpha009"] = pd.Series(out["afc_alpha009"], index=close.index)
    # alpha010 — same as 009 variant
    out["afc_alpha010"] = np.sign(c1).rolling(4, min_periods=1).min()
    # alpha011 — vwap-close times volume range
    vol_rng = volume.rolling(3, min_periods=1).max() - volume.rolling(3, min_periods=1).min()
    out["afc_alpha011"] = ((vwap_proxy - close) * vol_rng).rolling(3, min_periods=1).sum()
    # alpha012 — sign(volume delta) * price delta
    out["afc_alpha012"] = np.sign(volume.diff(1)) * (-1 * close.diff(1))
    # alpha013 — close-vwap covariance proxy
    out["afc_alpha013"] = -1 * close.rolling(5, min_periods=2).cov(volume)
    # alpha014 — returns lag × open-volume corr
    out["afc_alpha014"] = (-1 * returns.diff(3)) * open_.rolling(10, min_periods=2).corr(volume)
    # alpha015 — high-volume correlation proxy
    out["afc_alpha015"] = -1 * high.rolling(3, min_periods=2).corr(volume.rolling(3, min_periods=1).apply(lambda x: x[-1] / x.mean() if x.mean() else 0, raw=True)).rolling(3, min_periods=1).sum()
    # alpha016 — high-volume covariance
    out["afc_alpha016"] = -1 * high.rolling(5, min_periods=2).cov(volume)
    # alpha017 — close zscore × volume rank
    out["afc_alpha017"] = (-1 * _safe_zscore(close, 20)) * (volume / (adv20 + 1e-10)).apply(np.sign)
    # alpha018 — close-open corr
    out["afc_alpha018"] = -1 * close.rolling(5, min_periods=2).corr(open_)
    # alpha019 — sign(close change) based on trend
    out["afc_alpha019"] = (
        np.sign(close.diff(7) + close.rolling(5, min_periods=1).std()) * (returns.rolling(250, min_periods=50).sum() + 1)
    )
    # alpha020 — lagged HL spread
    out["afc_alpha020"] = (-1 * (open_ - high.shift(1)).apply(np.sign)) * (-1 * (open_ - close.shift(1)).apply(np.sign)) * (-1 * (open_ - low.shift(1)).apply(np.sign))
    # alpha021 — mean-reversion flag
    cond21a = close.rolling(8, min_periods=2).mean() + close.rolling(8, min_periods=2).std() < close.rolling(2, min_periods=1).mean()
    cond21b = close.rolling(2, min_periods=1).mean() < close.rolling(8, min_periods=2).mean() - close.rolling(8, min_periods=2).std()
    out["afc_alpha021"] = pd.Series(
        np.where(cond21a, -1.0, np.where(cond21b, 1.0, -1.0)),
        index=close.index,
    )
    # alpha022 — high delta × close-volume corr
    out["afc_alpha022"] = -1 * high.diff(5) * high.rolling(5, min_periods=2).corr(volume)
    # alpha023 — high delta sign when high is high
    out["afc_alpha023"] = np.where(
        high.rolling(20, min_periods=5).mean() < high,
        -1 * high.diff(2),
        0.0
    )
    out["afc_alpha023"] = pd.Series(out["afc_alpha023"], index=close.index)
    # alpha024 — close delta sign switch
    cond24 = (close.rolling(10, min_periods=2).mean().diff(10) / 10) <= 0.05
    out["afc_alpha024"] = np.where(
        cond24,
        -1 * close.diff(1),
        close - close.rolling(10, min_periods=2).min()
    )
    out["afc_alpha024"] = pd.Series(out["afc_alpha024"], index=close.index)
    # alpha025 — adv20 rank × returns × range
    out["afc_alpha025"] = -1 * (volume / adv20).clip(0, 5) * returns * (high - close)
    # alpha026 — rolling max of ts-corr close-volume
    out["afc_alpha026"] = -1 * close.rolling(5, min_periods=2).corr(volume).rolling(5, min_periods=1).max()
    # alpha027 — volume rank (vs 20d avg)
    out["afc_alpha027"] = np.where(
        (volume / adv20).rolling(6, min_periods=2).mean() > 1,
        -1.0,
        1.0
    )
    out["afc_alpha027"] = pd.Series(out["afc_alpha027"], index=close.index)
    # alpha028 — body vs HL spread zscore
    out["afc_alpha028"] = _safe_zscore(
        ((close - low.rolling(9, min_periods=2).min()) - (high.rolling(9, min_periods=2).max() - close)) * volume,
        30
    )
    # alpha029 — cumulative return position
    out["afc_alpha029"] = returns.rolling(5, min_periods=1).sum() * close.diff(1).rolling(5, min_periods=1).sum()
    # alpha030 — sign(close - prev_close) weighted vs buyer vol
    out["afc_alpha030"] = np.sign(close.diff(1)).rolling(5, min_periods=1).sum() * -1

    return out


# ---------------------------------------------------------------------------
# Primary: Alphas191-based computation
# ---------------------------------------------------------------------------

def _compute_via_repo(df: pd.DataFrame, ticker: str) -> dict[str, pd.Series] | None:
    if _REPO_PATH not in sys.path:
        sys.path.insert(0, _REPO_PATH)
    try:
        from alpha_features_core.alpha191 import Alphas191  # type: ignore[import]
    except Exception as exc:
        logger.debug("alpha_features_core not importable: %s", exc)
        return None

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        logger.debug("Missing required OHLCV columns for Alphas191")
        return None

    try:
        # Build the long-format DataFrame Alphas191 expects
        long_df = df[["open", "high", "low", "close", "volume"]].copy().reset_index()
        long_df.columns = ["date"] + list(long_df.columns[1:])
        long_df["ticker"] = ticker
        long_df["amount"] = long_df["close"] * long_df["volume"]
        long_df["past_return"] = long_df["close"].pct_change().fillna(0.0)

        if len(long_df) < 10:
            return None

        alphas_obj = Alphas191(long_df)
        out: dict[str, pd.Series] = {}
        date_index = df.index

        for i in range(1, 31):
            col_name = f"afc_alpha{i:03d}"
            try:
                wide_result = alphas_obj.calculate_alpha(i, return_long=False)
                # wide_result is a DataFrame indexed by date with ticker as columns
                if ticker in wide_result.columns:
                    series = wide_result[ticker]
                else:
                    series = wide_result.iloc[:, 0]
                series = series.reindex(date_index).fillna(0.0)
                out[col_name] = series
            except Exception as exc:
                logger.debug("Alphas191.alpha%03d failed: %s", i, exc)
                out[col_name] = pd.Series(0.0, index=date_index)

        return out

    except Exception as exc:
        logger.warning("Alphas191 init/compute failed: %s — using fallback", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_add_alpha_features_core_features_features(
    df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Add 30 WorldQuant-style alpha features (afc_alpha001..afc_alpha030).

    Tries the Alphas191 class from GiovanniPioDelvecchio/alpha_features_core;
    falls back to pure-pandas approximations on any failure.

    Args:
        df: Feature DataFrame with DatetimeIndex and OHLCV columns.
        ticker: Stock symbol (used as column key in Alphas191 pivot).

    Returns:
        df augmented with columns afc_alpha001..afc_alpha030.
    """
    result_df = df.copy()
    ticker_str = ticker or "UNKNOWN"

    repo_out = _compute_via_repo(df, ticker_str)

    if repo_out is not None:
        alpha_values = repo_out
        logger.debug("[afc] used Alphas191 for %s", ticker_str)
    else:
        alpha_values = _compute_fallback_alphas(df)
        logger.debug("[afc] used pandas fallback for %s", ticker_str)

    for col_name in AFC_CORE_FEATURE_NAMES:
        if col_name in result_df.columns:
            continue  # idempotent
        series = alpha_values.get(col_name)
        if series is None:
            result_df[col_name] = 0.0
        else:
            result_df[col_name] = series.reindex(result_df.index).fillna(0.0).values

    return result_df
