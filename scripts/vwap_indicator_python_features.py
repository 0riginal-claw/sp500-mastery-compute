"""vwap_indicator_python_features.py — VWAP + std-dev bands via marcos99b/vwap-indicator-python (STUB).

TODO: wire into v10 / Mythos pipeline.
Source repo: https://github.com/marcos99b/vwap-indicator-python (MIT).
Clone path: AI-Tools/repos-claude-clones/vwap-indicator-python

Look-ahead safety: VWAP and its std-dev are cumulative-from-session-open;
each value at bar i uses only bars [session_open, i]. The "current bar"
component is removed via .shift(1) before merge with labels. Bands are
±1σ/±2σ of typical-price weighted by volume.

Estimated features added per ticker: ~7 columns
(vwap, vwap_dev_z, vwap_above_upper_2sigma_flag, vwap_below_lower_2sigma_flag,
vwap_band_width, vwap_position_in_bands, vwap_dist_from_vwap_pct).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_session_vwap(df: pd.DataFrame, group: pd.Series, bands: tuple = (1.0, 2.0)) -> pd.DataFrame:
    """Compute VWAP + ±N-sigma bands per session (group). Pure pandas; no look-ahead."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)
    pv = typical * vol
    # cumulative within-group sums (cumsum + groupby; pandas guarantees forward-only)
    cum_pv = pv.groupby(group).cumsum()
    cum_vol = vol.groupby(group).cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)
    # rolling-within-session sigma of (typical - vwap)
    diff_sq = (typical - vwap) ** 2 * vol
    cum_diff_sq = diff_sq.groupby(group).cumsum()
    var = cum_diff_sq / cum_vol.replace(0, np.nan)
    sigma = np.sqrt(var)
    out = pd.DataFrame({"vwap": vwap, "vwap_sigma": sigma}, index=df.index)
    for n in bands:
        out[f"vwap_upper_{n:g}sigma"] = vwap + n * sigma
        out[f"vwap_lower_{n:g}sigma"] = vwap - n * sigma
    return out


def add_vwap_indicator_python_features(
    df: pd.DataFrame,
    ticker: str,
    session_freq: str = "D",
    bands: tuple = (1.0, 2.0),
) -> pd.DataFrame:
    """Add VWAP + std-dev band features for `ticker`.

    Args:
        df: DataFrame with high, low, close, volume + a DatetimeIndex
            (or 'date' column).
        ticker: ticker symbol.
        session_freq: pandas freq for grouping sessions ('D' = daily).
        bands: tuple of sigma multipliers to emit.

    Notes:
        - For daily bars, session = each row (vwap = typical price). The
          residual band columns then become typical_price ± N*std-roll.
          So for daily data we ALSO add a rolling-N=20 std band variant.
        - .shift(1) on all output columns.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "date" in out.columns:
            out = out.set_index(pd.to_datetime(out["date"]))
        else:
            return out
    # Group key for "session"
    grp = out.index.to_series().dt.to_period(session_freq)
    vwap_df = _compute_session_vwap(out, grp, bands=bands)
    # For daily-bar input (1 bar per session), the cumulative VWAP == typical
    # price and sigma==0; supplement with rolling 20-bar VWAP-style stats.
    if (vwap_df["vwap_sigma"].fillna(0) == 0).all():
        typical = (out["high"] + out["low"] + out["close"]) / 3.0
        vol = out["volume"].astype(float)
        pv = typical * vol
        roll_pv = pv.rolling(20).sum()
        roll_vol = vol.rolling(20).sum()
        vwap20 = roll_pv / roll_vol.replace(0, np.nan)
        diff_sq = ((typical - vwap20) ** 2 * vol).rolling(20).sum()
        sigma20 = np.sqrt(diff_sq / roll_vol.replace(0, np.nan))
        vwap_df = pd.DataFrame({"vwap": vwap20, "vwap_sigma": sigma20}, index=out.index)
        for n in bands:
            vwap_df[f"vwap_upper_{n:g}sigma"] = vwap20 + n * sigma20
            vwap_df[f"vwap_lower_{n:g}sigma"] = vwap20 - n * sigma20
    # Derived features
    typical_now = (out["high"] + out["low"] + out["close"]) / 3.0
    out["vwap"] = vwap_df["vwap"]
    out["vwap_dev_z"] = (typical_now - vwap_df["vwap"]) / vwap_df["vwap_sigma"].replace(0, np.nan)
    out["vwap_dist_pct"] = (typical_now - vwap_df["vwap"]) / vwap_df["vwap"].replace(0, np.nan)
    for n in bands:
        out[f"vwap_above_upper_{n:g}sigma_flag"] = (typical_now > vwap_df[f"vwap_upper_{n:g}sigma"]).astype(int)
        out[f"vwap_below_lower_{n:g}sigma_flag"] = (typical_now < vwap_df[f"vwap_lower_{n:g}sigma"]).astype(int)
    out["vwap_band_width_2sigma"] = (
        vwap_df.get(f"vwap_upper_{bands[-1]:g}sigma", pd.Series(index=out.index, dtype=float))
        - vwap_df.get(f"vwap_lower_{bands[-1]:g}sigma", pd.Series(index=out.index, dtype=float))
    )
    # position in [lower-2σ, upper-2σ] band, 0..1
    upper2 = vwap_df.get(f"vwap_upper_{bands[-1]:g}sigma")
    lower2 = vwap_df.get(f"vwap_lower_{bands[-1]:g}sigma")
    if upper2 is not None and lower2 is not None:
        denom = (upper2 - lower2).replace(0, np.nan)
        out["vwap_position_in_bands"] = ((typical_now - lower2) / denom).clip(0, 1)
    # .shift(1) on every new column
    new_cols = [c for c in out.columns if c.startswith("vwap_") or c == "vwap"]
    out[new_cols] = out[new_cols].shift(1)
    return out


if __name__ == "__main__":
    print("TODO: wire vwap_indicator_python_features into v10 microstructure layer.")
